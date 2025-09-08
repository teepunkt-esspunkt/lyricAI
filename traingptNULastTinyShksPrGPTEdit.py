import os
import math
import time
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import lyricAI_functions

""" basically andrej karpathys script mid to late video before he switched to a dataset, with some chatgpt edits"""
# =========================
# CONFIG (tweak these here)
# =========================

# Autosave
SAVE_EVERY      = 20      # save every N steps (e.g., 10 / 20 / 50)
KEEP_LAST_N     = 2       # keep last N step-tagged checkpoints

# Choose checkpoint type
SAVE_OPTIMIZER  = False   # True = FULL resume (model + optimizer + meta) (not tested!)
SAVE_SCALER     = False   # only relevant if you use GradScaler (we don't here)

# Paths anchored to this .py file
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH       = os.path.join(SCRIPT_DIR, "checkpoint.pt")
DATA_PATH       = os.path.join(SCRIPT_DIR, "combined.txt")  # change if needed

# Device / precision
SEED            = 1337
USE_BFLOAT16    = True    # set False to use float16 (on MPS you likely want float16)
ALLOW_COMPILE   = False   # torch.compile if available

# Training data
B               = 16      # micro-batch size per step
T               = 1024    # sequence length
TOTAL_BATCH     = 524_288 # tokens per optimizer step across grad accumulation * world size

# Optimizer
LEARNING_RATE   = 6e-4
WEIGHT_DECAY    = 0.1
GRAD_CLIP       = 1.0

# LR schedule
WARMUP_STEPS    = 10
MAX_STEPS       = 9000      # total training steps

# Generation
NUM_RETURN      = 2
MAX_GEN_TOK     = 300
TOP_K           = 10
PROMPT          = lyricAI_functions.PROMPT

# Model size
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer:   int = 12
    n_head:    int = 12
    n_embd:    int = 768

MODEL_CFG = GPTConfig(block_size=T, vocab_size=50304, n_layer=12, n_head=12, n_embd=768)

# =========================
# Model components
# =========================
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,     config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte   = nn.Embedding(config.vocab_size, config.n_embd),
            wpe   = nn.Embedding(config.block_size, config.n_embd),
            h     = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f  = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # tie weights (lm_head shares weights with token embedding)
        self.lm_head.weight = self.transformer.wte.weight # type: ignore

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"seq len {T} > block size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos) # type: ignore
        for block in self.transformer.h: # type: ignore
            x = block(x)
        x = self.transformer.ln_f(x) # type: ignore
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_str, master_process=True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() <  2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device_str)
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

# =========================
# Data loader
# =========================
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, data_path):
        self.B = B; self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"DATA_PATH not found: {data_path}")
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        if master_process:
            print(f"loaded {len(self.tokens)} tokens from {data_path}")
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        N = self.tokens.numel()
        start = self.current_position
        need = B * T + 1
        end = start + need
        # circular slice so we always get exactly need tokens
        if end <= N:
            buf = self.tokens[start:end]
        else:
            first = self.tokens[start:N]
            rem = end - N
            second = self.tokens[0:rem]
            buf = torch.cat([first, second], dim=0)
        # shape into (B, T)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # advance pointer for next call, respecting multi-process striding
        stride = B * T * self.num_processes
        self.current_position = (start + stride) % N
        return x, y        

# =========================
# DDP / device setup
# =========================
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import tiktoken
enc = tiktoken.get_encoding('gpt2')

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP branch requires CUDA for now"
    init_process_group(backend='nccl')
    ddp_rank        = int(os.environ['RANK'])
    ddp_local_rank  = int(os.environ['LOCAL_RANK'])
    ddp_world_size  = int(os.environ['WORLD_SIZE'])
    device          = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process  = (ddp_rank == 0)
else:
    ddp_rank       = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = ("cuda" if torch.cuda.is_available()
              else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                    else "cpu"))
    print(f"using device: {device}")

device_type = 'cuda' if 'cuda' in device else ('mps' if device == 'mps' else 'cpu')

torch.manual_seed(SEED)
if 'cuda' in device:
    torch.cuda.manual_seed(SEED)

assert TOTAL_BATCH % (B * T * ddp_world_size) == 0, "TOTAL_BATCH must be divisible by B*T*world_size"
grad_accum_steps = TOTAL_BATCH // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size (tokens): {TOTAL_BATCH}")
    print(f"=> grad accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, data_path=DATA_PATH)

# matmul precision (no-op on CPU)
torch.set_float32_matmul_precision('high')

# =========================
# Model / optimizer
# =========================
model = GPT(MODEL_CFG)

# Optional compile (PyTorch 2+)
if ALLOW_COMPILE and hasattr(torch, 'compile'):
    try:
        model = torch.compile(model)
        if master_process: print("torch.compile enabled")
    except Exception as e:
        if master_process: print(f"compile skipped: {e}")

model.to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

optimizer = raw_model.configure_optimizers( # type: ignore
    weight_decay=WEIGHT_DECAY, learning_rate=LEARNING_RATE, device_str=device, master_process=master_process
)

# =========================
# LR schedule
# =========================
def get_lr(it):
    if it < WARMUP_STEPS:
        return LEARNING_RATE * (it + 1) / WARMUP_STEPS
    if it > MAX_STEPS:
        return LEARNING_RATE * 0.1
    decay_ratio = (it - WARMUP_STEPS) / max(1, (MAX_STEPS - WARMUP_STEPS))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (LEARNING_RATE * 0.1) + coeff * (LEARNING_RATE - LEARNING_RATE * 0.1)

amp_dtype = (torch.bfloat16 if (device_type != 'mps' and USE_BFLOAT16) else torch.float16)

# =========================
# Checkpoint helpers (LIGHT + FULL)
# =========================
from pathlib import Path
import gc

def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on most filesystems

def cpu_state_dict(module: nn.Module):
    return {k: v.detach().to('cpu') for k, v in module.state_dict().items()}

def _rotate_ckpts(base_path: Path):
    if KEEP_LAST_N:
        ckpts = sorted(base_path.parent.glob(f"{base_path.stem}_step*{base_path.suffix}"))
        for old in ckpts[:-KEEP_LAST_N]:
            try: old.unlink()
            except: pass

def save_ckpt_light(step, raw_model):
    """Model-only checkpoint + metadata; small & fast."""
    path = Path(CKPT_PATH)
    step_path = path.with_name(f"{path.stem}_step{step:06d}{path.suffix}")
    payload = {
        "model": cpu_state_dict(raw_model),
        "meta": {
            "step": step,
            "time": time.time(),
            "current_position": getattr(train_loader, "current_position", 0),
            "bt_world": (B, T, ddp_world_size),
        },
    }
    atomic_save(payload, step_path)
    atomic_save(payload, path)  # "latest"
    _rotate_ckpts(path)
    if master_process:
        size_mb = os.path.getsize(step_path) / (1024*1024)
        print(f"[ckpt] saved {step_path.name} ({size_mb:.1f} MB)")
    del payload; gc.collect()

def _optimizer_state_to_cpu(opt_state):
    for state in opt_state['state'].values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.detach().to('cpu')

def save_ckpt_full(step, raw_model, optimizer, grad_scaler=None):
    """Full-resume: model + optimizer (+ RNG + optional scaler)."""
    path = Path(CKPT_PATH)
    step_path = path.with_name(f"{path.stem}_step{step:06d}{path.suffix}")
    payload = {
        "model": cpu_state_dict(raw_model),
        "optimizer": optimizer.state_dict(),
        "meta": {
            "step": step,
            "time": time.time(),
            "dtype": str(next(raw_model.parameters()).dtype),
            "lr": [pg.get("lr", None) for pg in optimizer.param_groups],
            "grad_accum_steps": grad_accum_steps,
            "ddp_world_size": ddp_world_size,
        },
        "rng": {
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if SAVE_SCALER and grad_scaler is not None:
        payload["scaler"] = grad_scaler.state_dict()

    _optimizer_state_to_cpu(payload["optimizer"])
    atomic_save(payload, step_path)
    atomic_save(payload, path)
    _rotate_ckpts(path)

    if master_process:
        size_mb = os.path.getsize(step_path)/(1024*1024)
        print(f"[ckpt] FULL saved {step_path.name} ({size_mb:.1f} MB)")
    del payload; gc.collect()

# =========================
# Unified resume logic (supports light/full)
# =========================
start_step = 0
loaded_lr = None

if os.path.exists(CKPT_PATH):
    state = torch.load(CKPT_PATH, map_location='cpu')

    # model (supports both light payload and raw state_dict)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    raw_model.load_state_dict(state_dict, strict=True) # type: ignore

    if isinstance(state, dict) and "meta" in state:
        start_step = int(state["meta"].get("step", 0))
        loaded_lr = state["meta"].get("lr", None)
          # --- restore loader cursor if present ---
        if "current_position" in state["meta"]:
            train_loader.current_position = int(state["meta"]["current_position"])
        else:
            # fallback: advance by consumed tokens (what you already had)
            train_loader.current_position += start_step * (B * T * ddp_world_size)
    if SAVE_OPTIMIZER and isinstance(state, dict) and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    if master_process:
        print(f"Loaded checkpoint from {CKPT_PATH} (step={start_step})")

# Optional: continue exactly the same LR as in the checkpoint
if loaded_lr is not None:
    for i, pg in enumerate(optimizer.param_groups):
        if i < len(loaded_lr) and loaded_lr[i] is not None:
            pg['lr'] = loaded_lr[i]

# =========================
# Training loop (periodic saves + emergency save)
# =========================
step = start_step - 1  # predefine for safe Ctrl-C handling
try:
    for step in range(start_step, MAX_STEPS):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # type: ignore
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step()

        if 'cuda' in device:
            torch.cuda.synchronize()

        dt = time.time() - t0
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        if master_process:
            print(f"step {step:4d}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, "
                  f"norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_processed / max(dt, 1e-8):.2f}")

        # periodic save (and also at the final step)
        if master_process and SAVE_EVERY and ((step + 1) % SAVE_EVERY == 0 or step + 1 == MAX_STEPS):
            if SAVE_OPTIMIZER:
                save_ckpt_full(step + 1, raw_model, optimizer)
            else:
                save_ckpt_light(step + 1, raw_model)
            # === periodic sample generation ===
            model.eval()
            sample_prompt = PROMPT
            tokens = enc.encode(sample_prompt)
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

            with torch.no_grad():
                x = tokens
                for _ in range(MAX_GEN_TOK):
                    logits, _ = model(x)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, TOP_K, dim=-1)
                    ix = torch.multinomial(topk_probs, 1)
                    xcol = torch.gather(topk_indices, -1, ix)
                    x = torch.cat((x, xcol), dim=1)

            decoded = enc.decode(x[0].tolist())
            print(f"\n=== SAMPLE @ step {step+1} ===\n{decoded}\n")
            model.train()

except KeyboardInterrupt:
    if master_process:
        print("\n[ckpt] Ctrl-C — saving emergency checkpoint...")
        last = step if step >= 0 else 0
        if SAVE_OPTIMIZER:
            save_ckpt_full(last, raw_model, optimizer)
        else:
            save_ckpt_light(last, raw_model)
    raise

if ddp:
    destroy_process_group()

# =========================
# Generation
# =========================


model.eval()
num_return_sequences = NUM_RETURN
max_length = MAX_GEN_TOK

prompt = PROMPT
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)

torch.manual_seed(42)
if 'cuda' in device:
    torch.cuda.manual_seed(42)

x = tokens
with torch.no_grad():
    while x.size(1) < max_length:
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, TOP_K, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    decoded = enc.decode(x[i, :max_length].tolist())
    print(":", decoded)
