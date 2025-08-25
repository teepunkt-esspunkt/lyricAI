import re, os
import lyricAI_functions as L


_BRACKET_BLOCK_RE = re.compile(r"\[[^\]]*\]", re.DOTALL)  # match [ ... ] across lines
_BRACKET_INLINE_RE = re.compile(r"\[[^\]\n]+\]")          # inline single-line
_REPEAT_TAG_RE = re.compile(r"\(\s*(x?\d+|\d+x)\s*\)", re.IGNORECASE)
TITLE_LYRICS_RE = re.compile(r'^\s*.+\s+lyrics\s*$', re.IGNORECASE)


_META_RE = re.compile(
    r"""
    \b(
        contributors? |
        lyrics$ | songtext |
        produced\s+by | prod\. |
        album | mixtape | single | tracklist |
        release\s+date | label |
        embed | credits? | genius |
        translation | romanization | romanisierung |
        informationen\s+zum | musikvideo |
        auf\s+deutsch | aus\s+dem\s+album
    )\b
    """,
    re.IGNORECASE | re.VERBOSE
)

_BRACKET_TAG_RE = re.compile(r'^\s*\[[^\]]+\]\s*$', re.MULTILINE)

def _looks_like_lyric_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if len(s) > 120:                     # prose/blurbs tend to be long
        return False
    if _META_RE.search(s):               # obvious metadata words
        return False
    if re.search(r'https?://|\bwww\.', s, flags=re.IGNORECASE):
        return False
    # must have a few words, mostly letters
    words = re.findall(r"\w+", s, flags=re.UNICODE)
    if len(words) < 3:
        return False
    letters = sum(ch.isalpha() for ch in s)
    digits  = sum(ch.isdigit()  for ch in s)
    if letters < digits * 2:
        return False
    return True

def strip_genius_preamble(text: str):
    """
    Remove front-matter like 'Contributors', 'TITLE Lyrics', blurbs, etc.
    Returns a tuple: (cleaned_text, removed_text)
    """
    removed_parts = []

    # 0) If top line is "Title Lyrics", drop JUST that line (not the whole block)
    lines = text.splitlines()
    if lines and TITLE_LYRICS_RE.match(lines[0]):
        removed_parts.append(lines[0])
        lines = lines[1:]
        text = "\n".join(lines)

    # 1) Jump to first [Section] ONLY if it's truly early AND nothing lyric-like precedes it
    m = _BRACKET_TAG_RE.search(text)
    if m:
        before = text[:m.start()]
        before_lines = [l for l in before.splitlines() if l.strip()]
        # count lyric-ish lines before the tag
        lyricish_before = sum(1 for l in before_lines if _looks_like_lyric_line(l))
        if len(before_lines) <= 3 and lyricish_before == 0:
            removed_parts.append(before.strip())
            return text[m.start():].lstrip(), "\n".join(p for p in removed_parts if p).strip()

    # 2) Otherwise, scan for the first pair of lyric-like lines
    lines = text.splitlines()
    n = len(lines)
    start_idx = None

    for i in range(n - 1):
        if _looks_like_lyric_line(lines[i]) and _looks_like_lyric_line(lines[i + 1]):
            start_idx = i
            break

    if start_idx is None:
        for i in range(n - 2):
            if _looks_like_lyric_line(lines[i]) and not lines[i+1].strip() and _looks_like_lyric_line(lines[i+2]):
                start_idx = i
                break

    if start_idx is not None:
        removed_parts.append("\n".join(lines[:start_idx]).strip())
        return "\n".join(lines[start_idx:]).lstrip(), "\n".join(p for p in removed_parts if p).strip()

    # 3) Fallback: keep everything (but still log any trivial removed parts like the title line)
    return text, "\n".join(p for p in removed_parts if p).strip()

    
def _strip_brackets_and_repeat_tags(text: str) -> str:
    # remove [ ... ] blocks
    txt = _BRACKET_BLOCK_RE.sub("", text)
    txt = _BRACKET_INLINE_RE.sub("", txt)
    # remove (2x), (x2), etc.
    txt = _REPEAT_TAG_RE.sub("", txt)
    return txt

def combine_lyrics_corpus(
    input_dir: str,
    output_file: str = "combined.txt",
    skip_first_line: bool = True,
) -> str:
    """
    Build a cleaned training corpus from Lyrics/<Artist>/<Album>/<Song>.txt
    - drop first line (often title/meta)
    - remove ALL [bracketed] tags (inline + standalone)
    - drop header/metadata lines (… Lyrics, Songtext zu …, Informationen …, etc.)
    - strip boilerplate; collapse blanks
    """
    out_path = os.path.join(L.script_dir, output_file)

    def _fix_split_parens(text: str) -> str:
    # merge cases where a "(" or ")" is alone on its own line
        text = re.sub(r"\(\s*\n\s*", "(", text)   # join open paren with next line
        text = re.sub(r"\n\s*\)", ")", text)      # join closing paren with previous line
        return text


    def keep_line(s: str) -> bool:
        low = s.lower()
        if not s: return False
        # headers / metadata lines to drop
        if low.endswith(" lyrics"): return False
        if low.startswith(("songtext zu", "informationen zum", "einleitung", "intro]")): return False
        if low.startswith(("der track ist", "informationen zum musikvideo")): return False
        if low.startswith(("sample:", "filmausschnitt:", "outro", "hook", "part ")): return False
        if low in ("read more", "you might also like"): return False
        if s.endswith("Embed"): return False
        return True

    with open(out_path, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(input_dir):
            for name in sorted(files):
                if not name.endswith(".txt") or name == os.path.basename(out_path):
                    continue
                fpath = os.path.join(root, name)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except Exception as e:
                    print(f"[WARN] Skipped {fpath}: {e}")
                    continue

                # drop first line if requested
                if skip_first_line and lines:
                    lines = lines[1:]

                raw_text = "".join(lines)

                # # drop first line if requested
                # if skip_first_line:
                #     raw_text = "\n".join(raw_text.splitlines()[1:])
                raw_text = _fix_split_parens(raw_text)
                # remove ALL [ ... ] blocks (multi-line and inline)
                cleaned, removed = strip_genius_preamble(raw_text)

                if removed:
                    with open(os.path.join(L.script_dir,"removed_preamble.txt"), "a", encoding="utf-8") as f:
                        f.write(f"=== REMOVED from {name} ===\n")
                        f.write(removed + "\n\n")

                cleaned = _strip_brackets_and_repeat_tags(cleaned)
                # join line -1 if line starts with ", [Character]"

                # remove ALL [ ... ] blocks (multi-line and inline)

                # raw_text = _strip_brackets_and_repeat_tags(raw_text)

                # join line -1 if line starts with ", [Character]"
                # remove 2x / 3x (without brackets)
                
                # now split back into lines and clean
                kept_lines = []
                for line in cleaned.splitlines():
                    s = line.replace("\u200b", "").strip()
                    if not keep_line(s):
                        continue
                    kept_lines.append(s)
                kept_lines = collapse_repeats(kept_lines, max_consecutive=2, max_per_song=3, window=8)

                # collapse multiple blanks
                out_lines, blank = [], 0
                for l in kept_lines:
                    if l == "":
                        blank += 1
                        if blank <= 1:
                            out_lines.append("")
                    else:
                        blank = 0
                        out_lines.append(l)

                body = "\n".join(out_lines).strip()
                if body:
                    out.write(body + "\n\n")

    return out_path

def collapse_repeats(lines, max_consecutive=2, max_per_song=3, window=8):
    out, consec_count = [], 0
    seen_counts = {}
    recent = []

    prev = None
    for s in lines:
        if not s:  # keep blanks as-is; they break consecutive runs
            out.append(s)
            prev, consec_count = None, 0
            recent.clear()
            continue

        # total cap per song
        cnt = seen_counts.get(s, 0)
        if cnt >= max_per_song:
            continue

        # consecutive cap
        if prev is not None and s == prev:
            consec_count += 1
            if consec_count >= max_consecutive:
                continue
        else:
            consec_count = 0

        # sliding window de-echo (optional)
        if window and s in recent:
            # allow if we haven't hit per-song cap yet but avoid quick echoes
            if cnt >= max_per_song - 1:
                continue

        out.append(s)
        seen_counts[s] = cnt + 1
        prev = s

        # update recent window
        recent.append(s)
        if len(recent) > window:
            recent.pop(0)

    return out

def main():
    out_path = combine_lyrics_corpus(
        input_dir=L.LyricsOutputDir,
        output_file=L.LyricsOutputFile,  # usually "combined.txt"
        skip_first_line=True,
    )
    print(f"[OK] Clean training corpus -> {out_path}")

if __name__ == "__main__":
    main()
