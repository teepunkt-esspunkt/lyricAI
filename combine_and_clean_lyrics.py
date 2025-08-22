import re, os
import lyricAI_functions as L


_BRACKET_RE = re.compile(r"\[[^\]\n]+\]")  # inline [ ... ] anywhere on a line
_BRACKET_BLOCK_RE = re.compile(r"\[[^\]]*\]", re.DOTALL)  # match [ ... ] across lines
_BRACKET_INLINE_RE = re.compile(r"\[[^\]\n]+\]")          # inline single-line

def _strip_brackets(text: str) -> str:
    # Remove multi-line [ ... ] blocks first
    txt = _BRACKET_BLOCK_RE.sub("", text)
    # Then remove any inline [ ... ]
    txt = _BRACKET_INLINE_RE.sub("", txt)
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
    out_path = os.path.join(input_dir, output_file)

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

                # drop first line if requested
                if skip_first_line:
                    raw_text = "\n".join(raw_text.splitlines()[1:])

                # remove multi-line and inline [ ... ] blocks
                raw_text = _strip_brackets(raw_text)

                # now split back into lines and clean
                cleaned = []
                for line in raw_text.splitlines():
                    s = line.replace("\u200b", "").strip()
                    if not keep_line(s):
                        continue
                    cleaned.append(s)
                
                # collapse multiple blanks
                out_lines, blank = [], 0
                for l in cleaned:
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

def main():
    out_path = combine_lyrics_corpus(
        input_dir=L.LyricsOutputDir,
        output_file=L.LyricsOutputFile,  # usually "combined.txt"
        skip_first_line=True,
    )
    print(f"[OK] Clean training corpus -> {out_path}")

if __name__ == "__main__":
    main()
