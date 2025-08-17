# combine_and_clean_lyrics.py

import os
import re
import lyricAI_functions

# --- tuning knobs ---

SECTION_KEYWORDS = (
    "intro","outro","interlude","bridge","hook","chorus",
    "refrain","kehrreim","verse","vers","strophe","part",
    "pre-chorus","post-chorus","prehook","pre-hook","posthook","post-hook",
)

CONTRIB_LINE   = re.compile(r"^\s*\d+\s+contributors?\b", re.IGNORECASE)
RELEASE_LINE   = re.compile(r"^\s*(veröffentlicht|veroeffentlicht|erschien|release date)\b", re.IGNORECASE)
COPYRIGHT_LINE = re.compile(r"^\s*(©|copyright)\b", re.IGNORECASE)

META_TAG = re.compile(
    r"""^\[
            .*?\b(
                sample|samples|scratch|scratches|cut|cuts|skit|instrumental|instrumentals|
                songtext\s+zu|voice\s*sample|filmsampler|film\s*sampler|interview|
                erzähler|erzählerin
            )\b.*?
        \]$""",
    re.IGNORECASE | re.VERBOSE
)

URL_LINE = re.compile(r"^\s*(https?://|www\.)", re.IGNORECASE)

ALLOWED_SECTION_TAG = re.compile(
    rf"""^\[
            (?:.*?\b(?:{'|'.join(SECTION_KEYWORDS)})\b.*? | \s*\d+\.?\s*(?:{'|'.join(SECTION_KEYWORDS)})\b.*?)
        \]$""",
    re.IGNORECASE | re.VERBOSE
)

# NOISE_CONTAINS = (
#     "contributors",
#     "contributor",            # NEW
#     "you might also like",
#     "embed",
#     "genius romanizations",
#     "genius translations",
#     "transliteration",
#     "all rights reserved",
#     "songtext zu",            # de helper
#     "videobeschreibung",      # de helper
#     "hidden-track",           # de/en helper
#     "hidden track",           # de/en helper
#     "snippet",                # de/en helper
#     "veröffentlicht", "veroeffentlicht",  # de publish words
# )
CREDIT_PREFIXES = (
    "produced by",
    "producer:",
    "written by",
    "writer:",
    "composed by",
    "composer:",
    "recorded at",
    "recorded by",
    "engineer:",
    "mixing engineer",
    "mastered by",
    "label:",
    "release date:",
    "feat:",
)

URL_LINE = re.compile(r"^\s*(https?://|www\.)", re.IGNORECASE)
VERY_LONG = 600  # drop absurdly long single lines

DIVIDER = "=" * 50  # must match what you used when saving blocks
HEADER = re.compile(r"^\{.*?\}$")  # your {Title - Album - Year}

# --- cleaning logic ---

def split_at_first_section_tag(lines: list[str]) -> tuple[list[str], list[str]]:
    """
    If a REAL section tag ([Verse]/[Part]/[Hook]/...) appears, cut everything before it.
    If only custom labels like [Jamais-Vu] exist (not a section), do NOT cut.
    """
    for i, line in enumerate(lines):
        if ALLOWED_SECTION_TAG.match(line.strip()):
            return lines[i:], lines[:i]
    return lines, []  # no real section tag → keep all

def is_noise_line(line: str) -> bool:
    l = line.strip()
    if not l:
        return False
    if URL_LINE.search(l):
        return True
    if len(l) > VERY_LONG:
        return True

    # anchored admin lines
    if CONTRIB_LINE.match(l):
        return True
    if RELEASE_LINE.match(l):
        return True
    if COPYRIGHT_LINE.match(l):
        return True

    # drop ONLY bracketed meta; keep other bracketed labels
    if l.startswith("[") and l.endswith("]") and META_TAG.match(l):
        return True

    # no fuzzy keyword dropping
    return False

BRACKET_LINE = re.compile(r"^\[.*\]$")

def split_at_first_marker(lines: list[str]) -> tuple[list[str], list[str]]:
    """
    Marker = (a) REAL section tag, or (b) any NON-META bracketed tag.
    Handles multi-line tags like:
        [Part 1: Lokikzz &
        Prezident
        ]
    """
    n = len(lines)
    i = 0
    while i < n:
        s = lines[i].strip()

        # single-line [ ... ]
        if s.startswith("[") and s.endswith("]"):
            if not META_TAG.match(s):       # keep non-meta tags as markers
                return lines[i:], lines[:i]
            i += 1
            continue

        # multi-line [ ... \n ... \n ... ]
        if s.startswith("[") and not s.endswith("]"):
            j = i + 1
            while j < n and not lines[j].strip().endswith("]"):
                j += 1
            if j < n:
                tag_text = " ".join(x.strip() for x in lines[i:j+1])
                if not META_TAG.match(tag_text):
                    return lines[i:], lines[:i]
                i = j + 1
                continue

        # real section tag without brackets edge-cases (kept backward-compat)
        if ALLOWED_SECTION_TAG.match(s):
            return lines[i:], lines[:i]

        i += 1

    return lines, []  # no marker: keep all

SKIPPED_LOG = os.path.join(lyricAI_functions.script_dir, "skipped_lines.txt")

def log_skipped(heading: str, lines: list[str]):
    if not lines:
        return
    with open(SKIPPED_LOG, "a", encoding="utf-8") as log:
        log.write(f"\n=== {heading} ===\n")
        for s in lines:
            log.write(s + "\n")
        log.write("\n" + "="*60 + "\n\n")

def clean_block(body: str, heading: str) -> str:
    lines = body.splitlines()

    lines, preface_cut = split_at_first_marker(lines)
    if preface_cut:
        log_skipped(heading, preface_cut)

    kept, skipped = [], []
    last_blank = False
    for raw in lines:
        line = raw.rstrip()
        if is_noise_line(line):
            skipped.append(line); continue
        if not line.strip():
            if last_blank:
                skipped.append(line); continue
            last_blank = True
            kept.append("")
            continue
        last_blank = False
        kept.append(line)

    while kept and kept[0] == "":
        skipped.append(kept.pop(0))
    while kept and kept[-1] == "":
        skipped.append(kept.pop())

    if skipped:
        log_skipped(heading, skipped)

    return "\n".join(kept)

# --- parser for your saved structure ---

def iter_blocks_from_file(path: str):
    """
    Yields (header, body) for each block in a file that uses:
      {Title - Album - Year}
      <blank>
      lyrics...
      <blank>
      ==================================================
      <blank>
    """
    with open(path, "r", encoding="utf-8") as f:
        buf_header = None
        buf_lines = []
        for line in f:
            s = line.rstrip("\n")
            if buf_header is None:
                if HEADER.match(s.strip()):
                    buf_header = s.strip()
                # else ignore leading noise
                continue
            # we are inside a block
            if s.strip() == DIVIDER:
                yield (buf_header, "\n".join(buf_lines).strip())
                buf_header, buf_lines = None, []
            else:
                buf_lines.append(s)
        # handle file not ending with divider (just in case)
        if buf_header is not None and buf_lines:
            yield (buf_header, "\n".join(buf_lines).strip())

def main():
    src_root = lyricAI_functions.LyricsOutputDir  # e.g., script_dir/Lyrics
    if not os.path.isdir(src_root):
        print(f"[WARN] Lyrics folder not found: {src_root}")
        return

    out_path = os.path.join(lyricAI_functions.script_dir, "AllLyrics_Clean.txt")
    seen_headers = set()
    kept, skipped = 0, 0

    with open(out_path, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(src_root):
            for name in sorted(files):
                if not name.endswith(".txt"):
                    continue
                fp = os.path.join(root, name)
                for header, body in iter_blocks_from_file(fp):
                    # de-dupe by header
                    if header in seen_headers:
                        continue
                    seen_headers.add(header)

                    cleaned = clean_block(body, header)
                    # optional: skip super-short blocks

                    out.write(header + "\n\n")
                    out.write(cleaned + "\n\n")
                    out.write(DIVIDER + "\n\n")
                    kept += 1

    print(f"[OK] Combined into: {out_path}")
    print(f"    Blocks kept: {kept} | Skipped as noise/too short: {skipped}")

if __name__ == "__main__":
    main()
