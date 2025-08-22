import os
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


HEADLESS = True # True = hide browser window

EXTRA_ARTISTS = [
    # "Some Artist",
    # "Another Artist"
]

script_dir = os.path.dirname(os.path.abspath(__file__))

AlbumLinkOutputFolder = "albumLinks"
AlbumLinkOutputDir = os.path.join(script_dir, AlbumLinkOutputFolder)

AlbumLinkOutputFile = "allAlbumLinks.txt"
LyricLinkOutputFolder = "lyricLinks"
LyricLinkOutputDir = os.path.join(script_dir, LyricLinkOutputFolder)

LyricLinkOutputFile = "AllLyricLinks.txt"
LyricsOutputFolder = "Lyrics"
LyricsOutputDir = os.path.join(script_dir, LyricsOutputFolder)

LyricsOutputFile = "combined.txt"

prompt_file = os.path.join(script_dir, "prompt.txt")

if os.path.exists(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        PROMPT = f.read().strip() or "So"
else: 
    PROMPT = "So"

SCROLL_PAUSE = 1.2
MAX_SCROLLS = 50

ARTIST_FILE = os.path.join(script_dir, "artistsList.txt")

def load_artists(path=ARTIST_FILE, extra=None):
    """Load artists from file + optional manual list."""
    artists = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            artists = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        print(f"[WARN] Artist file not found: {path}")
    if extra:
        artists.extend(extra)
    # dedupe, preserve order
    seen, ordered = set(), []
    for a in artists:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    return ordered

ARTISTS = load_artists(extra=EXTRA_ARTISTS)


def ensure_output_dir(output_dir):
    out = os.path.join(script_dir, output_dir)
    os.makedirs(out, exist_ok=True)
    return out

import re
import json

def normalize_url(u: str) -> str:
    return u.split("#", 1)[0].split("?", 1)[0].rstrip("/")

def combine_and_clean_links(input_dir: str,
                            output_file: str,
                            keep_pattern: re.Pattern | None = None,
                            drop_if_contains: tuple[str, ...] = ("instrumental",)):
    """
    Combine all .txt files under input_dir into output_file (at script_dir),
    apply cleaning:
      - drop lines containing any of drop_if_contains (case-insensitive)
      - normalize URLs (strip query/hash/trailing slash)
      - keep only lines matching keep_pattern (if provided)
      - de-dupe while preserving order
    Returns absolute path to output_file.
    """
    out_path = os.path.join(script_dir, output_file)

    def _line_ok(line: str) -> bool:
        low = line.lower()
        if any(tok in low for tok in drop_if_contains):
            return False
        return True

    seen = set()
    cleaned = []

    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if not name.endswith(".txt"):
                continue
            file_path = os.path.join(root, name)
            with open(file_path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    if not _line_ok(line):
                        continue
                    line = normalize_url(line)
                    if keep_pattern and not keep_pattern.match(line):
                        continue
                    if line not in seen:
                        seen.add(line)
                        cleaned.append(line)

    with open(out_path, "w", encoding="utf-8") as out:
        for line in cleaned:
            out.write(line + "\n")

    return out_path

def combine_all(input_dir, output_file="combined.txt"):
    out_path = os.path.join(script_dir, output_file)
    with open(out_path, "w", encoding="utf-8") as outfile:
        for root, _, files in os.walk(input_dir):
            for name in sorted(files):
                if name.endswith(".txt") and name != output_file:
                    file_path = os.path.join(root, name)
                    with open(file_path, "r", encoding="utf-8") as fh:
                        outfile.write(clean_lines(fh))
    return out_path

def clean_lines(x):
    content = ""
    for y in x:
        if y.lower().find("instrumental") == -1:
            content += y
    return content
        

# browser settings
def make_driver(headless=HEADLESS):
    opts = Options()
    if headless:
        opts.add_argument("--headless")
    opts.set_preference(
        "general.useragent.override",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0"
    )
    svc = Service(GeckoDriverManager().install())
    return webdriver.Firefox(service=svc, options=opts)

def dismiss_cookies_if_present(driver):
    selectors = [
        (By.ID, "onetrust-reject-all-handler"),  # OneTrust reject button
        (By.XPATH, "//button[contains(.,'Reject All')]"),
        (By.XPATH, "//button[contains(.,'Decline')]"),
        (By.XPATH, "//button[contains(.,'Reject')]"),
    ]
    for how, sel in selectors:
        try:
            btn = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((how, sel))
            )
            btn.click()
            time.sleep(0.5)
            return True
        except:
            pass
    return False

def scroll_to_bottom(driver, max_scrolls=MAX_SCROLLS, pause=SCROLL_PAUSE):
    last_height = driver.execute_script("return document.body.scrollHeight")
    stalls = 0
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            stalls += 1
            if stalls >= 2:
                break
        else:
            stalls = 0
        last_height = new_height

def slugify(s: str) -> str:
    return re.sub(r"[^\w\-_. ]+", "_", s).strip()

ARTIST_MAP_FILE = os.path.join(AlbumLinkOutputDir, "_artists.json")

def update_artist_map(artist_display: str):
    """Store a simple mapping slug -> display name for reuse later."""
    slug = slugify(artist_display)
    data = {}
    if os.path.exists(ARTIST_MAP_FILE):
        try:
            with open(ARTIST_MAP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[slug] = artist_display
    with open(ARTIST_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return slug
def normalize_url(u: str) -> str:
    return u.split("#", 1)[0].split("?", 1)[0].rstrip("/")

def build_provenance_map(lyric_links_root: str) -> dict[str, tuple[str, str]]:
    """
    Walk LyricLinks/<artist>/<album>_lyric_links.txt and map:
        lyric_url -> (artist_slug, album_slug)
    """
    prov = {}
    for artist in os.listdir(lyric_links_root):
        a_dir = os.path.join(lyric_links_root, artist)
        if not os.path.isdir(a_dir):
            continue
        for fname in os.listdir(a_dir):
            if not fname.endswith("_lyric_links.txt"):
                continue
            album_slug = fname[:-len("_lyric_links.txt")]
            file_path = os.path.join(a_dir, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        url = normalize_url(line.strip())
                        if url:
                            prov[url] = (artist, album_slug)
            except Exception:
                # keep going even if one file is borked
                pass
            