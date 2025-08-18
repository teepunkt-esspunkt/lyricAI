import os
import re
import time
from bs4 import BeautifulSoup

import lyricAI_functions as L  # central helpers & config

# =====================
# CONFIG
# =====================

LYRIC_LINKS_ROOT = L.LyricLinkOutputDir           # input folder with per-album lyric link files
LYRIC_LINKS_COMBINED = L.LyricLinkOutputFile      # e.g., "AllLyricLinks.txt"

LYRICS_OUT_ROOT = L.LyricsOutputDir               # output folder for lyrics (structured)

LYRIC_URL_RE = re.compile(r"^https://genius\.com/[^\"?#]+-lyrics/?$", re.IGNORECASE)

SLEEP_BETWEEN_PAGES = 0.2
RETRIES = 2
RETRY_WAIT = 1.0

# =====================
# HELPERS
# =====================

def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-_. ]+", "_", s).strip()

def normalize_url(u: str) -> str:
    return u.split("#", 1)[0].split("?", 1)[0].rstrip("/")

def read_and_clean_lyric_links(combined_path: str):
    """Read AllLyricLinks.txt, filter to valid lyric URLs, normalize, dedupe (order-preserving)."""
    with open(combined_path, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]

    seen = set()
    cleaned = []
    for u in raw:
        nu = normalize_url(u)
        if LYRIC_URL_RE.match(nu) and nu not in seen:
            seen.add(nu)
            cleaned.append(nu)
    return cleaned

def parse_artist_album_song(url: str, soup: BeautifulSoup):
    """
    Try to parse artist / album / song names.
    - Artist: from breadcrumb or <a href="/artists/...">
    - Album: from breadcrumb if present
    - Song: from page <h1> or URL slug
    """
    artist, album, song = "UnknownArtist", "UnknownAlbum", "Untitled"

    # Artist
    a = soup.select_one('a[href^="/artists/"]')
    if a:
        artist = a.get_text(strip=True)

    # Album (breadcrumb or meta)
    alb = soup.select_one('a[href*="/albums/"]')
    if alb:
        album = alb.get_text(strip=True)

    # Song (title h1 or slug)
    h1 = soup.find("h1")
    if h1:
        song = h1.get_text(strip=True)
    else:
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        song = slug.replace("-lyrics", "").replace("-", " ")

    return artist, album, song

def extract_lyrics_text(soup: BeautifulSoup) -> str:
    """Extract lyrics text from Genius page."""
    blocks = soup.select('div[data-lyrics-container="true"]')
    if blocks:
        text = "\n".join(div.get_text("\n", strip=False) for div in blocks)
    else:
        legacy = soup.select_one("div.lyrics")
        if legacy:
            text = legacy.get_text("\n", strip=False)
        else:
            containers = soup.select("div.Lyrics__Container-sc-1ynbvzw-6")
            text = "\n".join(c.get_text("\n", strip=False) for c in containers) if containers else ""

    # Cleanup
    cleaned_lines = []
    for line in text.splitlines():
        s = line.strip("\u200b ").rstrip()
        if not s:
            cleaned_lines.append("")
            continue
        if s.lower().startswith("you might also like"):
            continue
        if s.endswith("Embed"):
            continue
        cleaned_lines.append(s)

    # Collapse multiple blank lines
    out, blank = [], 0
    for l in cleaned_lines:
        if l == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(l)

    return "\n".join(out).strip()

def save_song_lyrics(base_root: str, artist: str, album: str, song: str, lyrics_text: str) -> str:
    artist_dir = os.path.join(base_root, safe_name(artist))
    album_dir = os.path.join(artist_dir, safe_name(album))
    os.makedirs(album_dir, exist_ok=True)

    fname = safe_name(song)[:150] + ".txt"
    path = os.path.join(album_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(lyrics_text + "\n")
    return path

# =====================
# MAIN
# =====================

def main():
    os.makedirs(LYRICS_OUT_ROOT, exist_ok=True)

    # 1) Combine all lyric-link lists under LyricLinks/ → AllLyricLinks.txt
    combined_links_path = L.combine_all(L.LyricLinkOutputDir, L.LyricLinkOutputFile)
    print(f"[OK] Combined lyric links -> {combined_links_path}")

    # 2) Read + clean + dedupe lyric URLs
    lyric_urls = read_and_clean_lyric_links(combined_links_path)
    if not lyric_urls:
        print("[WARN] No lyric URLs found. Did STEP TWO create files in LyricLinks/?")
        return
    print(f"[INFO] Fetching {len(lyric_urls)} lyric pages")

    # 3) Visit each lyric page and save lyrics into Artist/Album/Song.txt
    driver = L.make_driver(headless=L.HEADLESS)
    saved = 0
    try:
        for i, url in enumerate(lyric_urls, 1):
            print(f"[{i}/{len(lyric_urls)}] {url}")
            attempt = 0
            while True:
                try:
                    attempt += 1
                    driver.get(url)
                    L.dismiss_cookies_if_present(driver)
                    time.sleep(0.5)

                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    artist, album, song = parse_artist_album_song(url, soup)
                    lyrics = extract_lyrics_text(soup)

                    if not lyrics:
                        raise RuntimeError("No lyrics extracted (empty text)")

                    out_path = save_song_lyrics(LYRICS_OUT_ROOT, artist, album, song, lyrics)
                    print(f"  -> saved: {out_path}")
                    saved += 1
                    break
                except Exception as e:
                    if attempt <= RETRIES:
                        print(f"  ! retry {attempt}/{RETRIES} due to: {e}")
                        time.sleep(RETRY_WAIT)
                        continue
                    else:
                        print(f"[ERROR] Failed to fetch {url}: {e}")
                        break

            time.sleep(SLEEP_BETWEEN_PAGES)
    finally:
        driver.quit()

    print(f"[OK] Saved {saved} songs into {LYRICS_OUT_ROOT}/<Artist>/<Album>/")

if __name__ == "__main__":
    main()
