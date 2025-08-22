import os
import re
import time
import json
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By  # <-- proper By import

import lyricAI_functions as L  # central helpers & config

# =====================
# CONFIG
# =====================

LYRIC_LINKS_ROOT = L.LyricLinkOutputDir
LYRIC_LINKS_COMBINED = L.LyricLinkOutputFile
LYRICS_OUT_ROOT = L.LyricsOutputDir

LYRIC_URL_RE = re.compile(r"^https://genius\.com/[^\"?#]+-lyrics/?$", re.IGNORECASE)

provenance = L.build_provenance_map(LYRIC_LINKS_ROOT)

SLEEP_BETWEEN_PAGES = 0.2
RETRIES = 2
RETRY_WAIT = 1.0
INCLUDE_ABOUT = False  # set True to include full About text (expanded) in files

# =====================
# HELPERS
# =====================

def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-_. ]+", "_", s).strip()

def normalize_url(u: str) -> str:
    return u.split("#", 1)[0].split("?", 1)[0].rstrip("/")

def parse_artist_album_song(url: str, soup: BeautifulSoup):
    """
    Prefer JSON-LD; then breadcrumbs; then OG/title fallbacks.
    Returns (artist, album, song) with sensible defaults.
    """
    artist, album, song = None, None, None

    # --- JSON-LD ---
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        objs = []
        if isinstance(data, list):
            objs = data
        elif isinstance(data, dict) and "@graph" in data and isinstance(data["@graph"], list):
            objs = data["@graph"]
        else:
            objs = [data]

        for obj in objs:
            if not isinstance(obj, dict):
                continue
            t = obj.get("@type")
            if t in ("MusicRecording", "Song"):
                song = song or obj.get("name")
                by = obj.get("byArtist")
                if isinstance(by, dict):
                    artist = artist or by.get("name")
                elif isinstance(by, list) and by and isinstance(by[0], dict):
                    artist = artist or by[0].get("name")
                alb = obj.get("inAlbum")
                if isinstance(alb, dict):
                    album = album or alb.get("name")
            elif t == "MusicAlbum":
                album = album or obj.get("name")
                by = obj.get("byArtist")
                if isinstance(by, dict):
                    artist = artist or by.get("name")

    # --- Breadcrumb fallbacks ---
    if not artist:
        a = soup.select_one('nav a[href^="/artists/"], a[href^="/artists/"]')
        if a:
            artist = a.get_text(strip=True) or None
    if not album:
        alb = soup.select_one('nav a[href*="/albums/"], a[href*="/albums/"]')
        if alb:
            album = alb.get_text(strip=True) or None

    # --- OG/title fallbacks (very reliable on Genius) ---
    if not artist or not song:
        og = soup.find("meta", attrs={"property": "og:title"})
        content = (og.get("content") or "").strip() if og else ""
        # Patterns like "Artist – Song Lyrics" or "Artist - Song Lyrics"
        if " Lyrics" in content:
            core = content.replace(" Lyrics", "")
            # split on en dash, em dash, or hyphen
            for sep in (" – ", " — ", " - "):
                if sep in core:
                    maybe_artist, maybe_song = core.split(sep, 1)
                    maybe_artist = maybe_artist.strip()
                    maybe_song = maybe_song.strip()
                    if not artist and maybe_artist:
                        artist = maybe_artist
                    if not song and maybe_song:
                        song = maybe_song
                    break

    if not song:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            song = h1.get_text(strip=True)
        else:
            slug = url.rstrip("/").rsplit("/", 1)[-1]
            song = slug.replace("-lyrics", "").replace("-", " ")

    return artist or "UnknownArtist", album or "UnknownAlbum", song or "Untitled"


def expand_about_if_present(driver):
    try:
        # About section button (EN: Read More, DE: Mehr anzeigen / Weiterlesen)
        xp = (
            "//section[.//h2[contains(translate(., 'ABOUT', 'about'),'about')]]"
            "//button[contains(.,'Read More') or contains(.,'Mehr anzeigen') or contains(.,'Weiterlesen')]"
        )
        btn = driver.find_element(By.XPATH, xp)
        if btn.is_displayed():
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.3)
    except Exception:
        pass


def extract_lyrics_text(soup: BeautifulSoup) -> str:
    """Only real lyrics: modern data-lyrics-container or legacy .lyrics."""
    blocks = soup.select('div[data-lyrics-container="true"]')
    if blocks:
        text = "\n".join(div.get_text("\n", strip=False) for div in blocks)
    else:
        legacy = soup.select_one("div.lyrics")
        text = legacy.get_text("\n", strip=False) if legacy else ""

    # Cleanup boilerplate + collapse blanks
    cleaned_lines = []
    for line in text.splitlines():
        s = line.strip("\u200b ").rstrip()
        if not s:
            cleaned_lines.append("")
            continue
        low = s.lower()
        if low.startswith("you might also like") or low == "read more":
            continue
        if s.endswith("Embed"):
            continue
        cleaned_lines.append(s)

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

def extract_about_text(soup: BeautifulSoup) -> str:
    """Best-effort About extractor on expanded DOM (after expand_about_if_present)."""
    about_section = None
    for sec in soup.find_all(["section", "div"]):
        h = sec.find(["h2", "h3"])
        if h and "about" in h.get_text(strip=True).lower():
            about_section = sec
            break
    if not about_section:
        return ""
    # Remove any lyrics blocks that might appear inside About
    for bad in about_section.select('div[data-lyrics-container="true"]'):
        bad.decompose()
    txt = about_section.get_text("\n", strip=True)
    lines = [l.strip() for l in txt.splitlines() if l.strip() and l.strip().lower() != "read more"]
    return "\n".join(lines)

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

    # 1) Combine & clean lyric-link lists using the helper (keeps only proper lyric URLs)
    combined_links_path = L.combine_and_clean_links(
        input_dir=LYRIC_LINKS_ROOT,
        output_file=LYRIC_LINKS_COMBINED,
        keep_pattern=LYRIC_URL_RE,
        drop_if_contains=("instrumental",),
    )
    print(f"[OK] Clean lyric links -> {combined_links_path}")

    # 2) Read cleaned URLs
    with open(combined_links_path, "r", encoding="utf-8") as f:
        lyric_urls = [line.strip() for line in f if line.strip()]

    if not lyric_urls:
        print("[WARN] No lyric URLs found. Did STEP TWO create files in LyricLinks/?")
        return

    # 3) Fetch & save
    driver = L.make_driver(headless=L.HEADLESS)
    saved = 0
    try:
        total = len(lyric_urls)
        for i, url in enumerate(lyric_urls, 1):
            print(f"[{i}/{total}] {url}")
            attempt = 0
            while True:
                try:
                    attempt += 1
                    driver.get(url)
                    L.dismiss_cookies_if_present(driver)
                    expand_about_if_present(driver)  # <-- ensure full About if you want it
                    time.sleep(0.5)

                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    artist, album, song = parse_artist_album_song(url, soup)
                    prov = provenance.get(normalize_url(url))
                    if prov:
                        prov_artist, prov_album = prov
                        artist = prov_artist or artist
                        album  = prov_album  or album
                    lyrics = extract_lyrics_text(soup)
                    if not lyrics:
                        raise RuntimeError("No lyrics extracted (empty text)")

                    if INCLUDE_ABOUT:
                        about = extract_about_text(soup)
                        if about:
                            lyrics = "[ABOUT]\n" + about + "\n\n[LYRICS]\n" + lyrics

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
