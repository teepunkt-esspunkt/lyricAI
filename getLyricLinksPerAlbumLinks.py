import re
import os
from bs4 import BeautifulSoup
import lyricAI_functions
import time

LINK_PATTERN = re.compile(r"https://genius\.com/[^\"?#]+-lyrics")
content = ""

def safe(s):
    return re.sub(r"[^\w\-_. ]+", "_", s).strip()

def normalize_url(u: str) -> str:
    return u.split("#", 1)[0].split("?", 1)[0].rstrip("/")

def parse_artist_album_from_album_url(url):
    """
    expects: https://genius.com/albums/<ArtistSlug>/<AlbumSlug>
    returns (artist_slug, album_slug)
    """
    parts = url.split("/")
    if len(parts) >= 6 and parts[3] == "albums":
        return parts[4], parts[5]
    return "UnknownArtist", "UnknownAlbum"

def save_album_links(artist_slug, album_slug, links):
    artist_dir = lyricAI_functions.ensure_output_dir(
        os.path.join(lyricAI_functions.LyricLinkOutputDir, safe(artist_slug))
    )
    out_path = os.path.join(artist_dir, f"{safe(album_slug)}_lyric_links.txt")

    seen = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for h in links:
            if h not in seen:
                seen.add(h)
                f.write(h + "\n")
    print(f"[OK] {artist_slug}/{album_slug}: {len(seen)} song links -> {out_path}")
    return out_path

def get_song_links(driver, album_url):
    print(f"Loading album page: {album_url}")
    driver.get(album_url)
    lyricAI_functions.dismiss_cookies_if_present(driver)
    lyricAI_functions.scroll_to_bottom(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    raw = []
    for a in soup.select("div.chart_row-content a[href]"):
        href = (a.get("href") or "").strip()   # <-- bugfix: use a.get("href")
        if LINK_PATTERN.match(href):
            raw.append(normalize_url(href))

    # dedupe in order
    seen, out = set(), []
    for h in raw:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out

def main():
    # Build the combined ALBUM list here (from step 1 outputs)
    combined_albums = lyricAI_functions.combine_all(
        lyricAI_functions.AlbumLinkOutputDir,
        lyricAI_functions.AlbumLinkOutputFile
    )
    print(f"[OK] Using combined album list: {combined_albums}")

    with open(combined_albums, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]
    # keep only album pages & de-dupe in order
    album_urls = []
    seen = set()
    for u in raw:
        if u.startswith("https://genius.com/albums/"):
            if u not in seen:
                seen.add(u)
                album_urls.append(u)


    driver = lyricAI_functions.make_driver(headless=lyricAI_functions.HEADLESS)
    try:
        total = len(album_urls)
        for i, url in enumerate(album_urls, 1):
            print(f"[{i}/{total}] {url}")
            artist_slug, album_slug = parse_artist_album_from_album_url(url)
            links = get_song_links(driver, url)
            save_album_links(artist_slug, album_slug, links)
            time.sleep(0.2)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
