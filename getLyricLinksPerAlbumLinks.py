import re
import os
from bs4 import BeautifulSoup
import lyricAI_functions
import time

LINK_PATTERN = re.compile(r"https://genius\.com/[^\"?#]+-lyrics")
output_dir = lyricAI_functions.LyricLinkOutputDir
input_dir = lyricAI_functions.AlbumLinkOutputDir
content = ""

def safe(s):
    return re.sub(r"[^\w\-_. ]+", "_", s).strip()

def parse_artist_album_from_album_url(url):
    """
    expects: https://genius.com/albums/<ArtistSlug>/<AlbumSlug>
    returns (artist_slug, album_slug) -- slugs, not display names
    """
    parts = url.split("/")
    # ... 0  1   2        3       4           5
    # ["https:", "", "genius.com", "albums", "<ArtistSlug>", "<AlbumSlug>", ...]
    if len(parts) >= 6 and parts[3] == "albums":
        return parts[4], parts[5]
    return "UnknownArtist", "UnknownAlbum"

def save_album_links(artist_slug, album_slug, links):
    # lyricLinks/<ArtistSlug>/ exists (under script dir via ensure_output_dir)
    artist_dir = lyricAI_functions.ensure_output_dir(os.path.join(lyricAI_functions.LyricLinkOutputDir, safe(artist_slug)))
    out_path = os.path.join(artist_dir, f"{safe(album_slug)}_lyric_links.txt")
    # write deduped, one per line
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
    links = []
    # Only find links inside the tracklist (chart_row-content)
    for a in soup.select("div.chart_row-content a[href]"):
        href = (a.get["href"] or "").strip()
        if LINK_PATTERN.match(href):
            links.append(href.strip())

    # Deduplicate and preserve order
    seen = set()
    ordered_links = []
    for href in links:
        if href not in seen:
            seen.add(href)
            ordered_links.append(href)

    return ordered_links

def main():
    combined_path = lyricAI_functions.combine_all(lyricAI_functions.AlbumLinkOutputDir, lyricAI_functions.AlbumLinkOutputFile)
    print(f"[OK] Using combined album list: {combined_path}")

    with open(combined_path, "r", encoding="utf-8") as f:
        album_urls = [line.strip() for line in f if line.strip()]
    
    driver = lyricAI_functions.make_driver(headless=lyricAI_functions.HEADLESS)
    all_lyric_links = []
    try:
        for i, url in enumerate(album_urls, 1):
            print(f"[{i}/{len(album_urls)}] {url}")
            artist_slug, album_slug = parse_artist_album_from_album_url(url)
            links = get_song_links(driver, url)
            save_album_links(artist_slug, album_slug, links)
            time.sleep(0.3)
            #all_lyric_links.extend(get_song_links(driver, url))
    finally:
        driver.quit()
        
    out_dir = lyricAI_functions.ensure_output_dir(lyricAI_functions.LyricLinkOutputFolder)
    out_path = os.path.join(out_dir, lyricAI_functions.LyricLinkOutputFile)  # "AllLyricLinks.txt"
    seen = set()
    with open(out_path, "w", encoding="utf-8") as out:
        for h in all_lyric_links:
            if h not in seen:
                seen.add(h)
                out.write(h + "\n")

    print(f"[OK] Saved {len(seen)} lyric links -> {out_path}")

if __name__ == "__main__":
    main()
