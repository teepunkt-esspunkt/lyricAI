import lyricAI_functions
from bs4 import BeautifulSoup
import re
import os
import time

def get_soup(driver, url: str):
    driver.get(url)
    lyricAI_functions.dismiss_cookies_if_present(driver)
    time.sleep(0.3)  # tiny settle
    return BeautifulSoup(driver.page_source, "html.parser")

def extract_title(soup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return "Unknown Title"

def extract_artist_album(soup) -> tuple[str, str]:
    # Best-effort: first /artists/ and first /albums/ link text
    artist = None
    a1 = soup.select_one("a[href*='/artists/']")
    if a1 and a1.get_text(strip=True):
        artist = a1.get_text(strip=True)

    album = None
    a2 = soup.select_one("a[href*='/albums/']")
    if a2 and a2.get_text(strip=True):
        album = a2.get_text(strip=True)

    return artist or "Unknown Artist", album or "Unknown Album"

def extract_year(soup) -> str:
    m = re.search(r"\b(19|20)\d{2}\b", soup.get_text(" ", strip=True))
    return m.group(0) if m else "____"

def extract_lyricsold(driver, url):
    print(f"Visiting {url}")
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    lyrics_blocks = soup.find_all("div", {"data-lyrics-container": "true"})
    if not lyrics_blocks:
        return None
    # Collect lines while filtering out non-lyrics content
    lines = []
    for block in lyrics_blocks:
        for line in block.stripped_strings:
            # Skip common non-lyrics patterns
            if "Contributors" in line or "Read More" in line or "Lyrics" in line:
                continue
            lines.append(line)
    lyrics_text = "\n".join(lines)
    return lyrics_text.strip()

def extract_lyrics(driver, url) -> str | None:
    driver.get(url)
    lyricAI_functions.dismiss_cookies_if_present(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # 1) Primary: modern Genius markup
    blocks = soup.select('div[data-lyrics-container="true"]')

    # 2) Fallback: older markup used by some pages
    if not blocks:
        blocks = soup.select("div[class*='Lyrics__Container']")

    if not blocks:
        return None

    lines = []
    for blk in blocks:
        # Skip any “Read More” button or expandable UI within the block
        for el in blk.select("[data-read-more], [aria-expanded], button"):
            el.decompose()

        # Convert <br> to newlines so we don’t lose line breaks
        for br in blk.find_all("br"):
            br.replace_with("\n")

        # Extract text preserving inline newlines
        text = blk.get_text("\n", strip=False)

        # Normalize Windows/mac line breaks and collapse \r
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split to lines and strip trailing spaces, but keep empty lines
        for raw in text.split("\n"):
            ln = raw.rstrip()

            # Keep custom bracket tags like [Jamais-Vu], [Hook: ...], etc.
            # (Do NOT filter here—save filtering for your combine_and_clean step)
            if ln is not None:
                lines.append(ln)

    # Trim leading/trailing blank lines produced by container joins
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    # Join with single newlines
    lyrics = "\n".join(lines).strip()
    return lyrics or None

def extract_lyrics_from_soup(soup) -> str | None:
    blocks = soup.select('div[data-lyrics-container="true"]')
    if not blocks:
        blocks = soup.select("div[class*='Lyrics__Container']")
    if not blocks:
        return None

    lines = []
    for blk in blocks:
        for el in blk.select("[data-read-more], [aria-expanded], button"):
            el.decompose()
        for br in blk.find_all("br"):
            br.replace_with("\n")
        text = blk.get_text("\n", strip=False).replace("\r\n","\n").replace("\r","\n")
        lines.extend(raw.rstrip() for raw in text.split("\n"))

    while lines and not lines[0].strip(): lines.pop(0)
    while lines and not lines[-1].strip(): lines.pop()
    return ("\n".join(lines)).strip() or None


def safe(s: str) -> str:
    return re.sub(r'[\\/*?:\"<>|]', "_", s).strip()

def save_song_into_album_file(artist: str, album: str, title: str, year: str, lyrics: str) -> str:
    # Write to your new Lyrics root
    # Make sure lyricAI_functions.py defines: LyricsOutputDir = os.path.join(script_dir, "Lyrics")
    artist_dir = lyricAI_functions.ensure_output_dir(os.path.join(lyricAI_functions.LyricsOutputDir, safe(artist)))
    out_path = os.path.join(artist_dir, f"{safe(album)}.txt")
    heading = f"{{{title} - {album} - {year}}}"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(heading + "\n\n")
        f.write(lyrics + "\n\n")
        f.write("=" * 50 + "\n\n")
    return out_path



def main():
    combined_path = lyricAI_functions.combine_all(lyricAI_functions.LyricLinkOutputDir, lyricAI_functions.LyricLinkOutputFile)
    print(f"[OK] Using combined link list: {combined_path}")
    with open(combined_path, "r", encoding="utf-8") as fh:
        urls = [u.strip() for u in fh if u.strip()]

    driver = lyricAI_functions.make_driver(headless=lyricAI_functions.HEADLESS)
    seen = set()
    written, skipped = 0, 0
    try:
        total = len(urls)
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{total}] {url}")
            if url in seen:
                continue
            seen.add(url)
            soup = get_soup(driver, url)
            title = extract_title(soup)
            artist, album = extract_artist_album(soup)
            year = extract_year(soup)
            lyrics = extract_lyrics_from_soup(soup)
            if not lyrics:
                print("   [!] No lyrics container found")
                skipped += 1
                continue

            save_song_into_album_file(artist, album, safe(title), year, lyrics)
            written += 1
            time.sleep(0.2)  # gentle pacing
    finally:
        driver.quit()

    print(f"output root: {lyricAI_functions.LyricsOutputDir}")


if __name__ == "__main__":
    main()