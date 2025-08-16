import lyricAI_functions
from bs4 import BeautifulSoup
import re
import os


def extract_lyrics(driver, url):
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

def save_lyrics(title, lyrics, album_name, year, output_file):
    safe_title = re.sub(r'[\\/*?:\"<>|]', "_", title)
    heading = f"{{{safe_title} - {album_name} - {year}}}"

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{heading}\n\n{lyrics}\n\n{'=' * 50}\n\n")

combined_path = lyricAI_functions.combine_all(lyricAI_functions.LyricLinkOutputDir, lyricAI_functions.LyricLinkOutputFile)
print(f"[OK] Using combined album list: {combined_path}")