import os
import re
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import lyricAI_functions

# ===== CONFIG =====
ARTIST_NAMES = lyricAI_functions.ARTISTS

output_folder = lyricAI_functions.AlbumLinkOutputFolder

def extract_album_links(html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("https://genius.com/albums/"):
            links.add(href)
    return sorted(links)

def fetch_albums_for_artist(driver, artist_name):
    print(f"[INFO] Searching for artist: {artist_name}")
    driver.get("https://genius.com/")
    lyricAI_functions.dismiss_cookies_if_present(driver)

    # Find search input and submit
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "q"))
    )
    search_box.clear()
    search_box.send_keys(artist_name)
    search_box.submit()

    # Click first matching artist link
    try:
        artist_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/artists/')]"))
        )
        artist_url = artist_link.get_attribute("href")
        driver.get(artist_url)
    except Exception as e:
        print(f"[WARN] Could not find artist link for {artist_name}: {e}")
        return []

    lyricAI_functions.dismiss_cookies_if_present(driver)

    # Try direct /albums page first
    albums_url = artist_url.rstrip("/") + "/albums"
    driver.get(albums_url)
    lyricAI_functions.dismiss_cookies_if_present(driver)

    # If albums page fails, click "Show all albums"
    if "404" in (driver.title or "").lower():
        driver.get(artist_url)
        try:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(@class,'full_width_button')][contains(.,'Show all albums')]"))
            )
            btn.click()
        except:
            print("[WARN] No 'Show all albums' button found.")

    # Scroll and collect
    lyricAI_functions.scroll_to_bottom(driver)
    return extract_album_links(driver.page_source)

def save_links(artist_name, links):
    out_dir = lyricAI_functions.ensure_output_dir(output_folder)
    safe = re.sub(r"[^\w\-_. ]+", "_", artist_name).strip()
    path = os.path.join(out_dir, f"{safe}_album_links.txt")
    with open(path, "w", encoding="utf-8") as f:
        for url in sorted(set(links)):
            f.write(url + "\n")
    print(f"[OK] Saved {len(links)} album links -> {path}")

def main():
    driver = lyricAI_functions.make_driver(headless=lyricAI_functions.HEADLESS)
    try:
        for name in ARTIST_NAMES:
            links = fetch_albums_for_artist(driver, name)
            save_links(name, links)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
