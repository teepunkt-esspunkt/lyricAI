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

PROMPT = "Manchmal denke ich mir"

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
