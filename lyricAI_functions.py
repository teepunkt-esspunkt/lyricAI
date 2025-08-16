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


ARTISTS = [
               
                ]   # list of artist names

script_dir = os.path.dirname(os.path.abspath(__file__))

AlbumLinkOutputFolder = "albumLinks"
AlbumLinkOutputDir = os.path.join(script_dir, AlbumLinkOutputFolder)
AlbumLinkOutputFile = "allAlbumLinks.txt"

LyricLinkOutputFolder = "lyricLinks"
LyricLinkOutputDir = os.path.join(script_dir, LyricLinkOutputFolder)
LyricLinkOutputFile = "AllLyricLinks.txt"

SCROLL_PAUSE = 1.2
MAX_SCROLLS = 50

# options = Options()
# options.add_argument("--headless")
# service = Service(GeckoDriverManager().install())
# driver = webdriver.Firefox(service=service, options=options)

def ensure_output_dir(output_dir):
    out = os.path.join(script_dir, output_dir)
    os.makedirs(out, exist_ok=True)
    return out

#to deprecate combineAlbumLinks.py
def combine_all(input_dir, output_file="combined.txt"):
    content = ""
    txt_files = [x for x in os.listdir(input_dir) if x.endswith(".txt") and x !=output_file]
    for file in txt_files:
        file_path = os.path.join(input_dir, file)
        with open(file_path, "r", encoding="utf-8") as links:
            content += clean_lines(links)
    out_path = os.path.join(script_dir, output_file)
    with open(out_path, "w", encoding="utf-8") as outfile:
        outfile.write(content)
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

