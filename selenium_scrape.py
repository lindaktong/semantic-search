from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

def fetch_page_with_selenium(url):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(5)  # Wait for the page to load
    page_source = driver.page_source
    driver.quit()
    return page_source

def main():
    url = "https://www.mercatus.org/emergentventures"
    page_source = fetch_page_with_selenium(url)
    print(page_source)

if __name__ == "__main__":
    main()