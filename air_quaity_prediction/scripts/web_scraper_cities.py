import time
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException

def run_scraper():
    # Define the download directory
    download_dir = os.path.join(os.getcwd(), 'outdoor_pm25_dataset')
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run headless Chrome
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    prefs = {"download.default_directory": download_dir}
    chrome_options.add_experimental_option("prefs", prefs)

    # Set up the WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # URL to be scraped
        url = 'https://citiesair.com/dashboard/nyuad'

        # Open the URL
        driver.get(url)

        # Wait for the page to load
        time.sleep(5)

        # Wait for the "Raw Dataset" button to be clickable and then click it
        try:
            initial_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div[3]/div/div[2]/button"))
            )
            initial_button.click()

            raw_dataset_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "/html//div[@id='root']/div[@class='MuiBox-root css-ghcfah']/div[@class='MuiBox-root css-0']//div[@class='MuiStack-root css-jj2ztu']/button[2]"))
            )
            raw_dataset_button.click()

            # Wait for the download button to be clickable and then click it to download the file
            download_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div[5]/div[@role='presentation']/div[@role='dialog']//button[@type='button']"))
            )
            download_button.click()

            # Wait for the download to complete
            time.sleep(10)  # Adjust time as needed for download to complete

            # Rename the downloaded file with the timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            for filename in os.listdir(download_dir):
                if filename.startswith("nyuad-outdoors-hourly"):
                    new_filename = f"nyuad-outdoors-hourly-{timestamp}.csv"
                    os.rename(os.path.join(download_dir, filename), os.path.join(download_dir, new_filename))
                    print(f"File renamed to {new_filename}")
                    return new_filename  # Return the new filename
                    break

        except (NoSuchElementException, TimeoutException) as e:
            print("Error: Element not found or timed out -", e)
        except WebDriverException as e:
            print("Error: WebDriver exception -", e)

    finally:
        # Close the WebDriver
        driver.quit()

    return None
