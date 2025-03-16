import pyautogui
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
import requests
import os

# Set up Chrome WebDriver with options to keep it open
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument("--start-maximized")  # Start maximized
service = ChromeService()
driver = webdriver.Chrome(service=service, options=chrome_options)

# Function to download an image

save_directory = "Ai_images"
categories=[5133,5193,111763,5241,5232,111768,111757,414,55,111805,3915,617,5132,6594,8363,2539,111833,172,5188]
# Scroll down the page to load more images and download them
scroll_pause_time = 4  # Adjust pause time as needed
scroll_amount = -99999  # Negative value to scroll down
def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        file_name = os.path.basename(image_url)

        with open(f"{save_path}/{file_name}", 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded and saved at {file_name}.")
    except Exception as e:
        print(f"Failed to download image: {e}")

for tag in categories:
    # Open the webpage
    url = f"https://civitai.com/images?tags={tag}"
    driver.get(url)
    time.sleep(5)  # Wait for the page to load


    # Center the mouse cursor on the screen
    screen_width, screen_height = pyautogui.size()
    center_x, center_y = screen_width / 2, screen_height / 2
    pyautogui.moveTo(center_x, center_y)
    time.sleep(1)  # Pause to ensure the mouse is centered
    last_image_name="last_image.png"
    # Use pyautogui to scroll the mouse wheel
    current_images = []
    for i in range(250):
        pyautogui.scroll(scroll_amount)
        time.sleep(scroll_pause_time)
        # Download images
        divs = driver.find_elements(By.CSS_SELECTOR, "div.mantine-16xlp3a")
        next_images = []

        for div in divs:
            try:
                imgs = div.find_elements(By.CSS_SELECTOR, "img.__mantine-ref-image.mantine-deph6u")
                for img in imgs:
                    image_url = img.get_attribute("src")
                    next_images.append(image_url)
                    download_image(image_url, save_directory)

            except Exception as e:
                print(f"Error retrieving or downloading image: {e}")
        if current_images == next_images:
            break
        else:
            current_images = next_images
driver.quit()
