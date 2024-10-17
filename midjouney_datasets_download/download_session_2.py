import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from PIL import Image

# Create a directory to save the original images if it doesn't exist
output_dir = '/weka/datasets/midjourney/from_kaggle/download/part10_raw'
#cropped_output_dir = '/weka/datasets/midjourney/from_kaggle/images'
os.makedirs(output_dir, exist_ok=True)
#os.makedirs(cropped_output_dir, exist_ok=True)

df = pd.read_csv('/weka/datasets/midjourney/from_kaggle/meta/prompts_part10.csv')


def log_error(message):
    with open('/home/xingyu.chen/Project/midjouney_datasets_download/error_image.txt', 'a') as error_file:
        error_file.write(message + '\n')

# Function to download an image
def download_image(row):
    image_url = row['image_url']
    image_name = row['img_name']
    
    # Only download images with '__crop0'
    if pd.isna(image_name) or '__crop0' not in image_name:
        return
    
    file_path = os.path.join(output_dir, image_name)
    
    if os.path.exists(file_path):
        return f"Skipping {image_name}, already downloaded."
    
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return
        else:
            error_message = f"Failed to download {image_name}, status code: {response.status_code}"
            log_error(error_message)
            return error_message
    except Exception as e:
        error_message = f"Error downloading {image_name}: {e}"
        log_error(error_message)
        return error_message

# Multithreading for downloading images
def download_images_multithreaded(df, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image, row) for _, row in df.iterrows()]
        
        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                print(result)

download_images_multithreaded(df, max_workers=8)
