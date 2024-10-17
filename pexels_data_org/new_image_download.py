##1. for new images, check if the id exist
##2. if id exists, continue to next
##3. if not, check if the hash exists, if yes, add to duplicates, continue
##4. new images, store meta, json, hash
# Write the new queries to settings.txt

import requests
import os
import re
import pandas as pd # type: ignore
import time  # To add delay between batches
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
# Your Unsplash API key here
API_KEY = 'uF7AnSt9jyL3HMf20f7BxDRZwQuiPvXq8w0amzFu5DeEI4ZcidwokaIT'  # Replace with your actual API key

# Directories to save images and metadata
IMAGE_DIR_SEC1 = '/weka/datasets/pexels_section2/section1/download'  # Directory to save images
IMAGE_DIR_SEC2 = '/weka/datasets/pexels_section2/section2/download'  # Directory to save images
META_DIR = '/weka/datasets/pexels_section2/section2/meta'  # Directory to save metadata
JSON_DIR='/weka/datasets/pexels_section2/section2/json'  

# Ensure the directories exist
if not os.path.exists(IMAGE_DIR_SEC1):
    os.makedirs(IMAGE_DIR_SEC1)
# Ensure the directories exist
if not os.path.exists(IMAGE_DIR_SEC2):
    os.makedirs(IMAGE_DIR_SEC2)
    
if not os.path.exists(META_DIR):
    os.makedirs(META_DIR)
    
if not os.path.exists(JSON_DIR):
    os.makedirs(JSON_DIR)

# Metadata and failed downloads file paths
metadata_file = os.path.join(META_DIR, 'pexels_section2.parquet')
failed_file = os.path.join(META_DIR, 'pexels_failed_downloads.parquet')

# Function to fetch images from Unsplash
def fetch_images(query, per_page=30, page=1):
    url = f'https://api.pexels.com/v1/search?query={query}&per_page={per_page}&page={page}'
    headers = {
        'Authorization': API_KEY,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  # Simulate a browser request
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# Function to sanitize and create a valid filename
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Function to check if an image file already exists in the directory
def image_exists(img_id):
    filename = sanitize_filename(f'{img_id}.jpeg')
    return os.path.exists(os.path.join(IMAGE_DIR_SEC1, filename)) or os.path.exists(os.path.join(IMAGE_DIR_SEC2, filename))

# Function to download an image
def download_image(photo, hash_table):
    img_id = photo['id']
    img_url = photo['src']['original']
    sanitized_filename = sanitize_filename(f'{img_id}.jpeg')
    save_path = os.path.join(IMAGE_DIR_SEC2, sanitized_filename)
    try:
        img_data = requests.get(img_url).content
        hash=get_hash(img_data)
        if hash not in hash_table:
            with open(save_path, 'wb') as handler:
                handler.write(img_data)
            return True, img_id, photo, hash
        
        else:
            return False, img_id, photo, 0
    except Exception as e:
        print(f"Failed to download image from {img_url}: {e}")
        return False, img_id, photo, 0

# Function to save metadata to a DataFrame and parquet
def save_metadata(data, metadata_file):
    df = pd.DataFrame(data)
    
    # If there is existing metadata, append the new data
    if os.path.exists(metadata_file):
        existing_df = pd.read_parquet(metadata_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_parquet(metadata_file, index=False)
    print(f"Saved metadata to '{metadata_file}'.")

# Function to save failed downloads
def save_failed_downloads(failed_metadata, failed_file):
    failed_df = pd.DataFrame(failed_metadata)

    if os.path.exists(failed_file):
        existing_df = pd.read_parquet(failed_file)
        combined_df = pd.concat([existing_df, failed_df], ignore_index=True)
    else:
        combined_df = failed_df

    combined_df.to_parquet(failed_file, index=False)
    print(f"Saved failed download information to '{failed_file}'.")

def get_hash(image_data):
    return hashlib.sha256(image_data).hexdigest()

# Main logic to fetch images, save metadata, and download images
def main():
    with open('pexels_settings.txt', 'r') as file:
        queries = [line.strip() for line in file]
    
    per_page = 80  # Unsplash limit on per_page (max is 30)
    total_pages = 100  # We want to fetch up to 300 pages
    batch_size = 20  # Process 1 pages at a time
    num_threads = 4  # Number of threads for downloading images
    
    for query in queries:
        print(f"Processing query: {query}")

        for batch_start in range(1, total_pages + 1, batch_size):
            futures = []
            metadata = []
            failed_downloads = []
            hash_code_1=set(pd.read_parquet('/weka/datasets/pexels_section2/section1/meta/pexels_section1.parquet')['HASH'])
            hash_code_2=set(pd.read_parquet('/weka/datasets/pexels_section2/section2/meta/pexels_section2.parquet')['HASH'])
            
            hash_table = hash_code_1 | hash_code_2
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                for page in range(batch_start, min(batch_start + batch_size, total_pages + 1)):
                    print(f"Fetching page {page} for query '{query}'")
                    images = fetch_images(query=query, per_page=per_page, page=page)
                    if 'photos' not in images or not images['photos']:
                        print(f"No more images found for query '{query}' at page {page}. Stopping pagination.")
                        break

                    for photo in images['photos']:
                        img_id = photo['id']
                        if image_exists(img_id):  # Check if image is already downloaded
                            print(f"Image {img_id} already downloaded, skipping.")
                            continue
                        future = executor.submit(download_image, photo, hash_table)
                        futures.append(future)

                for future in as_completed(futures):
                    success, img_id, photo, hash = future.result()
                    if success:
                        metadata.append({
                            'ID': photo['id'],
                            'OLD_PATH': os.path.join(IMAGE_DIR_SEC2, f"{photo['id']}.jpeg"),
                            'URL': photo['src']['original'],
                            'HASH': hash,
                            'HW': [photo['height'], photo['width']],
                            'TITLE': photo['alt'],
                            'DUPLICATE_OF': None,
                        })
                    else:
                        failed_downloads.append(img_id)
            # Save metadata and failed downloads after every batch
            save_metadata(metadata, metadata_file)
            if failed_downloads:
                save_failed_downloads(failed_downloads, failed_file)

            # Delay between batches to avoid overwhelming the API
            time.sleep(360)

if __name__ == '__main__':
    main()