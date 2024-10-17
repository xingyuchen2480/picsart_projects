import requests
import os
import re
import pandas as pd
import time  # To add delay between batches
from concurrent.futures import ThreadPoolExecutor, as_completed

# Your Unsplash API key here
API_KEY = 'rRm51FnON3nJCh3jgezm1Ii8zqG3KKuTLMOxqyd4nPM'  # Replace with your actual Unsplash API key

# Directories to save images and metadata
IMAGE_DIR = '/weka/datasets/unsplash/image'  # Directory to save images
META_DIR = '/weka/datasets/unsplash/meta'  # Directory to save metadata

# Ensure the directories exist
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

if not os.path.exists(META_DIR):
    os.makedirs(META_DIR)

# Metadata and failed downloads file paths
metadata_file = os.path.join(META_DIR, 'unsplash_images_metadata.csv')
failed_file = os.path.join(META_DIR, 'unsplash_failed_downloads.csv')

# Function to fetch images from Unsplash
def fetch_images(query, per_page=30, page=1):
    url = f'https://api.unsplash.com/search/photos?query={query}&per_page={per_page}&page={page}&client_id={API_KEY}'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Function to sanitize and create a valid filename
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Function to check if an image file already exists in the directory
def image_exists(img_id):
    filename = sanitize_filename(f'{img_id}.jpg')
    return os.path.exists(os.path.join(IMAGE_DIR, filename))

# Function to download an image
def download_image(photo):
    img_id = photo['id']
    img_url = photo['urls']['full']
    sanitized_filename = sanitize_filename(f'{img_id}.jpg')
    save_path = os.path.join(IMAGE_DIR, sanitized_filename)
    try:
        img_data = requests.get(img_url).content
        with open(save_path, 'wb') as handler:
            handler.write(img_data)
        return True, img_id, photo
    except Exception as e:
        print(f"Failed to download image from {img_url}: {e}")
        return False, img_id, photo

# Function to save metadata to a DataFrame and CSV
def save_metadata(data, metadata_file):
    df = pd.DataFrame(data)
    
    # If there is existing metadata, append the new data
    if os.path.exists(metadata_file):
        existing_df = pd.read_csv(metadata_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(metadata_file, index=False)
    print(f"Saved metadata to '{metadata_file}'.")

# Function to save failed downloads
def save_failed_downloads(failed_metadata, failed_file):
    failed_df = pd.DataFrame(failed_metadata)

    if os.path.exists(failed_file):
        existing_df = pd.read_csv(failed_file)
        combined_df = pd.concat([existing_df, failed_df], ignore_index=True)
    else:
        combined_df = failed_df

    combined_df.to_csv(failed_file, index=False)
    print(f"Saved failed download information to '{failed_file}'.")

# Main logic to fetch images, save metadata, and download images
def main():
    with open('unsplash_settings.txt', 'r') as file:
        queries = [line.strip() for line in file]
    
    per_page = 30  # Unsplash limit on per_page (max is 30)
    total_pages = 330  # We want to fetch up to 300 pages
    batch_size = 5  # Process 50 pages at a time
    num_threads = 4  # Number of threads for downloading images

    for query in queries:
        print(f"Processing query: {query}")

        for batch_start in range(1, total_pages + 1, batch_size):
            futures = []
            metadata = []
            failed_downloads = []
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                for page in range(batch_start, min(batch_start + batch_size, total_pages + 1)):
                    print(f"Fetching page {page} for query '{query}'")
                    images = fetch_images(query=query, per_page=per_page, page=page)
                    if 'results' not in images or not images['results']:
                        print(f"No more images found for query '{query}' at page {page}. Stopping pagination.")
                        break

                    for photo in images['results']:
                        img_id = photo['id']
                        if image_exists(img_id):  # Check if image is already downloaded
                            print(f"Image {img_id} already downloaded, skipping.")
                            continue

                        future = executor.submit(download_image, photo)
                        futures.append(future)

                for future in as_completed(futures):
                    success, img_id, photo = future.result()
                    if success:
                        metadata.append({
                            'ID': photo['id'],
                            'Slug': photo['slug'],
                            'Image URL': photo['urls']['full'],
                            'Image Width': photo['width'],
                            'Image Height': photo['height'],
                            'Color': photo['color'],
                            'Blur Hash': photo['blur_hash'],
                            'Description': photo.get('description', ''),
                            'Alt Description': photo.get('alt_description', ''),
                            'Topic Submissions': photo.get('topic_submissions', {}),
                            'Type': photo.get('asset_type', 'unknown'),
                            'Tags': [tag["title"] for tag in photo.get("tags", []) if tag.get("type") == "search"]
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