import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load the DataFrame
df = pd.read_parquet('/weka/datasets/pexels_section2/section2/meta/pexels_section2_noterrfree.parquet')

# Directory where the images will be downloaded
download_directory = '/weka/datasets/pexels_section2/section2/download/'

# Load hash table from section1 and ensure it's a set of unique hashes
hash_table = set(pd.read_parquet('/weka/datasets/pexels_section2/section1/meta/pexels_section1.parquet')['HASH'])

# Ensure the download directory exists
os.makedirs(download_directory, exist_ok=True)

# Function to download an image from a URL
def download_image(row):
    image_url = row['URL']
    image_hash = row['HASH']
    save_path = os.path.join(download_directory, f"{row['ID']}.jpeg")  # Adjust extension if necessary
    
    # Check if the file is already downloaded (in hash_table or in the file system)
    if image_hash in hash_table or os.path.exists(save_path):
        return None  # Skip if already in the hash_table or downloaded
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(image_url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return None  # Skip printing for successfully downloaded files
        else:
            return f"Failed to download {image_url}: {response.status_code}"
    except Exception as e:
        return f"Error downloading {image_url}: {e}"

# Number of threads to use (adjust based on your system)
num_threads = 4

# Use ThreadPoolExecutor to download images concurrently
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit tasks to the executor and add tqdm for the progress bar
    futures = {executor.submit(download_image, row): row for _, row in df.iterrows()}

    # Use tqdm to display progress as tasks are completed, only printing unsuccessful downloads
    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
        result = future.result()
        if result:  # Only print if there's an issue (error or failed download)
            print(result)

print("Download process completed.")