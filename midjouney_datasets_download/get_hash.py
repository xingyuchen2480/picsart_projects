import os
import hashlib
import pandas as pd
from tqdm import tqdm

# Path to the directory with images
image_dir = '/weka/datasets/midjourney/from_kaggle/image/part10/'
hash_store_file = '/weka/datasets/midjourney/from_kaggle/meta/hash_part10.parquet'

# Function to compute SHA-256 hash of an image
def compute_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# Function to store image hashes in a DataFrame and write to CSV
def store_image_hashes():
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    # Create an empty list to store the image data
    data = []

    for image_name in tqdm(image_files, desc="Hashing images", unit="image"):
        image_path = os.path.join(image_dir, image_name)
        imhash = compute_image_hash(image_path)
        data.append([image_name, imhash])
    
    # Construct a DataFrame
    df = pd.DataFrame(data, columns=['img_name', 'HASH'])
    
    # Write the DataFrame to CSV
    df.to_parquet(hash_store_file, index=False)
    print(f"Hashes stored in {hash_store_file}")

# Run the hash storing function
if __name__ == "__main__":
    store_image_hashes()
    debug=1
