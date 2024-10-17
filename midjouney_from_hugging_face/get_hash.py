import os
import hashlib
import pandas as pd
from tqdm import tqdm

# Load the DataFrame
df = pd.read_parquet('/weka/datasets/midjourney/from_huggingface/meta/midjourney_errfree.parquet')

# Directory containing the images
image_directory = '/weka/datasets/midjourney/from_huggingface/image'

# Function to calculate hash for an image file
def calculate_image_hash(file_path):
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None

# Add a new column "HASH" to the DataFrame
hashes = []

# Iterate over each row in the DataFrame with tqdm for progress
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating hashes"):
    file_path = os.path.join(image_directory, f"{row['id']}.jpg")  # Adjust the file name format if needed

    # Check if the image file exists and calculate its hash
    if os.path.isfile(file_path):
        image_hash = calculate_image_hash(file_path)
        hashes.append(image_hash)
    else:
        hashes.append(None)  # Append None if the file doesn't exist

# Add the "HASH" column to the DataFrame
df['HASH'] = hashes

# Save the updated DataFrame to a new parquet file
df.to_parquet('/weka/datasets/midjourney/from_huggingface/meta/midjourney_errfree.parquet')

print("Updated DataFrame with HASH column has been saved.")