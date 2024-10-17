import os
import json
import pandas as pd
from tqdm import tqdm
import shutil
from PIL import Image

import re
image_directory = '/weka/datasets/pexels_section2/section1/image'
#pinfo_path = '/weka/datasets/pexels_reorg/meta/clean_pexels.parquet'
#pinfo = pd.read_parquet(pinfo_path)
# Directory where your JSON files are stored
json_directory  = '/weka/datasets/pexels_section2/section1/json'

# Set of all image filenames in the image directory
image_files = set(os.listdir(image_directory))

# List to store missing image paths
missing_images = []
stored_images=[]
error_images=[]
count=0

def is_image_valid(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that the image is not corrupted
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image: {image_path}, error: {e}")
        return False


# Iterate through each JSON file with tqdm for a progress bar
for json_filename in tqdm(os.listdir(json_directory), desc="Processing JSON files"):
    json_path = os.path.join(json_directory, json_filename)
    
    # Open and load the JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        
        # Check if the image path exists
        local_image_path = os.path.basename(data['shifted path'])
        
        if local_image_path not in image_files:
            missing_images.append(local_image_path)
            # Copy the image from its original local path to the image directory
            try:
                original_image_path = data['local path']  # Full path to the original image
                shutil.copy(original_image_path, data['shifted path'])
            except Exception as e:
                print(f"Error copying {original_image_path}: {e}")
        else:
            if not is_image_valid(data['shifted path']):
                error_images.append(missing_images)
    # count+=1
    # if count==10:
    #     break
# Output the number of missing images and debug info




print(f'Stored images: {len(stored_images)}')
print(f'Missing images: {len(missing_images)}')
debug=1