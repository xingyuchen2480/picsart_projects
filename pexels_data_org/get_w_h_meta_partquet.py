import os
import json
import pandas as pd
import re
from PIL import Image
from tqdm import tqdm

# Paths to the directories containing the images, JSON files, and CSV
image_directory = '/weka/datasets/pexels_section2/section1/image'
json_directory = '/weka/datasets/pexels_section2/section1/json'
csv_file_path = '/weka/datasets/pexels_section2/section1/meta/cleaned_images.csv'
parquet_file_path = '/weka/datasets/pexels_section2/section1/meta/clean_images.parquet'


#extract_image_id
def extract_image_id(url):
    # Use regex to extract the ID from the URL
    match = re.search(r'photos/(\d+)/', url)
    if match:
        return match.group(1)
    return None



# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)
# Create new columns 'width' and 'height' if they don't exist
if 'WIDTH' not in df.columns:
    df['WIDTH'] = None  # Initialize the width column with None
if 'HEIGHT' not in df.columns:
    df['HEIGHT'] = None  # Initialize the height column with None
if 'ID' not in df.columns:
    df['ID'] = None  # Initialize the height column with None
    
# Assuming the JSON is stored in a column named 'info' in the CSV
# Modify this to the actual name of the column that stores JSON data
json_column_name = 'info'
# Iterate through each row in the CSV with a tqdm progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing CSV rows"):
    url = row['URL']
    image_id = extract_image_id(url)
    image_filename = f'{image_id}.jpeg'
    image_path = os.path.join(image_directory, image_filename)
    
    try:
        # Open the image and get its dimensions (width and height)
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Parse the JSON data from the CSV row
        # Store width and height in the respective columns
        df.at[index, 'WIDTH'] = width
        df.at[index, 'HEIGHT'] = height
        df.at[index, 'ID'] = image_filename
    except Exception as e:
        print(f"Failed to process {image_filename}: {e}")
# Save the updated DataFrame back to the CSV file
df.to_parquet(parquet_file_path, index=False)

print("CSV file updated with height and width for each image.")