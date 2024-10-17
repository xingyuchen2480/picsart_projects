# Import required modules
from PIL import Image
import os
import json
from tqdm import tqdm

# Directories for images and JSON files
image_directory = '/weka/datasets/pexels_section2/section1/image'
json_directory = '/weka/datasets/pexels_section2/section1/json'
new_json_directory = '/weka/datasets/pexels_section2/section1/new_json'

# Create the new JSON directory if it doesn't exist
os.makedirs(new_json_directory, exist_ok=True)

# Get the list of image files to process
image_files = os.listdir(image_directory)

# Iterate through each file in the image directory with a tqdm progress bar
for image_filename in tqdm(image_files, desc="Processing images", unit="file"):
    # Get the full path of the image
    image_path = os.path.join(image_directory, image_filename)
    
    try:
        # Open the image and get its dimensions (width and height)
        with Image.open(image_path) as img:
            width, height = img.size

        # Find the corresponding JSON file (assuming they have the same filename)
        json_filename = os.path.splitext(image_filename)[0] + '.json'
        json_path = os.path.join(json_directory, json_filename)
        
        if os.path.exists(json_path):
            # Load the JSON file
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)

            # Add or update the width and height in the JSON data
            data['hw']= [height, width]
            # Save the updated JSON file in the new directory
            new_json_path = os.path.join(new_json_directory, json_filename)
            with open(new_json_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        else:
            print(f"JSON file not found for {image_filename}")
    
    except Exception as e:
        print(f"Failed to process {image_filename}: {e}")
print("All images processed.")