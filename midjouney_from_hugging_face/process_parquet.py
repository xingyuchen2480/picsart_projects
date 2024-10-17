import os
import shutil
import pandas as pd

# Load your DataFrame (assuming you already have df)
# df = pd.read_parquet('/weka/datasets/midjourney/from_huggingface/train.parquet')

# Define the source directory where the images are currently located
source_dir = '/weka/datasets/midjourney/from_huggingface/unpack'  # Replace with your source folder path

# Define the destination directory where you want to move the images not in the DataFrame
move_to_dir = '/weka/datasets/midjourney/from_huggingface/not_in_meta_images'  # Replace with your destination folder path
os.makedirs(move_to_dir, exist_ok=True)  # Create the folder if it doesn't exist


df=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/train.parquet')



# Get the list of image IDs from the DataFrame (assuming 'id' contains image filenames without .png)
image_ids_in_df = set(df['id'].tolist())

# Loop through all files in the source directory
for filename in os.listdir(source_dir):
    # Check if the file is a .png file
    if filename.endswith('.png'):
        # Strip the .png extension to match the 'id' column in the DataFrame
        file_id = os.path.splitext(filename)[0]
        
        # If the file is NOT in the DataFrame, move it to the destination folder
        if file_id not in image_ids_in_df:
            source_file = os.path.join(source_dir, filename)
            destination_file = os.path.join(move_to_dir, filename)
            
            # Move the file
            shutil.move(source_file, destination_file)
            print(f"Moved: {source_file} to {destination_file}")
