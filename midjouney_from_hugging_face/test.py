import pandas as pd
import os
import json
from PIL import Image
from tqdm import tqdm

########################
# check missing images #
########################

# df=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/train.parquet')
# image_path='/weka/datasets/midjourney/from_huggingface/unpack'
# # Check if images exist and create the valid and missing DataFrames
# valid_images_df = df[df['id'].apply(lambda img_id: os.path.exists(os.path.join(image_path, f"{img_id}.png")))].reset_index(drop=True)
# missing_images_df = df[~df['id'].apply(lambda img_id: os.path.exists(os.path.join(image_path, f"{img_id}.png")))].reset_index(drop=True)
# # Save the valid and missing DataFrames to parquet files
# valid_images_df.to_parquet('/weka/datasets/midjourney/from_huggingface/valid_images.parquet', index=False)
# missing_images_df.to_parquet('/weka/datasets/midjourney/from_huggingface/missing_images.parquet', index=False)

# print(f"Valid images saved: {len(valid_images_df)}")
# print(f"Missing images saved: {len(missing_images_df)}")
# debug=1


#second_image_path='/weka/datasets/midjourney/from_huggingface/not_in_meta_data'


# missing_images_df['image_exists'] = missing_images_df['id'].apply(lambda img_id: os.path.exists(os.path.join(second_image_path, f"{img_id}.png")))
# reget_df=missing_images_df[missing_images_df['image_exists'] == True]


# debug=1




#########################
# delete invalid images #
#########################
# df=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/train.parquet')

# invalid_images_file = '/weka/datasets/midjourney/from_huggingface/invalid_files.txt'
# image_dir='/weka/datasets/midjourney/from_huggingface/unpack/'
# # Read the invalid image list
# with open(invalid_images_file, 'r') as file:
#     invalid_images = [line.strip() for line in file.readlines()]
    
# # Loop through the invalid image list and delete each file
# for image_name in invalid_images:
#     image_path = os.path.join(image_dir, image_name)
#     if os.path.exists(image_path):
#         os.remove(image_path)
#         print(f'Deleted: {image_path}')
#     else:
#         print(f'File not found: {image_path}')


####################
# write error file #
####################
# df=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/missing_images.parquet')

# # Directory where the individual .err files will be stored
# output_dir = '/weka/datasets/midjourney/from_huggingface/unpack'

# # Ensure the directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Create a unique .err file for each id
# for index, row in df.iterrows():
#     # Define the filename as {id}.err
#     file_name = f"{row['id']}.err"
#     file_path = os.path.join(output_dir, file_name)
    
#     # Write the URL to the unique .err file
#     with open(file_path, 'w') as f:
#         f.write(f"{row['url']}\n")

# print(f"{len(df)} .err files have been created, one for each ID.")
# debug=1


###############
#write as jpg #
###############

# #Define directories
# unpack_dir = '/weka/datasets/midjourney/from_huggingface/not_in_meta_images/original_png'
# output_dir = '/weka/datasets/midjourney/from_huggingface/not_in_meta_images/converted_jpg'
# total_files = sum([len(files) for r, d, files in os.walk(unpack_dir) if any(f.lower().endswith('.png') for f in files)])
# # Ensure the output directory exists
# os.makedirs(output_dir, exist_ok=True)
# with tqdm(total=total_files, desc="Converting PNG to JPG", unit="file") as pbar:
#     for root, dirs, files in os.walk(unpack_dir):
#         for file in files:
#             if file.lower().endswith('.png'):
#                 # Define input and output paths
#                 png_file_path = os.path.join(root, file)
#                 jpg_file_name = file.replace('.png', '.jpg')
#                 jpg_file_path = os.path.join(output_dir, jpg_file_name)
                
#                 # Open the .png image and save as .jpg with quality 95
#                 try:
#                     with Image.open(png_file_path) as img:
#                         img.save(jpg_file_path, 'JPEG', quality=95)
#                     # Update progress bar after successful conversion
#                     pbar.update(1)
#                 except Exception as e:
#                     print(f"Failed to convert {file}: {e}")
#                     pbar.update(1)  # Still update the progress even if conversion fails

# print("All PNG files have been processed.")


# debug=1

###########
#check hash
###########

# import pandas as pd
# import os
# import hashlib
# from PIL import Image
# import json
# from tqdm import tqdm
# import numpy as np
# base_dir='/weka/datasets/pexels_section2/section1/download/'
# def calculate_image_hash(image_path):
#     hash_sha256 = hashlib.sha256()
#     try:
#         with open(image_path, 'rb') as f:
#             while chunk := f.read(8192):
#                 hash_sha256.update(chunk)
#         return hash_sha256.hexdigest()
#     except Exception as e:
#         print(f"Error calculating hash for {image_path}: {e}")
#         return None

# def check_image_hashes(df, num_images=10):
#     for index, row in df.iloc[:num_images].iterrows():
#         # Construct the full image path using the base directory and image name
#         image_path = os.path.join(base_dir, f"{row['ID']}.jpeg")
        
#         # Check if the image file exists
#         if not os.path.exists(image_path):
#             print(f"File not found: {image_path}")
#             continue
        
#         # Calculate the hash of the image
#         calculated_hash = calculate_image_hash(image_path)
#         stored_hash = row['HASH']
        
#         # Print the calculated and stored hashes
#         print(f"\nImage Path: {image_path}")
#         print(f"Calculated Hash: {calculated_hash}")
#         print(f"Stored Hash: {stored_hash}")
        
#         # Compare the hashes
#         if calculated_hash:
#             if calculated_hash == stored_hash:
#                 print("Status: Hash matches")
#             else:
#                 print("Status: Hash does not match!")
#         else:
#             print(f"Could not calculate hash for {image_path}.")

# # Load the DataFrame
# df = pd.read_parquet('/weka/datasets/pexels_section2/section1/meta/pexels_section1.parquet')

# # Call the function to check image hashes
# check_image_hashes(df, num_images=100)
# debug=1


##########################
# check hash for 2 parts #
##########################
# import pandas as pd
# import os
# import json
# from tqdm import tqdm

# # Base directory where the JSON parts are stored
# base_directory = '/weka/datasets/midjourney/from_kaggle/json/'

# # List to hold all hashes
# all_hashes = []

# # Iterate over part directories from part001 to part010
# for part in range(1, 11):
#     part_directory = os.path.join(base_directory, f'part{part:03d}')
    
#     # Iterate over all JSON files in the current part directory
#     for filename in tqdm(os.listdir(part_directory), desc=f"Processing part{part:03d}"):
#         if filename.endswith('.json'):
#             file_path = os.path.join(part_directory, filename)
            
#             # Open and read the JSON file
#             with open(file_path, 'r') as file:
#                 try:
#                     data = json.load(file)
#                     # Extract the hash value from the JSON (adjust 'HASH' based on your JSON structure)
#                     if 'HASH' in data:
#                         all_hashes.append(data['HASH'])
#                     else:
#                         print(f"HASH key not found in {filename}")
#                 except json.JSONDecodeError as e:
#                     print(f"Error reading {file_path}: {e}")

# # Print total hashes found
# print(f"Total hashes found: {len(all_hashes)}")

# # Optionally save the hashes to a file
# hashes_file = os.path.join(base_directory, 'all_hashes.txt')
# with open(hashes_file, 'w') as f:
#     for hash_value in all_hashes:
#         f.write(f"{hash_value}\n")

# print(f"All hashes have been saved to {hashes_file}")

# hashlist2=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/meta/midjourney_errfree.parquet')['HASH'].tolist()

# debug=1


########################
# compare 2 parts hash #
########################
# import pandas as pd
# def read_txt_to_list(file_path):
#     with open(file_path, 'r') as file:
#         return [line.strip() for line in file]
    
# hashlist1 = read_txt_to_list('/weka/datasets/midjourney/from_kaggle/json/all_hashes.txt')
# hashlist2 = pd.read_parquet('/weka/datasets/midjourney/from_huggingface/meta/midjourney_errfree.parquet')['HASH'].tolist()
# debug=1


####################################
# get meta for not-recorded images #
####################################
# import os
# import hashlib
# from PIL import Image
# import pandas as pd
# image_dir='/weka/datasets/midjourney/from_huggingface//not_in_meta_images/converted_jpg'

# #df=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/meta/midjourney_errfree.parquet')

# # Function to compute SHA-256 hash of an image
# def compute_image_hash(image_path):
#     with open(image_path, "rb") as f:
#         return hashlib.sha256(f.read()).hexdigest()

# # Function to get the width and height (HW) of an image
# def get_image_hw(image_path):
#     with Image.open(image_path) as img:
#         return img.width, img.height
# # List to store the new metadata for images that don't have corresponding metadata
# new_data = []


# # Iterate through image files in the directory
# for image_name in os.listdir(image_dir):
#     if image_name.endswith(('jpg', 'jpeg', 'png')):  # Add more extensions as needed
#         image_path = os.path.join(image_dir, image_name)
        
#         # Extract the ID from the image name (without extension)
#         image_id = os.path.splitext(image_name)[0]
        
#         # Get the image hash and HW (height, width)
#         image_hash = compute_image_hash(image_path)
#         image_hw = get_image_hw(image_path)
        
#         # Create a dictionary for the new row with default/blank values for other fields
#         new_row = {
#             'id': image_id,
#             'version': '6.0',
#             'arguments': 'v 6.0',
#             'original_text': '',       # Blank
#             'caption': '',             # Blank
#             'gpt_caption': '',         # Blank
#             'url': '',                 # Blank
#             'width': image_hw[0],      # Width
#             'height': image_hw[1],     # Height
#             'reactions': '',           # Blank
#             'HASH': image_hash,        # Computed hash
#             'Duplicate_of': None       # None for no duplicates
#         }
        
#         # Add the new row to the list
#         new_data.append(new_row)

# # Convert the list of new data to a DataFrame
# new_df = pd.DataFrame(new_data)
# new_df.to_parquet('/weka/datasets/midjourney/from_huggingface/not_in_meta_images/composed_meta/composed_meta.parquet')
# # Print the DataFrame or save it to a file
# print(new_df)

# debug=1


####################################
# check composed part hash overlap #
####################################
# import pandas as pd
# df1=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/meta/midjourney_errfree.parquet')
# df2=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/not_in_meta_images/composed_meta/composed_meta.parquet')
# hash_to_id_map = df1.set_index('HASH')['id'].to_dict()
# def find_duplicates(row):
#     hash_value = row['HASH']
#     image_id = row['id']
    
#     # Check if the hash exists in the map (from df1 or previously processed rows in df2)
#     if hash_value in hash_to_id_map:
#         return hash_to_id_map[hash_value]  # Return the ID from df1 or df2 where the hash matches
    
#     # If the hash is not in the map, add it (for future rows in df2)
#     hash_to_id_map[hash_value] = image_id
    
#     return None  # No duplicate found
# # Apply the function to df2 to create the 'Duplicate_of' column
# df2['Duplicate_of'] = df2.apply(find_duplicates, axis=1)

# df2.to_parquet('/weka/datasets/midjourney/from_huggingface/not_in_meta_images/composed_meta/composed_meta.parquet')

# print("Duplicate_of column updated and file saved.")
# debug=1


###################
# cleanup caption #
###################
# import pandas as pd
# import re

# def clean_text(text):
#     # Remove content between <> if they exist
#     if '<' in text and '>' in text:
#         text = re.sub(r'<.*?>', '', text)
    
#     # Extract all instances of content between ** and return them combined
#     matches = re.findall(r'\*\*(.*?)\*\*', text)
    
    
    
    
#     # Join the extracted content (from ** **) with spaces and return the result
#     return ' '.join(matches).strip() if matches else text


# def clean_text_2(text):
#     # Remove version information like '--v 6.0'
#     text = re.sub(r'--v \d+(\.\d+)?', '', text)
#     # Clean up extra spaces if needed
#     text = re.sub(r'\s+', ' ', text).strip()
#     if text.startswith(', '):
#         text = re.sub(r'^,\s+', '', text)
#     return text



# df=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/not_in_meta_images/composed_meta/composed_meta.parquet')
# df['hw'] = df.apply(lambda row: [float(row['height']), float(row['width'])], axis=1)
# # df = df.drop(['reactions', 'arguments', 'url','width', 'height', 'version', 'caption', 'gpt_caption'], axis=1)
# # df['original_text']=df['original_text'].apply(clean_text).apply(clean_text_2)
# # df.rename(columns={'original_text': 'caption'}, inplace=True)
# df = df.drop(['reactions', 'arguments', 'url','width', 'height', 'version', 'original_text', 'gpt_caption'], axis=1)
# debug=1
# df.to_parquet('/weka/datasets/midjourney/from_huggingface/not_in_meta_images/composed_meta/composed_meta.parquet')

# debug=1

import pandas as pd

df1=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/meta/midjourney.parquet')
df2=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/not_in_meta_images/composed_meta/composed_meta.parquet')
df1['caption_error'] = None

# Add the 'caption_error' column to df2 as True
df2['caption_error'] = True

merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged DataFrame to a parquet file
output_path = '/weka/datasets/midjourney/from_huggingface/meta/midjourney_clean.parquet'
merged_df.to_parquet(output_path)

debug=1