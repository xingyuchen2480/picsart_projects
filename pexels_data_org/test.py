import pandas as pd
import os
import hashlib
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
###########
#check hash
###########
# base_dir='/weka/datasets/pexels_section2/section1/image_prev/'


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

# # Check the first 10 images and compare their hash with the HASH column
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
# check_image_hashes(df, num_images=10)


####################
# recalculate json #          
####################
# output_dir = '/weka/datasets/pexels_section2/section1/json_new'
# os.makedirs(output_dir, exist_ok=True) 
# df=pd.read_parquet('/weka/datasets/pexels_section2/section1/meta/pexels_section1.parquet')
# # Iterate over each row in the DataFrame with tqdm progress bar

# # Function to convert non-serializable objects to serializable types
# def convert_to_serializable(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()  # Convert ndarray to list
#     return obj  # Return the object if it's already serializable

# for index, row in tqdm(df.iterrows(), total=len(df), desc="Saving JSON files", unit="file"):
#     # Convert the row to a dictionary
#     row_dict = row.to_dict()

#     # Convert any non-serializable objects (like ndarray) to serializable types
#     row_dict = {key: convert_to_serializable(value) for key, value in row_dict.items()}
    
#     # Define the file name based on the 'ID' column (or any other unique column)
#     file_name = f"{row['ID']}.json"
    
#     # Define the full path for the output JSON file
#     file_path = os.path.join(output_dir, file_name)
    
#     # Write the dictionary to a JSON file
#     with open(file_path, 'w') as json_file:
#         json.dump(row_dict, json_file, indent=4)


# print("All rows have been saved as JSON files.")
# debug=1


######################
# read file to lists #
######################
# #Function to read a file and return its contents as a list
# def read_file_to_list(file_path):
#     with open(file_path, 'r') as file:
#         return [line.strip() for line in file]

# # Paths to the ID and TITLE files
# id_file_path = 'pexels_section1_id_lists.txt'
# title_file_path = 'pexels_section1_caption_lists.txt'

# # Read the files
# id_list = read_file_to_list(id_file_path)
# title_list = read_file_to_list(title_file_path)

# # Print the length of each list (number of entries)
# print(f"Number of IDs: {len(id_list)}")
# print(f"Number of Titles: {len(title_list)}")

# # Additional check: Ensure both lists have the same length
# if len(id_list) == len(title_list):
#     print("The number of IDs matches the number of Titles.")
# else:
#     print("Mismatch between the number of IDs and Titles!")

#############################
# check laion-v2 duplicates #
#############################
# import pandas as pd

# # Function to read a text file into a list
# def read_file_to_list(file_path):
#     with open(file_path, 'r') as file:
#         return [line.strip() for line in file]

# # Paths to the ID and TITLE files
# id_file_path = '/weka/datasets/laion-v2-aesthetics-5.0_subset6M/file_lists/images_list.txt'
# title_file_path = '/weka/datasets/laion-v2-aesthetics-5.0_subset6M/file_lists/text_prompts_list.txt'

# # Read the files into lists
# id_list = read_file_to_list(id_file_path)
# title_list = read_file_to_list(title_file_path)

# # Ensure both lists have the same length
# if len(id_list) != len(title_list):
#     raise ValueError("The number of IDs does not match the number of Titles.")

# # Create a pandas DataFrame from the lists
# df = pd.DataFrame({
#     'ID': id_list,
#     'TITLE': title_list
# })

# # Print the DataFrame and its length
# print(f"DataFrame loaded with {len(df)} rows.")
# print(df.head())  # Display the first few rows of the DataFrame

# debug=1


##############
# 
##############
import pandas as pd
df_sec2=pd.read_parquet('/weka/datasets/pexels_section2/section2/meta/pexels_section2.parquet')

debug=1