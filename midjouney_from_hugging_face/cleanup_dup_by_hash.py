###################
# get duplicates #
###################
# import pandas as pd
# from tqdm import tqdm
# df=pd.read_parquet('/weka/datasets/midjourney/from_huggingface/meta/midjourney_errfree.parquet')

# # Initialize a list to store the 'Duplicate_of' values
# duplicates = []

# # Dictionary to track the first occurrence of each hash
# hash_dict = {}

# # Iterate over each row in the DataFrame with tqdm for progress
# for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking for duplicates"):
#     image_hash = row['HASH']
    
#     if image_hash in hash_dict:
#         # If the hash already exists, mark this row as a duplicate of the first occurrence
#         duplicates.append(hash_dict[image_hash])
#     else:
#         # First occurrence of this hash, mark 'Duplicate_of' as None and track the first occurrence
#         hash_dict[image_hash] = row['id']
#         duplicates.append(None)

# # Add the "Duplicate_of" column to the DataFrame
# df['Duplicate_of'] = duplicates

# # Save the updated DataFrame to a new parquet file
# df.to_parquet('/weka/datasets/midjourney/from_huggingface/meta/midjourney_errfree.parquet')

# print("Updated DataFrame with Duplicate_of column has been saved.")
# debug=1

# ###################
# # clean up images #
# ###################
import os
import shutil
import pandas as pd
from tqdm import tqdm

# Load the DataFrame with the "Duplicate_of" column
df = pd.read_parquet('/weka/datasets/midjourney/from_huggingface/not_in_meta_images/composed_meta/composed_meta.parquet')

# Define directories
image_directory = '/weka/datasets/midjourney/from_huggingface/not_in_meta_images/converted_jpg'
duplicate_directory = '/weka/datasets/midjourney/from_huggingface/not_in_meta_images/duplicates'

# Ensure the duplicate directory exists
os.makedirs(duplicate_directory, exist_ok=True)
# Iterate over each row and handle duplicates
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing duplicates"):
    if row['Duplicate_of'] is not None:  # Only process if the 'Duplicate_of' column is not None
        # The current row is a duplicate
        original_id = row['Duplicate_of']
        duplicate_id = row['id']
        
        # Source file (the duplicate image) with .jpg extension
        source_file = os.path.join(image_directory, f"{duplicate_id}.jpg")
        
        if os.path.exists(source_file):
            # Move the duplicate image to the duplicate folder
            dest_file = os.path.join(duplicate_directory, f"{duplicate_id}.jpg")
            shutil.move(source_file, dest_file)
        
            # Create the corresponding .dup text file
            dup_file_path = os.path.join(image_directory , f"{duplicate_id}.jpg.dup")
            with open(dup_file_path, 'w') as dup_file:
                dup_file.write(f"{original_id}.jpg")  # Write the original image's ID that it's a duplicate of
        
            print(f"Moved {duplicate_id}.jpg and created {duplicate_id}.jpg.dup")
        else:
            print(f"File not found: {source_file}")
print("Duplicate images have been moved and .dup files created.")
