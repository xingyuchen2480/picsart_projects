import pandas as pd
import os
import json

import os
import pandas as pd
import json
from tqdm import tqdm

################################
# create combined data by json #
################################
# # Define directories
# json_base_dir = '/weka/datasets/midjourney/from_kaggle/json/'
# output_file = '/weka/datasets/midjourney/from_kaggle/meta/merged_metadata.parquet'

# # List to hold all the dataframes
# df_list = []

# # Count the total number of JSON files across all parts for the progress bar
# total_files = sum([len(files) for r, d, files in os.walk(json_base_dir) if files])

# # Use tqdm to track the progress of reading JSON files
# with tqdm(total=total_files, desc="Processing JSON files", unit="file") as pbar:
#     # Loop through each part directory (part1 to part10)
#     for part in range(1, 11):
#         part_dir = os.path.join(json_base_dir, f'part{part}')
#         # Loop through all JSON files in the current part directory
#         for json_file in os.listdir(part_dir):
#             if json_file.endswith('.json'):
#                 json_path = os.path.join(part_dir, json_file)
                
#                 # Read JSON file and append the data to the list
#                 with open(json_path, 'r') as f:
#                     data = json.load(f)
#                     df_list.append(pd.DataFrame([data]))  # Convert each JSON object to a DataFrame and append
                
#                 # Update the progress bar
#                 pbar.update(1)

# # Concatenate all DataFrames into a single DataFrame
# merged_df = pd.concat(df_list, ignore_index=True)

# # Save the merged DataFrame as a parquet file
# merged_df.to_parquet(output_file, index=False)

# print(f"Merged DataFrame has been saved to {output_file}")


#######################
# check combined data #
#######################

df=pd.read_parquet('/weka/datasets/midjourney/from_kaggle/meta/merged_metadata.parquet')

debug=1

duplicate_hashes = df[df.duplicated(subset=['HASH'], keep=False)]

# Print duplicate rows based on HASH
if not duplicate_hashes.empty:
    print(f"Found {len(duplicate_hashes)} duplicate records based on the HASH column:")
    print(duplicate_hashes)
else:
    print("All records are unique based on the HASH column.")

# Count unique HASH values
unique_hash_count = df['HASH'].nunique()
total_records = len(df)

print(f"\nTotal records: {total_records}")
print(f"Unique HASH values: {unique_hash_count}")
print(f"Duplicate entries (if any): {total_records - unique_hash_count}")

debug=1
