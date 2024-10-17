import pandas as pd
import hashlib
import os
import json
from tqdm import tqdm
import numpy as np

# Load the DataFrame
df = pd.read_parquet('/weka/datasets/midjourney/from_huggingface/meta/midjourney_clean.parquet')

# Define the output directory for the JSON files
output_dir = '/weka/datasets/midjourney/from_huggingface/json'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Iterate over each row in the DataFrame with tqdm progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Writing JSON files with hashes"):
    # Convert the row to a dictionary
    row_dict = row.to_dict()

    # Ensure all 'ndarray' types are converted to lists for JSON serialization
    for key, value in row_dict.items():
        if isinstance(value, np.ndarray):
            row_dict[key] = value.tolist()

    # Define the file name based on a column value (e.g., using 'id')
    file_name = f"{row['id']}.json"
    
    # Define the full path for the output JSON file
    file_path = os.path.join(output_dir, file_name)
    
    # Write the dictionary (including the hash) to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(row_dict, json_file, indent=4)

# Print a message when done
print("All JSON files with hashes have been written.")
