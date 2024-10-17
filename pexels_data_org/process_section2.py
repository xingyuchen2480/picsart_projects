import os
import pandas as pd

# Load the DataFrame
df = pd.read_parquet('/weka/datasets/pexels_section2/section2/meta/pexels_section2_noterrfree.parquet')

# Directory containing the images
image_directory = '/weka/datasets/pexels_section2/section2/download/'

# Iterate through each row and check if the image exists
rows_to_keep = []

for index, row in df.iterrows():
    # Construct the full path to the image file based on the 'HASH' or relevant column
    image_path = os.path.join(image_directory, str(row['ID']) + '.jpg')  # Adjust extension if necessary
    
    # Check if the file exists
    if os.path.exists(image_path):
        rows_to_keep.append(index)

# Keep only rows where the image exists
df_cleaned = df.loc[rows_to_keep]

debug=1

# Save the cleaned DataFrame back to a parquet file (optional)
df_cleaned.to_parquet('/weka/datasets/pexels_section2/section2/meta/pexels_section2_cleaned.parquet')

print("Rows with missing images have been removed.")

debug=1