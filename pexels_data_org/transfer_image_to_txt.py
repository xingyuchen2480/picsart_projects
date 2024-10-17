import os
import pandas as pd
import re

# Path to the image directory where images are stored
image_dir = '/weka/datasets/pexels_section2/section1/download'

# Load the parquet file
df = pd.read_parquet('/weka/datasets/pexels_section2/section1/meta/pexels_section1.parquet')

# Filter rows where both WIDTH and HEIGHT in the 'HW' column are >= 2000
sub_df = df[df['HW'].apply(lambda x: x[0] >= 2000 and x[1] >= 2000)]

# Filter out rows where DUPLICATE_OF is not None
sub_df = sub_df[sub_df['DUPLICATE_OF'].isna()]

# Function to clean up text (remove newlines, extra quotes, and other problematic symbols)
def clean_text(text):
    if pd.isna(text):
        return text
    # Remove newlines and strip leading/trailing whitespace
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    # Remove any surrounding quotes or special characters
    text = re.sub(r'[\'\"`]', '', text)
    return text

# Function to determine the correct file extension based on the file in the directory
def get_file_extension(image_dir, base_name):
    for ext in ['.jpg', '.jpeg', '.png', '.gif']:
        file_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(file_path):
            return ext
    return None  # Return None if no matching file is found

# Filter out rows where TITLE is NaN or empty
filtered_sub_df = sub_df[sub_df['TITLE'].notna() & (sub_df['TITLE'] != '')]

# Reset the index of the filtered DataFrame
filtered_sub_df = filtered_sub_df.reset_index(drop=True)

# Print the number of rows for debug purposes
print(f"Number of valid entries: {len(filtered_sub_df)}")
print(f"Number of IDs: {len(filtered_sub_df['ID'])}")
print(f"Number of Titles: {len(filtered_sub_df['TITLE'])}")

# Clean the 'ID' and 'TITLE' columns
filtered_sub_df['ID'] = filtered_sub_df['ID'].apply(clean_text)
filtered_sub_df['TITLE'] = filtered_sub_df['TITLE'].apply(clean_text)

# Check the actual file type for each ID based on the base name in the image directory
id_with_extension = []
for image_id in filtered_sub_df['ID']:
    base_name = os.path.splitext(image_id)[0]  # Remove any existing extension if present
    ext = get_file_extension(image_dir, base_name)
    
    if ext:
        id_with_extension.append(base_name + ext)
    else:
        print(f"Warning: No matching file found for ID {base_name}")
        id_with_extension.append(base_name)  # Append the base name if no extension found

# Replace the 'ID' column with the correct file paths (including extensions)
filtered_sub_df['ID'] = id_with_extension

# Save 'ID' and 'TITLE' columns to text files
filtered_sub_df['ID'].to_csv('pexels_section1_id_lists.txt', index=False, header=False)
filtered_sub_df['TITLE'].to_csv('pexels_section1_caption_lists.txt', index=False, header=False)

print("ID and TITLE lists have been saved with correct file types.")
