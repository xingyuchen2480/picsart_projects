import os
import os.path as osp

# Path to the directory with the images
image_dir = '/weka/datasets/midjourney/from_kaggle/download/part10_raw'

# Loop through all files in the directory
for file_name in os.listdir(image_dir):
    if file_name.endswith('__crop0.png'):
        old_path = osp.join(image_dir, file_name)
        # Remove '__crop0' from the file name
        new_file_name = file_name.replace('__crop0', '')
        new_path = osp.join(image_dir, new_file_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed {file_name} to {new_file_name}")

print("Renaming complete.")