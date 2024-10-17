import os
from PIL import Image

# Directories
processed_dir = '/weka/datasets/XC_Data/sam_vit_h_resize'  # Folder with processed images (with prefix label_mask_)
original_dir = '/weka/datasets/XC_Data/openimages_sample'  # Folder with original images

# Iterate over all processed images in the processed_dir
for file_name in os.listdir(processed_dir):
    if file_name.startswith('label_mask_') and file_name.endswith('.jpg'):
        # Remove 'label_mask_' prefix to find the original image
        original_file_name = file_name.replace('label_mask_', '')
        original_path = os.path.join(original_dir, original_file_name)
        processed_path = os.path.join(processed_dir, file_name)

        # Check if the original image exists
        if os.path.exists(original_path):
            # Load both images
            original_image = Image.open(original_path)
            processed_image = Image.open(processed_path)

            # Get dimensions of both images
            original_size = original_image.size  # (width, height)
            processed_size = processed_image.size  # (width, height)

            # Compare dimensions
            if original_size != processed_size:
                print(f"Dimension mismatch: {original_file_name} (original: {original_size}, processed: {processed_size})")
            else:
                print(f"Dimensions match: {original_file_name}")

        else:
            print(f"Original image not found for {file_name}")
