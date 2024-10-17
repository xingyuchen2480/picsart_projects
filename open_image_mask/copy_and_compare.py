import os
import shutil

# List of folders containing the .jpg images
folders_with_suffix = {
    '/weka/datasets/XC_Data/oneformer_swin_coco': '_swin_coco',
    '/weka/datasets/XC_Data/sam_vit_b': '_sam_vitb',
    '/weka/datasets/XC_Data/sam_vit_h': '_sam_vith',
    '/weka/datasets/XC_Data/sam_vit_l': '_sam_vitl',
    '/weka/datasets/XC_Data/sam_vit_b_resize': '_sam_vitb_resize',
    '/weka/datasets/XC_Data/sam_vit_h_resize': '_sam_vith_resize',
    '/weka/datasets/XC_Data/sam_vit_l_resize': '_sam_vitl_resize',
    '/weka/datasets/XC_Data/sam2.0_l': '_sam_2.0_l',
    '/weka/datasets/XC_Data/sam2.0_b+': '_sam_2.0_b+',
    '/weka/datasets/XC_Data/sam2.0_s': '_sam_2.0_s',
    '/weka/datasets/XC_Data/sam2.0_t': '_sam_2.0_t',
    '/weka/datasets/XC_Data/sam2.0_l_resize': '_sam_2.0_l_resize',
    '/weka/datasets/XC_Data/sam2.0_b+_resize': '_sam_2.0_b+_resize',
    '/weka/datasets/XC_Data/sam2.0_s_resize': '_sam_2.0_s_resize',
    '/weka/datasets/XC_Data/sam2.0_t_resize': '_sam_2.0_t_resize',
    '/weka/datasets/XC_Data/openimages_sample': '_original'
}


# Destination folder
destination_dir = '/weka/datasets/XC_Data/image_check'
os.makedirs(destination_dir, exist_ok=True)  # Create the folder if it doesn't exist

# Iterate over each folder and copy .jpg files to the destination folder with clean names
for folder, suffix in folders_with_suffix.items():
    for file_name in os.listdir(folder):
        if file_name.endswith('.jpg'):
            src_path = os.path.join(folder, file_name)

            # Clean up the filename: remove 'label_mask_' prefix if it exists
            cleaned_file_name = file_name.replace('label_mask_', '')

            # Add the suffix based on the folder
            cleaned_file_name = f"{os.path.splitext(cleaned_file_name)[0]}{suffix}.jpg"

            # Destination path for the file
            dst_path = os.path.join(destination_dir, cleaned_file_name)

            # Copy the file to the destination directory with the new cleaned name
            shutil.copy(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")

print("All .jpg files have been copied and renamed to the image_check folder.")
