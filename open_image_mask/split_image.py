import os
import numpy as np

# Set directories
image_dir = "/weka/datasets/OpenImagesV6/train"
output_dir = "/home/xingyu.chen/Project/open_image_mask/text_files"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Get the list of image files and split them into 2 major parts
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
major_splits = np.array_split(image_files, 2)

# Split each major part into 8 subsections
for major_part_idx, major_split in enumerate(major_splits):
    subsections = np.array_split(major_split, 8)
    
    # Write each subsection into a separate text file
    for i, split in enumerate(subsections):
        output_file = os.path.join(output_dir, f"image_split_major_{major_part_idx + 1}_part_{i + 1}.txt")
        
        # Write image file names to the text file
        with open(output_file, 'w') as f:
            for image_file in split:
                f.write(f"{image_file}\n")

print("Image file splits have been saved to individual text files.")
