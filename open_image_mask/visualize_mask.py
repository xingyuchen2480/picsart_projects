import numpy as np
from PIL import Image
import cv2
import os
import json

# Path to the color map file
color_map_file = './colormap_data/colormap.json'

# Load the color map from the saved file
def load_color_map():
    with open(color_map_file, 'r') as f:
        color_map = json.load(f)
    return color_map

# Function to color the mask using the color map
def color_mask(label_mask, color_map):
    # Create a blank RGB image
    height, width = label_mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign colors to each label based on the color map
    unique_labels = np.unique(label_mask)
    for label in unique_labels:
        label_str = str(label)  # Ensure the label is treated as a string for JSON compatibility
        if label_str in color_map:
            color = color_map[label_str]
            colored_mask[label_mask == label] = color

    return colored_mask

# Directory paths
mask_dir = '/weka/datasets/XC_Data/sam2.0_b+_resize'  # Directory where the mask PNG files are saved
output_dir = '/weka/datasets/XC_Data/sam2.0_b+_resize'  # Same directory to save colored masks as JPG

# Load the global color map
color_map = load_color_map()

# Iterate over all mask files and color them
for mask_file in os.listdir(mask_dir):
    if mask_file.endswith('.png'):  # Ensure we only process PNG mask files
        mask_path = os.path.join(mask_dir, mask_file)

        # Load the mask
        label_mask = np.array(Image.open(mask_path))

        # Color the mask using the loaded color map
        colored_mask = color_mask(label_mask, color_map)

        # Save the colored mask as JPG
        output_path = os.path.join(output_dir, mask_file.replace('.png', '.jpg'))
        cv2.imwrite(output_path, colored_mask)

        print(f"Processed and saved colored mask: {output_path}")
