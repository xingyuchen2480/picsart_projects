##################
# oneformer test #
##################

# import torch
# from PIL import Image
# import requests
# from transformers import OneFormerProcessor, OneFormerModel

# # download texting image
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# # load processor for preprocessing the inputs
# processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
# model = OneFormerModel.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
# inputs = processor(image, ["semantic"], return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# mask_predictions = outputs.transformer_decoder_mask_predictions
# class_predictions = outputs.transformer_decoder_class_predictions


# debug=1
# f"👉 Mask Predictions Shape: {list(mask_predictions.shape)}, Class Predictions Shape: {list(class_predictions.shape)}"
# '👉 Mask Predictions Shape: [1, 150, 128, 171], Class Predictions Shape: [1, 150, 151]'


###########################
# generation process test #
###########################
# from PIL import Image
# import os
# #import matplotlib.pyplot as plt
# import numpy as np
# import os
# import shutil

# # Directories for images and masks
# image_dir = "/weka/datasets/XC_Data"  # Update with your actual image directory
# mask_dir = "/weka/datasets/XC_Data"   # Update with your actual mask directory

# output_dir = "/weka/datasets/XC_Data/universal_masks_check"  # Directory to save human-readable masks

# # Create output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Define a function to convert the mask into a color map for human readability
# def convert_mask_to_readable_format(mask):
#     # Define a color palette: Each index corresponds to a color for a class
#     colors = np.array([
#     [0, 0, 0],        # Class 0: Background (black)
#     [255, 0, 0],      # Class 1: Red
#     [0, 255, 0],      # Class 2: Green
#     [0, 0, 255],      # Class 3: Blue
#     [255, 255, 0],    # Class 4: Yellow
#     [255, 0, 255],    # Class 5: Magenta
#     [0, 255, 255],    # Class 6: Cyan
#     [128, 128, 128],  # Class 7: Gray
#     [128, 0, 128],    # Class 8: Purple
#     [255, 165, 0],    # Class 9: Orange
#     [255, 20, 147],   # Class 10: Deep Pink
#     [75, 0, 130],     # Class 11: Indigo
#     [173, 255, 47],   # Class 12: Green Yellow
#     [0, 100, 0],      # Class 13: Dark Green
#     [0, 191, 255],    # Class 14: Deep Sky Blue
#     [165, 42, 42],    # Class 15: Brown
#     [0, 128, 128],    # Class 16: Teal
#     [192, 192, 192],  # Class 17: Silver
#     [255, 228, 196],  # Class 18: Bisque
#     [139, 69, 19],    # Class 19: Saddle Brown
#     [70, 130, 180],   # Class 20: Steel Blue
#     [238, 130, 238],  # Class 21: Violet
#     [0, 255, 127],    # Class 22: Spring Green
#     [255, 140, 0],    # Class 23: Dark Orange
#     [47, 79, 79],     # Class 24: Dark Slate Gray
#     [46, 139, 87],    # Class 25: Sea Green
#     [220, 20, 60],    # Class 26: Crimson
#     [255, 69, 0],     # Class 27: Orange Red
#     [255, 250, 205],  # Class 28: Lemon Chiffon
#     [255, 222, 173],  # Class 29: Navajo White
#     [128, 0, 0],      # Class 30: Maroon
#     [34, 139, 34],    # Class 31: Forest Green
#     [255, 255, 240],  # Class 32: Ivory
#     [127, 255, 212],  # Cl=100ass 33: Aquamarine
#     [218, 165, 32],   # Class 34: Goldenrod
#     [123, 104, 238],  # Class 35: Medium Slate Blue
#     [0, 206, 209],    # Class 36: Dark Turquoise
#     [75, 0, 130],     # Class 37: Indigo
#     [199, 21, 133],   # Class 38: Medium Violet Red
#     [135, 206, 250]   # Class 39: Light Sky Blue
# ])

#     mask = np.where(mask > 39, 39, mask)
#     # Convert mask (which contains class indices) to a color image
#     color_mask = colors[mask]
    
#     return color_mask

# # Function to save the human-readable mask and copy corresponding image
# def save_human_readable_ma=100ask as a PNG image
#         base_name = os.path.splitext(os.path.basename(mask_path))[0]
#         color_mask_image = Image.fromarray(color_mask.astype('uint8'))
#         color_mask_image.save(os.path.join(output_dir, base_name + "_readable_mask.png"))
#         print(f"Saved human-readable mask: {base_name}_readable_mask.png")

#         # Copy the corresponding image to the output directory
#         shutil.copy(image_path, os.path.join(output_dir, os.path.basename(image_path)))
#         print(f"Copied image: {os.path.basename(image_path)}")

#     except Exception as e:
#         print(f"Error processing mask or image: {str(e)}")

# # Process masks and copy corresponding images
# for img_file in os.listdir(image_dir):
#     if img_file.endswith(".jpg"):
#         # Derive mask filename from image basename
#         base_name = os.path.splitext(img_file)[0]
#         mask_path = os.path.join(mask_dir, base_name + ".png")
#         image_path = os.path.join(image_dir, img_file)
#         output_path = os.path.join(output_dir, base_name + "_readable_mask.png")

#         # Check if the mask exists
#         if os.path.exists(mask_path):
#             save_human_readable_mask_and_copy_image(image_path, mask_path, output_dir)



#############
# get model #
#############
# import requests

# url = "https://shi-labs.com/projects/oneformer/mapillary/250_16_intern_image_h_oneformer_mapillary_300k_1024.pth"
# response = requests.get(url)
# with open("model_checkpoint.pth", "wb") as f:
#     f.write(response.content)


###################
# check image dim #
############import os
from PIL import Image
import os
from tqdm import tqdm

# Directory containing images
image_dir = '/weka/datasets/OpenImagesV6/train'  # Replace with your directory path

# Initialize variables to store the largest image details
largest_image = None
max_area = 0  # To track the largest area (width * height)

# Get the list of image files
image_files = [file_name for file_name in os.listdir(image_dir) if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

# Iterate over all files in the directory with a progress bar
for file_name in tqdm(image_files, desc="Checking images"):
    image_path = os.path.join(image_dir, file_name)

    # Open the image and check its dimensions
    with Image.open(image_path) as img:
        width, height = img.size  # Get dimensions
        area = width * height     # Calculate area

        # If this image is larger than the previously recorded largest, update the largest
        if area > max_area:
            max_area = area
            largest_image = (file_name, width, height)

# Print the largest image details
if largest_image:
    print(f"Largest image: {largest_image[0]} with dimensions {largest_image[1]}x{largest_image[2]}")

print("Largest image dimension check complete.")
