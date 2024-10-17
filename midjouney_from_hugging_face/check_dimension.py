import os
from PIL import Image

# Path to the images directory
# image_dir = '/weka/datasets/midjourney/from_kaggle/images'
image_dir = '/weka/datasets/midjourney/from_huggingface/image/'

# Function to check if the image is 2048x2048
def check_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        print('error')
        return None

# Iterate through all the files in the directory
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    
    # Check if the file is an image
    if os.path.isfile(image_path) and image_name.lower().endswith(('png', 'jpg', 'jpeg')):
        dimensions = check_image_dimensions(image_path)
        if dimensions is None:
            print(f"Could not open {image_name}.")