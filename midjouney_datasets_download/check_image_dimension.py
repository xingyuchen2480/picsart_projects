import os
from PIL import Image

# Path to the images directory
image_dir = '/weka/datasets/midjourney/from_kaggle/download/part9/'
#image_dir = '/weka/datasets/midjourney/from_huggingface/image/'

# Function to check if the image is 2048x2048
def check_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size == (2048, 2048)  # Check if the image size is 2048x2048
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return False

# Iterate through all the files in the directory
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    
    # Check if the file is an image and has the right dimensions
    if os.path.isfile(image_path) and image_name.lower().endswith(('png', 'jpg', 'jpeg')):
        if not check_image_dimensions(image_path):
            print(f"{image_name} does not have dimensions 2048x2048.")