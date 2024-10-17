import os
from PIL import Image
from tqdm import tqdm

input_dir = '/weka/datasets/midjourney/from_huggingface/not_in_meta_images/converted_jpg'
#output_file = '/weka/datasets/midjourney/from_huggingface/invalid.txt'
#output_dir = '/weka/datasets/midjourney/from_kaggle/download/part10'

def is_image_valid(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verifies the file is an image
        return True
    except Exception as e:
        return False

def check_images_in_directory(directory):
    invalid_images = []
    total_files = sum(len(files) for _, _, files in os.walk(directory))
    
    with tqdm(total=total_files, desc='Checking images', unit='image') as pbar:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                    image_path = os.path.join(root, file)
                    if not is_image_valid(image_path):
                        invalid_images.append(image_path)
                    pbar.update(1)
    return invalid_images

# Check images and print results
invalid_images = check_images_in_directory(input_dir)
if invalid_images:
    print(f"Invalid images found: {len(invalid_images)}")

    # # Save the invalid images to a text file
    # with open(output_file, 'w') as f:
    #     for img in invalid_images:
    #         f.write(f"{img}\n")
else:
    print("All images are valid.")
    
    
debug=1