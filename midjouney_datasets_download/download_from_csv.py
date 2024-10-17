import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from PIL import Image
import shutil

# Create a directory to save the original images if it doesn't exist
output_dir = '/weka/datasets/midjourney/from_kaggle/download/part9_raw'
#cropped_output_dir = '/weka/datasets/midjourney/from_kaggle/images'
os.makedirs(output_dir, exist_ok=True)
#os.makedirs(cropped_output_dir, exist_ok=True)

df = pd.read_csv('/weka/datasets/midjourney/from_kaggle/meta/prompts_part9.csv')


def log_error(message):
    with open('/home/xingyu.chen/Project/midjouney_datasets_download/error_image.txt', 'a') as error_file:
        error_file.write(message + '\n')

# Function to download an image
def download_image(row):
    image_url = row['image_url']
    image_name = row['img_name']
    
    # Only download images with '__crop0'
    if pd.isna(image_name) or '__crop0' not in image_name:
        return
    
    file_path = os.path.join(output_dir, image_name)
    
    if os.path.exists(file_path):
        return f"Skipping {image_name}, already downloaded."
    
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return
        else:
            error_message = f"Failed to download {image_name}, status code: {response.status_code}"
            log_error(error_message)
            return error_message
    except Exception as e:
        error_message = f"Error downloading {image_name}: {e}"
        log_error(error_message)
        return error_message

# Multithreading for downloading images
def download_images_multithreaded(df, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image, row) for _, row in df.iterrows()]
        
        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                print(result)

# # Function to crop a 2048x2048 image into four 512x512 images
# def crop_image(image_name, progress_queue=None):
#     image_path = os.path.join(output_dir, image_name)
#     base_name = image_name.split('__crop')[0]
    
#     try:
#         # Check if the cropped files already exist
#         existing_crops = [os.path.join(cropped_output_dir, f"{base_name}__crop{idx}.png") for idx in range(4)]
#         if all(os.path.exists(cropped_img_path) for cropped_img_path in existing_crops):
#             if progress_queue:
#                 progress_queue.put(1)
#             return f"Cropped images for {base_name} already exist, skipping."

#         with Image.open(image_path) as img:
#             if img.size != (2048, 2048):
#                 if progress_queue:
#                     progress_queue.put(1)
#                 return f"{base_name} is not 2048x2048, skipping."
            
#             # Coordinates for cropping into four 512x512 images
#             crops = [
#                 (0, 0, 1024, 1024),
#                 (1024, 0, 2048, 1024),
#                 (0, 1024, 1024, 2048),
#                 (1024, 1024, 2048, 2048),
#             ]
            
#             # Crop and save the images
#             for idx, crop in enumerate(crops):
#                 cropped_img = img.crop(crop)
#                 cropped_img_name = f"{base_name}__crop{idx}.png"
#                 cropped_img.save(os.path.join(cropped_output_dir, cropped_img_name))

#             if progress_queue:
#                 progress_queue.put(1)

#             return f"Cropped and saved {base_name}"
#     except Exception as e:
#         if progress_queue:
#             progress_queue.put(1)
#         return f"Error processing {base_name}: {e}"

# Function to process images with multiprocessing and tqdm
# def process_images_multiprocessing():
#     images_to_process = [image_name for image_name in os.listdir(output_dir) if '__crop0' in image_name]
    
#     with Manager() as manager:
#         progress_queue = manager.Queue()
        
#         with tqdm(total=len(images_to_process)) as pbar:
#             def update_progress(*args):
#                 pbar.update()
            
#             with Pool(cpu_count()) as pool:
#                 results = [
#                     pool.apply_async(crop_image, args=(image_name, progress_queue), callback=update_progress)
#                     for image_name in images_to_process
#                 ]
                
#                 for r in results:
#                     r.wait()

# Download images and then crop them using multiprocessing
download_images_multithreaded(df, max_workers=8)
#process_images_multiprocessing()

# imfolder = '/weka/datasets/midjourney/from_kaggle/download/part6_raw'
# #imcfolder = '/weka/datasets/midjourney/from_kaggle/image/part2'
# imfiles = [imi for imi in os.listdir(imfolder) if imi.endswith('__crop0.png')]
# imfiles_newname = [imi.replace('__crop0.png', '.png') for imi in imfiles]

# imoutfolder = '/weka/datasets/midjourney/from_kaggle/download/part6'

# for imsrc, imdes in tqdm(zip(imfiles, imfiles_newname), total=len(imfiles)):
#     imsrc_fullpath = os.path.join(imfolder, imsrc)
#     imdes_fullpath = os.path.join(imoutfolder, imdes)
#     shutil.copy(imsrc_fullpath, imdes_fullpath)
