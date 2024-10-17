import os
from PIL import Image
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

# Path to the images directory
image_dir = '/weka/datasets/midjourney/from_kaggle/download/part10'
cropped_output_dir = '/weka/datasets/midjourney/from_kaggle/image/part10'

# Create the output directory if it doesn't exist
os.makedirs(cropped_output_dir, exist_ok=True)

# Function to check if an image is valid
def is_image_valid(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify the image is valid
        return True
    except Exception:
        return False

# Function to crop a 2048x2048 image into four 1024x1024 images and save as JPG with quality 95 using PIL
def crop_image(image_name, progress_queue=None):
    image_path = os.path.join(image_dir, image_name)
    base_name = os.path.splitext(image_name)[0]  # Get the original base name before extension
    
    try:
        # Check if the image is valid before processing
        if not is_image_valid(image_path):
            if progress_queue:
                progress_queue.put(1)
            return f"{base_name} is invalid, skipping."

        # Check if the cropped files already exist
        existing_crops = [os.path.join(cropped_output_dir, f"{base_name}__crop{idx}.jpg") for idx in range(4)]
        if all(os.path.exists(cropped_img_path) for cropped_img_path in existing_crops):
            if progress_queue:
                progress_queue.put(1)
            return f"Cropped images for {base_name} already exist, skipping."

        with Image.open(image_path) as img:
            if img.size != (2048, 2048):
                if progress_queue:
                    progress_queue.put(1)
                return f"{base_name} is not 2048x2048, skipping."
            
            # Coordinates for cropping into four 1024x1024 images
            crops = [
                (0, 0, 1024, 1024),        # Top-left
                (1024, 0, 2048, 1024),     # Top-right
                (0, 1024, 1024, 2048),     # Bottom-left
                (1024, 1024, 2048, 2048),  # Bottom-right
            ]
            
            # Crop and save the images as JPG with quality 95 using PIL
            for idx, crop in enumerate(crops):
                cropped_img = img.crop(crop)
                cropped_img_name = f"{base_name}__crop{idx}.jpg"
                cropped_img.save(os.path.join(cropped_output_dir, cropped_img_name), quality=95)

            if progress_queue:
                progress_queue.put(1)

            return f"Cropped and saved {base_name}"
    except Exception as e:
        if progress_queue:
            progress_queue.put(1)
        return f"Error processing {base_name}: {e}"

# Function to process images with multiprocessing and tqdm
def process_images_multiprocessing():
    # Get the list of all PNG image names in the directory
    images_to_process = [image_name for image_name in os.listdir(image_dir) if image_name.lower().endswith('.png')]
    
    # Use a multiprocessing Queue to track progress
    with Manager() as manager:
        progress_queue = manager.Queue()
        
        # Initialize tqdm with the total number of images
        with tqdm(total=len(images_to_process), desc='Processing PNG images', unit='image') as pbar:
            # Define a callback to update tqdm progress bar
            def update_progress(*args):
                pbar.update()
            
            # Use multiprocessing to process the images in parallel
            with Pool(cpu_count()) as pool:
                # Map images to the pool with a progress queue for each task
                results = [
                    pool.apply_async(crop_image, args=(image_name, progress_queue), callback=update_progress)
                    for image_name in images_to_process
                ]
                
                # Wait for all the tasks to complete
                for r in results:
                    r.wait()

# Run the multiprocessing crop with tqdm progress bar

if __name__ == '__main__':
    process_images_multiprocessing()
