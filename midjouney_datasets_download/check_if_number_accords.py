import os

# Base directories for images and JSONs
image_base_dir = '/weka/datasets/midjourney/from_kaggle/image'
json_base_dir = '/weka/datasets/midjourney/from_kaggle/json'

# List of parts (subdirectories) to check
parts = ['part1', 'part2', 'part3', 'part4', 'part5', 'part6', 'part7', 'part8', 'part9', 'part10']

def get_basenames(directory, extension):
    """Returns a set of basenames (without extensions) for all files with the specified extension."""
    return set(os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(extension))

def check_image_and_json_dirs(image_base_dir, json_base_dir, parts):
    for part in parts:
        image_part_dir = os.path.join(image_base_dir, part)
        json_part_dir = os.path.join(json_base_dir, part)

        if not os.path.exists(image_part_dir) or not os.path.exists(json_part_dir):
            print(f"Directory for {part} is missing in either image or JSON directory.")
            continue

        # Get basenames of images and JSONs
        image_basenames = get_basenames(image_part_dir, '.jpg')  # Assuming images have .jpg extension
        json_basenames = get_basenames(json_part_dir, '.json')  # Assuming JSON files have .json extension

        # Compare basenames
        missing_json = image_basenames - json_basenames
        missing_images = json_basenames - image_basenames

        if not missing_json and not missing_images:
            print(f"{part}: All images have corresponding JSONs and vice versa.")
        else:
            if missing_json:
                print(f"{part}: Missing JSONs for images: {sorted(missing_json)}")
            if missing_images:
                print(f"{part}: Missing images for JSONs: {sorted(missing_images)}")

if __name__ == '__main__':
    check_image_and_json_dirs(image_base_dir, json_base_dir, parts)
