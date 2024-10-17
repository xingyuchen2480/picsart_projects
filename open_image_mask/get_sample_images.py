import os
import shutil

# Define the source and destination directories
src_dir = "/weka/datasets/OpenImagesV6/train"
dst_dir = "/weka/datasets/openimages_sample"

# Check if the source directory exists
if not os.path.exists(src_dir):
    print(f"Source directory does not exist: {src_dir}")
else:
    # Get the list of all jpg images in the source directory
    images = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]

    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Copy top 20 images to the destination directory
    for img in images[:20]:
        src_img_path = os.path.join(src_dir, img)
        dst_img_path = os.path.join(dst_dir, img)
        shutil.copy(src_img_path, dst_img_path)

    print(f"Copied {len(images[:20])} images to {dst_dir}")