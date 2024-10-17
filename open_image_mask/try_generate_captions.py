from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from PIL import Image
from tqdm import tqdm
import json
import sys

# from torchvision import transforms


# Set the desired GPU ID and image part ID
# gpu_id = 0
# image_part = 0

def process_images(gpu_id, image_major_part, image_sub_part):
    # Set the device based on the GPU ID
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    # Path to the sample images
    image_dir = "/weka/datasets/XC_Data/openimages_sample"
    output_dir = "/weka/datasets/OpenImagesV6/XC_Data"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    input_file = f"text_files/image_split_major_{image_major_part+1}_part_{image_sub_part+1}.txt"
    with open(input_file, 'r') as f:
        image_files = [line.strip() for line in f]  # Remove newline characters

    sam = sam_model_registry["vit_h"](checkpoint="/weka/datasets/XC_Data/ckp/sam_vit_h_4b8939.pth").to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    max_dim = 2048
    for img_file in tqdm(image_files, desc="Processing Images"):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(image_dir, img_file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}.png")
            json_path = os.path.join("/weka/datasets/OpenImagesV6/instance_segmentation_label/train_json", f"{os.path.splitext(img_file)[0]}.json")
            if os.path.exists(json_path):
                continue
            # Load the image
            image= cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            #resize
            # Get the original dimensions
            if height > max_dim or width > max_dim:
                if height > width:
                    scale = max_dim / height
                    
                else:
                    scale = max_dim / width
                new_width = int(width * scale)
                new_height = int(height * scale)
            else:
                new_width = width
                new_height =height
            # Set the maximum dimension to 2048

            image_resized = cv2.resize(image_rgb, (new_width, new_height))
            # # Resize the image to half of its original dimensions
            # half_width = original_width // 2
            # half_height = original_height // 2
            # resized_image = cv2.resize(image, (half_width, half_height))

            # masks = mask_generator.generate(resized_image)

            # Create an empty mask to store the labels
            masks = mask_generator.generate(image_resized)
            
            #########################
            # try generate captions #
            #########################
            
            
            
            
            
            
            
            
            
            label_mask = np.zeros((new_height, new_width), dtype=np.uint32)

            # Iterate over the masks and assign unique labels
            for idx, mask in enumerate(masks):
                mask_binary = mask['segmentation']
                label_mask[mask_binary] = idx
            
            label_mask_32 = label_mask.astype(np.uint16)
            label_mask_32_restored = cv2.resize(label_mask_32, (width, height))
            label_mask_uint32_restored=label_mask_32_restored.astype(np.uint32)
            label_mask_pil = Image.fromarray(label_mask_uint32_restored)
            # Save the label mask for the image
            label_mask_pil.save(output_path)
            # Create a new mask dictionary without the 'segmentation' key
            json_output = {
                "size": {
                    "original_hw": [height, width],
                    "resized_hw": [new_height, new_width]
                },
                "masks": []
}
            for mask in masks:
                # Create a new mask dictionary without the 'segmentation' key
                filtered_mask = {k: v for k, v in mask.items() if k != 'segmentation'}
                json_output["masks"].append(filtered_mask)
            with open(f'/weka/datasets/OpenImagesV6/instance_segmentation_label/train_json/{os.path.splitext(img_file)[0]}.json', 'w') as f:
                json.dump(json_output, f, indent=4)

            print(f"Processed {img_file}, saved mask to {output_path}")
                  
if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    image_major_part = int(sys.argv[2])
    image_sub_part = int(sys.argv[3])
    process_images(gpu_id, image_major_part,image_sub_part)