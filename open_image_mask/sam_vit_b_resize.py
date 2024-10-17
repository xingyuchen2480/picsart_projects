from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import os
import time
from PIL import Image
from torchvision import transforms
# Set the desired GPU ID
gpu_id = 0

if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")



# Path to the sample images
sample_dir = "/weka/datasets/XC_Data/openimages_sample"
output_dir = "/weka/datasets/XC_Data/sam_vit_b_resize"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth").to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

start_time = time.time()  # Start the timer

for img_file in os.listdir(sample_dir):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(sample_dir, img_file)

        # Load the image
        image= cv2.imread(img_path)
        original_height, original_width, _ = image.shape

        # Resize the image to half of its original dimensions
        half_width = original_width // 2
        half_height = original_height // 2
        resized_image = cv2.resize(image, (half_width, half_height))

        # transform = transforms.ToTensor()  # This will convert the image to a tensor (C, H, W) with values in range [0, 1]
        # image_tensor = transform(image).to(device)

        masks = mask_generator.generate(resized_image)

        # Create an empty mask to store the labels
        # Create an empty mask to store the labels (with the size of the resized image)
        resized_label_mask = np.zeros((half_height, half_width), dtype=np.uint16)


        for idx, mask in enumerate(masks):
            mask_binary = mask['segmentation']
            resized_label_mask[mask_binary] = idx  # Use (idx + 1) to avoid label 0
            
        # Resize the label mask back to the original size
        original_label_mask = cv2.resize(resized_label_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        label_mask_uint32 = original_label_mask.astype(np.uint32)
        label_mask_pil = Image.fromarray(label_mask_uint32)
        # Save the label mask for the image
        output_path = os.path.join(output_dir, f"label_mask_{os.path.splitext(img_file)[0]}.png")
        label_mask_pil.save(output_path)

        print(f"Processed {img_file}, saved mask to {output_path}")
        
end_time = time.time()  # End the timer
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")