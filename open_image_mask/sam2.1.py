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
from hydra import initialize, compose  # Specific functions from Hydra
from hydra.core.global_hydra import GlobalHydra

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
output_dir = "/weka/datasets/XC_Data/sam2.0_l"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

#GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
#hydra.initialize_config_module('/home/xingyu.chen/Project/open_image_mask/sam2/sam2/', version_base='1.2')
checkpoint = "sam2_hiera_large.pt"
# this should work now
model = build_sam2('sam2_hiera_l.yaml', checkpoint)
# Initialize the SAM2 mask generator
mask_generator = SAM2AutomaticMaskGenerator(model)



start_time = time.time()  # Start the timer

for img_file in os.listdir(sample_dir):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(sample_dir, img_file)

        # Load the image
        image= cv2.imread(img_path)
        if image.shape[2] == 4:  # If the image has 4 channels, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # transform = transforms.ToTensor()  # This will convert the image to a tensor (C, H, W) with values in range [0, 1]
        # image_tensor = transform(image).to(device)

        with torch.amp.autocast(device_type='cuda'):  # Updated autocast usage
            masks = mask_generator.generate(image) 

        # Create an empty mask to store the labels
        height, width, _ = image.shape
        label_mask = np.zeros((height, width), dtype=np.uint32)

        # Iterate over the masks and assign unique labels
        for idx, mask in enumerate(masks):
            mask_binary = mask['segmentation']
            label_mask[mask_binary] = idx
        label_mask_uint32 = label_mask.astype(np.uint32)
        label_mask_pil = Image.fromarray(label_mask_uint32)
        # Save the label mask for the image
        output_path = os.path.join(output_dir, f"label_mask_{os.path.splitext(img_file)[0]}.png")
        label_mask_pil.save(output_path)

        print(f"Processed {img_file}, saved mask to {output_path}")
        
end_time = time.time()  # End the timer
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")