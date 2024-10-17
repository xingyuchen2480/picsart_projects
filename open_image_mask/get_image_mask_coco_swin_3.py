from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import os
import torch
from tqdm import tqdm
import numpy as np
#model = OneFormerForUniversalSegmentation.

# Set directories
image_dir = "/weka/datasets/OpenImagesV6/train"
output_dir = "/weka/datasets/OpenImagesV6/instance_segmentation_label/train_swin_l"

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

gpu_id = 3  # Change this to the desired GPU (e.g., 1 for GPU 1)
torch.cuda.set_device(gpu_id)

input_file = f"image_split_part_{gpu_id}.txt"
with open(input_file, 'r') as f:
    image_files = [line.strip() for line in f]  # Remove newline characters

# Set the device to GPU if available, otherwise CPU
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model and processor
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
model.eval()

# Function to process a single image
def process_image(model, processor, image_path, output_dir):
    # Load and preprocess the image
    image = Image.open(image_path)
    original_size = image.size
    
    if image.mode != "RGB":
        image = image.convert('RGB')

    # Preprocess the image
    inputs = processor(image, ["panoptic"], return_tensors="pt")
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Ensure input tensors are on the same device

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process and save mask
    predicted_panoptic_map = processor.post_process_panoptic_segmentation(outputs, target_sizes=[original_size[::-1]])[0]["segmentation"]
    predicted_panoptic_map = predicted_panoptic_map.cpu().numpy()

    # Save the mask
    mask_image = Image.fromarray(predicted_panoptic_map.astype('uint32'))
    mask_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    mask_path = os.path.join(output_dir, mask_filename)

    mask_image.save(mask_path)

# Process images one by one
for image_file in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(image_dir, image_file)
    process_image(model, processor, image_path, output_dir)