from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from transformers import T5Tokenizer, T5Model


caption=''

# Load CLIP ViT-L/14 (L14)
model_l14 = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor_l14 = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

inputs = processor_l14(text=caption, return_tensors="pt", padding=True)
text_features_l14 = model_l14.get_text_features(**inputs)


# Load T5-XXL model
t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-xxl-lm-adapt")
t5_model = T5Model.from_pretrained("google/t5-xxl-lm-adapt")

inputs = t5_tokenizer(caption, return_tensors="pt")
text_features_t5xxl = t5_model(**inputs).last_hidden_state