from huggingface_hub import snapshot_download
import zipfile
import os
#login(token='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')
# Define model repo or dataset and file path
repo_id = "terminusresearch/midjourney-v6-520k-raw"

destination_folder = "/weka/datasets/midjourney/from_huggingface/"
os.makedirs(destination_folder, exist_ok=True)
snapshot_dir = snapshot_download(repo_id=repo_id, local_dir=destination_folder,repo_type="dataset")

print(f"Dataset downloaded to: {snapshot_dir}")