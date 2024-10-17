import tarfile
import os
from tqdm import tqdm

# Base directories for images and JSONs
tar_base_dir = '/weka/datasets/midjourney/from_huggingface/'
extract_dir = os.path.join(tar_base_dir, 'image')

# Create the 'image' directory if it doesn't exist
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

def extract_tar_file_ignore_errors(tar_path, extract_to):
    """
    Extracts a tar file and skips incomplete or corrupted files, continuing with the remaining files.
    
    :param tar_path: Path to the tar file
    :param extract_to: Directory where the contents will be extracted
    """
    if os.path.exists(tar_path):
        print(f"Extracting {tar_path}...")
        try:
            with tarfile.open(tar_path) as tar:
                members = tar.getmembers()
                for member in tqdm(members, desc="Extracting files", unit="file"):
                    try:
                        # Check if the file is likely incomplete (common in multi-part tar)
                        if member.size == 0:
                            print(f"Skipping incomplete file {member.name}")
                            continue
                        tar.extract(member, path=extract_to)
                    except tarfile.TarError as e:
                        print(f"Error extracting {member.name}: {e}")
            print(f"Finished extracting {tar_path}")
        except (tarfile.ReadError, tarfile.CompressionError) as e:
            print(f"Failed to open {tar_path}: {e}")
    else:
        print(f"{tar_path} not found. Skipping.")

def extract_all_tars(base_dir, start, end, extract_to):
    """
    Extracts all tar files from a range, ignoring errors and incomplete files.
    
    :param base_dir: Directory where the tar files are located
    :param start: Starting number of the tar files (inclusive)
    :param end: Ending number of the tar files (inclusive)
    :param extract_to: Directory where the contents will be extracted
    """
    for i in tqdm(range(start, end + 1), desc="Processing tar files", unit="file"):
        tar_filename = f'train_{i:04d}.tar'  # Format the filename to match train_XXXX.tar
        tar_path = os.path.join(base_dir, tar_filename)
        extract_tar_file_ignore_errors(tar_path, extract_to)

# Extract tar files from train_0000.tar to train_0109.tar
extract_all_tars(tar_base_dir, 7, 109, extract_dir)
