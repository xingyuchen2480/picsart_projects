import subprocess
from concurrent.futures import ThreadPoolExecutor

# Define the list of arguments for each run (e.g., gpu_id and image_part)
params = [(i, 0, i) for i in range(8)]

def run_script(gpu_id, image_major_part,image_sub_part):
    try:
        # Run sam_vit_h.py with gpu_id and image_part as arguments
        result = subprocess.run(["python", "sam_vit_h.py", str(gpu_id), str(image_major_part), str(image_sub_part)], check=True)
        print(f"sam_vit_h.py with gpu_id={gpu_id}, and image_major_part={image_major_part}, and image_sub_part={image_sub_part} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running sam_vit_h.py with gpu_id={gpu_id}  and image_part={image_major_part}, and image_part={image_sub_part}: {e}")

# Use ThreadPoolExecutor to run all scripts concurrently
if __name__ == "__main__":
    # Create a thread pool with 8 workers to run the script in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Map the parameters (gpu_id, image_part) to the run_script function
        executor.map(lambda p: run_script(p[0], p[1], p[2]), params)

    print("All sam_vit_h.py scripts are running in parallel with different parameters.")
