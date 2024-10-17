import random
import os
import json

# Default directory for storing the color map file
default_dir = './colormap_data'
os.makedirs(default_dir, exist_ok=True)  # Ensure the directory exists

# File to save the color map
color_map_file = os.path.join(default_dir, 'colormap.json')

# Generate a random color for each label
def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]  # RGB color

# Create and save the color map once
def create_color_map_once(labels):
    # Check if the color map already exists
    if os.path.exists(color_map_file):
        print(f"Color map already exists at {color_map_file}. Exiting without creating a new one.")
        return
    
    # Create a new color map
    color_map = {}
    for label in labels:
        color_map[label] = generate_random_color()

    # Save the color map to a file
    with open(color_map_file, 'w') as f:
        json.dump(color_map, f)

    print(f"Color map created and saved to {color_map_file}")

# List of labels (you should replace this with your actual labels)
labels = [str(i) for i in range(100)]  # Example: creating colors for labels 0-99

# Call the function to create and save the color map
create_color_map_once(labels)