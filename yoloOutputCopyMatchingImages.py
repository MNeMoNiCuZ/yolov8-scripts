"""
This script is designed to filter and copy selected image files based on their presence in two distinct directories. The primary use case is to identify image files that have corresponding annotation files in one directory ('curated_dir') and also exist in another directory ('original_dir'). Once identified, these selected image files are copied from the 'original_dir' to a third directory ('matching_dir'), intended for further processing or analysis.

In simple terms, use the script to find the original image files based on a curated selection of images, which may have been modified (scaled down, compressed, overlaid with prediction graphics etc.)

The script operates as follows:
1. It reads the names of image files from the 'curated_dir', assuming that the presence of image files in this directory indicates that it has a corresponding annotation file.
2. It then checks if these image files also exist in the 'original_dir' directory, ensuring that only files present in both directories are considered for copying.
3. Files that meet the above criteria are copied to the 'matching_dir', creating this directory if it does not already exist.
4. Throughout the process, the script logs its actions, providing feedback on the files being processed and any discrepancies encountered, such as missing files in expected locations.

This tool is particularly useful in machine learning workflows where a subset of images from a larger dataset (represented by 'original_dir') needs to be isolated based on specific criteria, such as the availability of annotations.

Run this script from the root of the project.
"""
import os
import shutil

# Directory paths configuration
original_dir = 'input/original'  # Directory containing all image files, this is the source image folder, it does not need to have any annotations or captions. This is where the script will copy the matching images from.
curated_dir = 'input/curated'  # Directory containing your selection of images to keep, but the images are downsized or have overlays saved over the images.
matching_dir = 'input/matching'  # Destination directory for selected image files, consider this the output directory.

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png']

# Script execution starts here
print(f"Pairs directory: {curated_dir}")
print(f"All directory: {original_dir}")
print(f"Selected directory: {matching_dir}")

# Create the matching directory if it doesn't exist
if not os.path.exists(matching_dir):
    os.makedirs(matching_dir)
    print(f"Created matching directory: {matching_dir}")

# Function to check if file extension matches supported image types
def is_supported_image(file_name):
    return any(file_name.lower().endswith(ext) for ext in image_extensions)

# List all supported image files in the curated directory
curated_files = [f for f in os.listdir(curated_dir) if is_supported_image(f)]
print(f"Image files in curated directory: {curated_files}")

# Loop through each file in the curated directory
for file_name in curated_files:
    # Construct the full file path in both original and curated directories
    original_file_path = os.path.join(original_dir, file_name)
    curated_file_path = os.path.join(curated_dir, file_name)

    # Check if the image file exists in both original and curated directories
    if os.path.exists(original_file_path) and os.path.exists(curated_file_path):
        # Construct the destination path in the matching directory
        matching_file_path = os.path.join(matching_dir, file_name)

        # Copy the file from the original directory to the matching directory
        shutil.copy(original_file_path, matching_file_path)
        print(f"Copied from {original_file_path} to {matching_file_path}")
    else:
        # Print error messages if files are missing in either directory
        if not os.path.exists(original_file_path):
            print(f"Original file does not exist: {original_file_path}")
        if not os.path.exists(curated_file_path):
            print(f"Curated file does not exist: {curated_file_path}")

print("Process completed.")