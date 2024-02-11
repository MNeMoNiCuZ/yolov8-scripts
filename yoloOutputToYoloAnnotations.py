"""
This script is designed to remap detection outputs from YOLO text format to the YOLO training format. It processes a given folder containing image files and their corresponding detection annotation files. For each image and its associated detection file, the script reads the detections, converts their bounding box coordinates from absolute pixel values to normalized values relative to the image dimensions, and then writes these normalized values to a new output file in YOLO format.

Before running the script, ensure the 'folder_path' variable is set to the directory containing your images and detection files. The detection files should be in the format '<image_name>.txt', with each line representing a detected object in the format 'class_name confidence x_start y_start x_end y_end'.

Run this script from the root of the project.
"""

from PIL import Image
import os

# Default folder paths - These can be the same if images and text files are in the same folder
images_folder = 'output/overlays'  # Folder for images
texts_folder = 'output/detections'  # Folder containing text files with detection

# Mapping of class names to class IDs
class_ids = {
    'watermark': 0,
    'rabbits': 1
    # Add more class names and IDs as needed
}

# Converts detection formats from one type to YOLO format and saves to an output file
def convert_detections(input_file, image_file, output_file):
    # Open the image file to get dimensions
    with Image.open(image_file) as img:
        image_width, image_height = img.size
        print(f"Image dimensions for {image_file}: width={image_width}, height={image_height}")

    # Read detections from the input file and write normalized values to the output file
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            detections = line.strip().split(', ')
            for detection in detections:
                parts = detection.split()
                # Validate detection format
                if len(parts) != 6:
                    print(f"Skipping invalid detection: {detection}")
                    continue

                # Extract detection details
                class_name, confidence, x_start, y_start, x_end, y_end = parts
                class_id = class_ids.get(class_name, -1)  # Use -1 for unknown class names

                if class_id == -1:
                    print(f"Unknown class '{class_name}', skipping.")
                    continue

                # Convert absolute coordinates to normalized values
                x_start, y_start, x_end, y_end = map(float, [x_start, y_start, x_end, y_end])
                x_center_ratio = ((x_start + x_end) / 2) / image_width
                y_center_ratio = ((y_start + y_end) / 2) / image_height
                width_ratio = (x_end - x_start) / image_width
                height_ratio = (y_end - y_start) / image_height

                # Write the converted detection to the output file
                outfile.write(f"{class_id} {x_center_ratio} {y_center_ratio} {width_ratio} {height_ratio}\n")

# Processes all images in a folder and converts their detections to the desired format
def process_folders(images_folder, texts_folder):
    for text_file in os.listdir(texts_folder):
        base, ext = os.path.splitext(text_file)
        if ext.lower() == '.txt':
            # Construct paths for corresponding image and output files
            corresponding_image_file = base + '.jpg'  # Change extension if needed
            image_file_path = os.path.join(images_folder, corresponding_image_file)
            text_file_path = os.path.join(texts_folder, text_file)
            output_file_path = os.path.join(texts_folder, base + '_converted.txt')

            # Process the files if the corresponding image exists
            if os.path.exists(image_file_path):
                print(f"Processing {base}...")
                convert_detections(text_file_path, image_file_path, output_file_path)
            else:
                print(f"No corresponding image file for {text_file_path}")

# Run the process on the specified folder
process_folders(images_folder, texts_folder)