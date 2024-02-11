"""
This script is designed to convert annotation files from COCO format to YOLO format. The COCO format is a popular JSON-based format for object detection that includes information such as image annotations, object instances, object segmentation, and categorization. The YOLO format is simpler, using plain text files where each line represents one object instance in an image, with the object class and bounding box coordinates normalized to the range [0, 1].

To use this script, ensure it is placed in the same directory as your COCO annotation file, typically named '_annotations.coco.json'. The script will create a new directory for the YOLO formatted annotations, converting each annotation in the COCO file into the YOLO format, where each image's annotations are stored in a separate text file with the same base name as the image file.

Usage:
1. Place this script in the same directory as your COCO format JSON annotation file.
2. Modify the 'coco_json' variable below if your COCO file has a different name.
3. Run the script. It will read the COCO annotations, convert them to YOLO format, and save them in the specified output directory.

Note: The following code should also work. I just found it after creating this script.
from ultralytics.data.converter import convert_coco
convert_coco(labels_dir='path/to/coco/annotations/')
"""
# Imports
import json
import os

# Path to your COCO JSON file
coco_json = '_annotations.coco.json'

# Output directory to store YOLO formatted annotation files
output_dir = 'yolo_annotations'

# Converts COCO JSON format annotations to YOLO format text files.
def convert_coco_to_yolo(coco_json, output_dir):
    # Load the COCO JSON file
    with open(coco_json) as file:
        data = json.load(file)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract image dimensions
    image_dimensions = {image['id']: (image['width'], image['height']) for image in data['images']}

    # Process annotations
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id'] - 1  # Subtract 1 to start class IDs from 0
        bbox = annotation['bbox']

        # Calculate normalized values
        file_name = next((img['file_name'] for img in data['images'] if img['id'] == image_id), None)
        if file_name is None:
            continue

        base_file_name = os.path.splitext(file_name)[0]
        width, height = image_dimensions[image_id]
        x_center = (bbox[0] + bbox[2] / 2) / width
        y_center = (bbox[1] + bbox[3] / 2) / height
        norm_width = bbox[2] / width
        norm_height = bbox[3] / height

        # Prepare the line to write to the file
        line = f"{category_id} {x_center} {y_center} {norm_width} {norm_height}\n"

        # Write to corresponding txt file
        output_file = os.path.join(output_dir, f"{base_file_name}.txt")
        with open(output_file, 'a') as f:
            f.write(line)

# Convert the dataset
convert_coco_to_yolo(coco_json, output_dir)
