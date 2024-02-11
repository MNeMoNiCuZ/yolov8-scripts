"""
This script extracts the class names from a COCO format JSON annotation file and saves them into a text file in a format suitable for use in machine learning frameworks, specifically formatted for a YAML configuration file. The COCO (Common Objects in Context) format is a standard for object detection datasets, where 'categories' represent the different classes of objects annotated in the dataset.

To use this script, ensure it is in the same directory as your COCO annotation file, typically named '_annotations.coco.json'. The script reads the class names from the 'categories' section of the JSON file, formats them according to the YAML list format, and saves them to a specified output file. This output can then be used to configure models or datasets in machine learning projects that require a list of class names, especially in object detection tasks.

Usage:
1. Place this script in the same directory as your COCO format JSON annotation file.
2. If your annotation file has a different name from '_annotations.coco.json', update the 'coco_json' variable below.
3. Run the script. It will extract the class names, format them, and save them to the specified output file.
"""
# Imports
import json

# Path to your COCO JSON file
coco_json = '_annotations.coco.json'

# Output file path
output_file = 'classes.yaml'

# Extracts class names from a COCO JSON file and saves them in a YAML-friendly format.
def extract_classes_and_save(coco_json, output_file):
    # Load the COCO JSON file
    with open(coco_json) as file:
        data = json.load(file)

    # Extract class names
    classes = [category['name'] for category in data['categories']]
    num_classes = len(classes)

    # Format for YAML
    formatted_classes = ", ".join([f"'{cls}'" for cls in classes])
    yaml_content = f"nc: {num_classes} # number of classes\nnames: [{formatted_classes}]"

    # Save to file
    with open(output_file, 'w') as f:
        f.write(yaml_content)
    print(f"Classes saved to {output_file}")

# Extract classes and save
extract_classes_and_save(coco_json, output_file)