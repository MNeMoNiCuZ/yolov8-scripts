# Imports
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Directories
input_dir = Path('./generate_input')
output_dir = Path('./generate_output')

# Image output settings
image_output_dir = output_dir / "overlays"  # Specify the directory for processed images
image_output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
image_prefix = "" # Image prefix, can be empty
image_suffix = "" # Image suffix, can be empty

# Text output settings
text_output_dir = output_dir / "detections"
text_output_dir.mkdir(parents=True, exist_ok=True)
text_prefix = ""  # Text prefix, can be empty
text_suffix = ""  # Text suffix, can be empty

# Load your trained model
model_path = './models/watermarks_s_yolov8_v1.pt'
model = YOLO(model_path)

# Classes to detect
# Example: ['SpeechBalloons', 'General_speech', 'hit_sound', 'blast_sound', 'narration speech', 'thought_speech', 'roar']
selected_classes = ['watermark']

# Class override mapping, treats the left side of the mapping as if it was the class of the right side
# Example: thought_speech annotations will be treated as SpeechBalloons annotations.
class_overrides = {
    'thought_speech': 'SpeechBalloons',
}

# Confidence threshold, minimum detection confidence to consider success
confidence_threshold = 0.15

# Label settings
label_boxes = True  # Set to True to draw class names, False for just boxes
font_size = 30  # Font size for the class labels

try:
    font = ImageFont.truetype("arial.ttf", 30)  # Update font size as needed
except IOError:
    font = ImageFont.load_default()
    print("Default font will be used, as custom font not found.")

# Label colors by index
predefined_colors_with_text = [
    ((204, 0, 0),     'white'),  # Darker red, white text
    ((0, 204, 0),     'black'),  # Darker green, black text
    ((0, 0, 204),     'white'),  # Darker blue, white text
    ((204, 204, 0),   'black'),  # Darker yellow, black text
    ((204, 0, 204),   'white'),  # Darker magenta, white text
    ((0, 204, 204),   'black'),  # Darker cyan, black text
    ((153, 0, 0),     'white'),  # Darker maroon, white text
    ((0, 153, 0),     'white'),  # Darker green, white text
    ((0, 0, 153),     'white'),  # Darker navy, white text
    ((153, 153, 0),   'black'),  # Darker olive, black text
    # Add more color pairs if needed
]

# Assign colors to each class
class_colors = {class_name: predefined_colors_with_text[i][0] for i, class_name in enumerate(selected_classes)}
text_colors = {class_name: predefined_colors_with_text[i][1] for i, class_name in enumerate(selected_classes)}

# Store input images in a variable
image_paths = []
for extension in ['*.jpg', '*.jpeg', '*.png']:
    image_paths.extend(input_dir.glob(extension))

# Function to estimate text size
def estimate_text_size(label, font_size):
    approx_char_width = font_size * 0.6
    text_width = len(label) * approx_char_width
    text_height = font_size
    return text_width, text_height

def write_detections_to_file(image_path, detections):
    # Create a text file named after the image
    text_file_path = text_output_dir / f"{text_prefix}{image_path.stem}{text_suffix}.txt"

    with open(text_file_path, 'w') as file:
        for detection in detections:
            file.write(f"{detection}\n")

# Process images with progress bar
for image_path in tqdm(image_paths, desc='Processing Images'):
    img = Image.open(image_path)
    results = model.predict(img)
    draw = ImageDraw.Draw(img)
    detections = []

    if len(results) > 0 and results[0].boxes.xyxy is not None:
        for idx, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = box[:4].tolist()
            cls_id = int(results[0].boxes.cls[idx].item())
            conf = results[0].boxes.conf[idx].item()
            cls_name = results[0].names[cls_id] if 0 <= cls_id < len(results[0].names) else "Unknown"
            cls_name = class_overrides.get(cls_name, cls_name)

            if cls_name in selected_classes and conf >= confidence_threshold:
                box_color = class_colors.get(cls_name, (255, 0, 0))
                text_color = text_colors.get(cls_name, 'black')
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=7)

                if label_boxes:
                    label = f"{cls_name}: {conf:.2f}"
                    text_size = estimate_text_size(label, font_size)
                    draw.rectangle([x1, y1 - text_size[1] - 5, x1 + text_size[0], y1], fill=box_color)
                    draw.text((x1, y1 - text_size[1] - 5), label, fill=text_color, font=font)

                # Add detection data to the list
                detections.append(f"{cls_name} {conf:.2f} {x1} {y1} {x2} {y2}")
    
    # Save images
    img.save(image_output_dir / f"{image_prefix}{image_path.stem}{image_suffix}{image_path.suffix}")
    
    # Write detections to a text file
    write_detections_to_file(image_path, detections)

print(f"Processed {len(image_paths)} images. Output saved to '{image_output_dir}' and '{text_output_dir}'.")
