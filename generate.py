# Imports
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import numpy as np

# Directories
input_dir = Path('./generate_input')
input_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path('./generate_output')
output_dir.mkdir(parents=True, exist_ok=True)

overlay_dir = output_dir / 'overlays'
overlay_dir.mkdir(parents=True, exist_ok=True)
overlay_prefix = "" # Image prefix, can be empty
overlay_suffix = "" # Image suffix, can be empty

detection_dir = output_dir / 'detections'
detection_dir.mkdir(parents=True, exist_ok=True)
detection_prefix = ""  # Text prefix, can be empty
detection_suffix = ""  # Text suffix, can be empty

mask_dir = output_dir / 'masks'
mask_dir.mkdir(parents=True, exist_ok=True)
mask_prefix = ""  # Text prefix, can be empty
mask_suffix = ""  # Text suffix, can be empty

# Load your trained model
model_path = './models/best.pt'
model = YOLO(model_path)

# Mode selection: detection or segmentation
mode = "detection"

# Detect all classes or selected classes only
detect_all_classes = True  # Set to True to detect all classes, False to detect only specific classes below

# Classes to detect
# Example: ['SpeechBalloons', 'General_speech', 'hit_sound', 'blast_sound', 'narration speech', 'thought_speech', 'roar']
selected_classes = ['socks']

# Class override mapping, treats the left side of the mapping as if it was the class of the right side
# Example: thought_speech annotations will be treated as SpeechBalloons annotations.
class_overrides = {
    'thought_speech': 'SpeechBalloons',
}

# Confidence threshold
confidence_threshold = 0.15

# Label settings
label_boxes = True  # Draw class names or just boxes
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

# Assign colors to each class, wrapping around if there are more classes than colors
class_colors = {class_name: predefined_colors_with_text[i % len(predefined_colors_with_text)][0] for i, class_name in enumerate(selected_classes)}
text_colors = {class_name: predefined_colors_with_text[i % len(predefined_colors_with_text)][1] for i, class_name in enumerate(selected_classes)}


# Store input images in a variable
image_paths = []
for extension in ['*.jpg', '*.jpeg', '*.png']:
    image_paths.extend(input_dir.glob(extension))

# Segmentation class
class YOLOSEG:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        height, width, _ = img.shape
        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]

        segmentation_contours_idx = []
        if len(result) > 0:
            for seg in result.masks.xy:
                segment = np.array(seg, dtype=np.float32)
                segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contours_idx, scores

ys = YOLOSEG(model_path)

# Function to estimate text size
def estimate_text_size(label, font_size):
    approx_char_width = font_size * 0.6
    text_width = len(label) * approx_char_width
    text_height = font_size
    return text_width, text_height

def write_detections_to_file(image_path, detections):
    # Create a text file named after the image
    text_file_path = detection_dir / f"{detection_prefix}{image_path.stem}{detection_suffix}.txt"

    with open(text_file_path, 'w') as file:
        for detection in detections:
            file.write(f"{detection}\n")

# Process images with progress bar
print(f"Generating outputs in {mode} mode.")
for image_path in tqdm(image_paths, desc='Processing Images'):
    # Detection Mode
    if mode == "detection":
        img_cv = cv2.imread(str(image_path))  # Load the image with OpenCV for mask generation
        mask_img = np.zeros(img_cv.shape[:2], dtype=np.uint8)  # Initialize a blank mask for all detections

        img_pil = Image.open(image_path)  # Load the image with PIL for overlay generation
        results = model.predict(img_pil)
        draw = ImageDraw.Draw(img_pil)
        detections = []

        if len(results) > 0 and results[0].boxes.xyxy is not None:
            for idx, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = box[:4].tolist()
                cls_id = int(results[0].boxes.cls[idx].item())
                conf = results[0].boxes.conf[idx].item()
                cls_name = results[0].names[cls_id] if 0 <= cls_id < len(results[0].names) else "Unknown"
                cls_name = class_overrides.get(cls_name, cls_name)

                if (cls_name in selected_classes or detect_all_classes) and conf >= confidence_threshold:
                    box_color = class_colors.get(cls_name, (255, 0, 0))
                    text_color = text_colors.get(cls_name, 'black')
                    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=7)

                    # Fill mask image for this detection
                    cv2.rectangle(mask_img, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=-1)  # -1 thickness fills the rectangle

                    if label_boxes:
                        label = f"{cls_name}: {conf:.2f}"
                        text_size = estimate_text_size(label, font_size)
                        draw.rectangle([x1, y1 - text_size[1] - 5, x1 + text_size[0], y1], fill=box_color)
                        draw.text((x1, y1 - text_size[1] - 5), label, fill=text_color, font=font)

                    # Add detection data to the list
                    detections.append(f"{cls_name} {conf:.2f} {x1} {y1} {x2} {y2}")

        # Save overlay images
        img_pil.save(overlay_dir / f"{overlay_prefix}{image_path.stem}{overlay_suffix}{image_path.suffix}")

        # Write detections to a text file
        write_detections_to_file(image_path, detections)

        # Save the combined mask image
        mask_output_path = mask_dir / f"{mask_prefix}{image_path.stem}{mask_suffix}.png"
        cv2.imwrite(str(mask_output_path), mask_img)
    elif mode == "segmentation":
        img_cv = cv2.imread(str(image_path))  # Load the image with OpenCV for segmentation and mask generation
        height, width, _ = img_cv.shape

        # Perform inference using YOLOSEG for segmentation masks
        bboxes, classes, segmentations, scores = ys.detect(img_cv)

        # Initialize a blank mask for all segmentations
        mask_img = np.zeros(img_cv.shape[:2], dtype=np.uint8)

        # Perform inference using the original YOLO model for initial annotation
        img_pil = Image.open(image_path)  # Load the image with PIL for overlay generation
        results = model.predict(img_pil)
        if hasattr(results[0], 'render'):
            annotated_img = results[0].render()[0]  # Use 'render' if available
        else:
            annotated_img = results[0].plot()  # Use 'plot' as a fallback
        annotated_img = np.array(annotated_img)  # Convert PIL image to NumPy array for CV2 processing

        # Text file for saving segmentation data
        txt_output_path = detection_dir / f"{detection_prefix}{image_path.stem}{detection_suffix}.txt"
        with open(txt_output_path, 'w') as f:
            for bbox, class_id, seg in zip(bboxes, classes, segmentations):
                # Normalize the segmentation data
                seg_normalized = seg / [width, height]
                # Write normalized data to text file
                seg_data = ' '.join([f'{x:.6f},{y:.6f}' for x, y in seg_normalized])
                f.write(f'{class_id} {seg_data}\n')

                # Draw segmentation mask on the combined mask image
                cv2.fillPoly(mask_img, [np.array(seg, dtype=np.int32)], 255)

                # Draw bounding box and segmentation mask on the annotated image
                x, y, x2, y2 = bbox
                cv2.rectangle(annotated_img, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.polylines(annotated_img, [np.array(seg, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

        # Save the final annotated image with bounding boxes and segmentation masks
        overlay_output_path = overlay_dir / f"{overlay_prefix}{image_path.stem}{overlay_suffix}{image_path.suffix}"
        cv2.imwrite(str(overlay_output_path), annotated_img)

        # Save the combined mask image
        mask_output_path = mask_dir / f"{mask_prefix}{image_path.stem}{mask_suffix}.png"
        cv2.imwrite(str(mask_output_path), mask_img)

print(f"Processed {len(image_paths)} images. Overlays saved to '{overlay_dir}', Detections saved to '{detection_dir}', and Masks saved to '{mask_dir}'.")
