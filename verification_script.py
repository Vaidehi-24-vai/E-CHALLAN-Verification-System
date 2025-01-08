import os
import sys
import torch
from models.experimental import attempt_load
import cv2
import pytesseract
from PIL import Image
import numpy as np

# Root directory of YOLOv5 repository
YOLOV5_PATH = 'E:/YOLO-V5/yolov5-master/yolov5-master'

# Add YOLOv5 directory to system path
sys.path.append(YOLOV5_PATH)

# Replace 'best.pt' with the actual name of your trained weights file
weights_path = 'E:/YOLO-V5/yolov5-master/yolov5-master/runs/train/exp/weights/best.pt'

# Load YOLOv5 model
device = torch.device('cpu')
model = attempt_load(weights_path, device=device)  # Ensure weights are loaded from the specified path
model.eval()

# Root directory where YOLOv5 results are saved
results_root = os.path.join(YOLOV5_PATH, 'runs/detect')
exp_prefix = 'exp'

# Example function to process an uploaded image
def process_uploaded_image(image_path):
    # Run YOLOv5 detection on the image
    os.system(f'python {os.path.join(YOLOV5_PATH, "detect.py")} --source {image_path} --weights {weights_path}')

    # Find the latest experiment directory
    exp_dirs = [d for d in os.listdir(results_root) if d.startswith(exp_prefix)]
    if not exp_dirs:
        print("No experiment directory found.")
        return

    latest_exp_dir = max(exp_dirs, default="",
                         key=lambda d: int(d[len(exp_prefix):]) if d[len(exp_prefix):].isdigit() else 0)

    if not latest_exp_dir:
        print("No valid experiment directory found.")
        return

    results_path = os.path.join(results_root, latest_exp_dir)

    # Process the results directly from the YOLOv5 output
    output_image_path = os.path.join(results_path, os.path.basename(image_path))
    print(f"Processed Image Path: {output_image_path}")

    # Convert the image to a PyTorch tensor
    image = Image.open(output_image_path).convert('RGB')
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Resize the image to match the expected input size of YOLOv5
    image = torch.nn.functional.interpolate(image, size=(640, 640), mode='bilinear', align_corners=False)

    # Assuming YOLOv5 output provides bounding boxes in the format [x_center, y_center, width, height]
    detected_bounding_boxes = model(image)
    for bbox in detected_bounding_boxes:
     if isinstance(bbox, (list, torch.Tensor)):
        # Flatten the nested structure and convert to NumPy array
        bbox = np.array(bbox).squeeze()

        # Check if the array has an irregular shape
        if bbox.shape and any(isinstance(sub_array, np.ndarray) for sub_array in bbox):
            # Handle irregular shape, such as (3, 1, 3) + inhomogeneous part
            flattened_bbox = np.concatenate(bbox.flatten())
        else:
            # Regular flattening for other cases
            flattened_bbox = bbox.flatten()

        # Check if the flattened array has the expected length
        if len(flattened_bbox) == 4:
            x_center, y_center, width, height = flattened_bbox

            # Convert YOLOv5 format to (x, y, w, h) format
            x = int(x_center - width / 2)
            y = int(y_center - height / 2)
            w = int(width)
            h = int(height)

            roi = image[:, :, y:y + h, x:x + w]

            # Apply text detection or OCR within the ROI to locate the number plate
            number_plate_text = pytesseract.image_to_string(roi[0].numpy().astype(np.uint8), config='--psm 8')

            print(f"Number Plate Text: {number_plate_text}")
        else:
            print("Invalid bounding box format:", flattened_bbox)
    else:
        print("Invalid bounding box format:", bbox)
    
# Example usage
uploaded_image_path = 'E:/YOLO-V5/yolov5-master/yolov5-master/inference/images/bike-valid1.jpeg'
process_uploaded_image(uploaded_image_path)
