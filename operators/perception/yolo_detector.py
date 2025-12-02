"""
YOLO Detection Operator (Physical Operator)

Uses YOLOv8 model to detect objects in images, returns label, confidence, and bounding box for each object.
"""
from pathlib import Path
from typing import List
from PIL import Image
from ultralytics import YOLO

from agent.shared.state import ObjectBox
from common.config_loader import load_config

# Cache model (load once, avoid repeated loading)
_yolo_model = None


def get_yolo_model() -> YOLO:
    """Call YOLO model"""
    global _yolo_model
    if _yolo_model is None:  # If YOLO model in cache is empty, get model configuration
        config = load_config()
        project_root = Path(__file__).resolve().parents[2]  # operators/perception/yolo_detector.py -> project root directory
        model_path = (project_root / config['yolo']['model_path']).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")
        _yolo_model = YOLO(str(model_path))
    return _yolo_model  # Get model from model path


def detect_objects(image_path: str) -> List[ObjectBox]:
    """
    Use YOLOv8 model to detect objects in images, returns label, confidence, and bounding box for each object.
    
    :param image_path: Image file path
    :return: List of detected objects
    """
    model = get_yolo_model()
    # Load image
    image = Image.open(image_path)
    # Execute detection
    results = model(image)  # Image detection results
    boxes = results[0].boxes  # Bounding boxes
    cls_names = results[0].names  # Class names
    seen_labels = set()  # Already identified object class names
    detected_objects = []  # List of specific information for all detected objects
    for i in range(len(boxes)):  # Loop through detection boxes for each object in the image
        box = boxes[i]  # Extract detection box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()  # Extract confidence
        class_id = int(box.cls[0].item())  # Extract class ID
        label = cls_names[class_id]  # Extract class name
        if label in seen_labels:  # If class label already exists, skip
            continue
        seen_labels.add(label)  # If class label doesn't exist, add to seen_labels set
        detected_objects.append({
            "label": label,  # Label name
            "confidence": confidence,  # Confidence
            "bbox": [x1, y1, x2, y2]  # Bounding box coordinates
        })
    return detected_objects  # Return list of specific information for all detected objects

