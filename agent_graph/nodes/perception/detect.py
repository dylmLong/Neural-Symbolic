from pathlib import Path
from typing import List
from PIL import Image
from langchain_core.tools import tool
from ultralytics import YOLO

from agent_graph.state_schema import AgentState, ObjectBox
from common.config_loader import load_config

# Caching model (loaded once)
_yolo_model = None

def get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        config = load_config()
        project_root = Path(__file__).resolve().parents[3]
        model_path = (project_root / config['yolo']['model_path']).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found:{model_path}")
        _yolo_model = YOLO(str(model_path))
    return _yolo_model


@tool
def detect_objects(image_path: str) -> List[ObjectBox]:
    """
    Use the YOLOv8 model to detect objects in an image and return the label, confidence score, and bounding box for each object.
    """
    model = get_yolo_model()
    # Load image
    image = Image.open(image_path)
    # Perform detection
    results = model(image)
    boxes = results[0].boxes
    cls_names = results[0].names
    seen_labels = set()
    detected_objects = []
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        label = cls_names[class_id]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        detected_objects.append({
            "label": label,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })
    return detected_objects

# LangGraph Node Functions
def detect_node(state: AgentState) -> AgentState:
    if not state.get("image_path"):
        raise ValueError("image_path not provided, unable to perform detection")
    print("\n=============================[Perception Message]===============================\n")
    print(f"Identifying image:{state.get('image_path')}")
    objects = detect_objects.invoke(state["image_path"])
    print("The following objects were detected:")
    for i, obj in enumerate(objects, 1):
        print(f"[{i}] Category: {obj.get('label')} | Confidence: {obj.get('confidence'):.2f} | Location: {obj.get('bbox')}")
    new_state = state.copy()
    new_state["objects"] = objects
    return new_state

if __name__ == '__main__':
    path = "C:\\Users\\lenovo\\Downloads\\Agent-master\\data\\test1.jpg"
    res = detect_objects.invoke(path)
    print("List of detected objects:")
    for i, obj in enumerate(res, 1):
        print(f"[{i}] Category: {obj.get('label')} | Confidence: {obj.get('confidence'):.2f} | Location: {obj.get('bbox')}")