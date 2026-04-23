from __future__ import annotations

import cv2
from ultralytics import YOLO

from common.camera import draw_label


class ObjectDetectionProject:
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = YOLO(model_name)

    def process_frame(self, frame):
        results = self.model(frame, verbose=False, imgsz=320, conf=0.35)
        annotated = results[0].plot()

        count = 0
        classes = []
        for box in results[0].boxes:
            count += 1
            cls_id = int(box.cls[0].item())
            classes.append(self.model.names[cls_id])

        summary = ", ".join(classes[:4]) if classes else "None"
        draw_label(annotated, f"Objects: {count}", 10, 30, (30, 70, 180))
        draw_label(annotated, f"Classes: {summary}", 10, 60, (30, 70, 180))
        return annotated, {"objects_detected": count, "classes": classes}
