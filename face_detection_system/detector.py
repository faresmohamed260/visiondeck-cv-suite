from __future__ import annotations

import cv2
import mediapipe as mp

from common.camera import draw_label


class FaceDetectionProject:
    def __init__(self, min_detection_confidence: float = 0.5):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self.landmark_indices = [33, 263, 1, 61, 291, 199]

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_result = self.face_detection.process(rgb)
        mesh_result = self.face_mesh.process(rgb)

        face_count = 0
        if detection_result.detections:
            height, width = frame.shape[:2]
            for detection in detection_result.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * width))
                y = max(0, int(bbox.ymin * height))
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 200, 80), 2)
                face_count += 1

        if mesh_result.multi_face_landmarks:
            height, width = frame.shape[:2]
            for face_landmarks in mesh_result.multi_face_landmarks:
                for index in self.landmark_indices:
                    landmark = face_landmarks.landmark[index]
                    px = int(landmark.x * width)
                    py = int(landmark.y * height)
                    cv2.circle(frame, (px, py), 3, (0, 120, 255), -1)

        draw_label(frame, f"Faces detected: {face_count}", 10, 30, (40, 140, 60))
        return frame, {"faces_detected": face_count}
