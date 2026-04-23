from __future__ import annotations

import cv2
import mediapipe as mp
from math import hypot

from common.camera import draw_label


class HandGestureProject:
    def __init__(self, min_detection_confidence: float = 0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self.drawer = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

    @staticmethod
    def _distance(a, b):
        return hypot(a.x - b.x, a.y - b.y)

    def _finger_states(self, landmarks, hand_label: str):
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]

        thumb_tip = landmarks[tips[0]]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        wrist = landmarks[0]

        thumb_reach = self._distance(thumb_tip, wrist)
        thumb_base = self._distance(thumb_mcp, wrist) + 1e-6
        thumb_extended = thumb_reach > thumb_base * 1.2

        if hand_label == "Right":
            thumb_open = thumb_extended and thumb_tip.x < thumb_ip.x
        else:
            thumb_open = thumb_extended and thumb_tip.x > thumb_ip.x

        fingers_open = [thumb_open]

        for tip, pip in zip(tips[1:], pips[1:]):
            fingers_open.append(landmarks[tip].y < landmarks[pip].y - 0.02)
        return fingers_open

    def _classify(self, hand_landmarks, hand_label: str):
        landmarks = hand_landmarks.landmark
        fingers = self._finger_states(landmarks, hand_label)
        thumb, index, middle, ring, pinky = fingers
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]

        other_folded = not index and not middle and not ring and not pinky
        thumb_high = thumb_tip.y < min(index_mcp.y, middle_mcp.y) - 0.05
        thumb_low = thumb_tip.y > max(index_mcp.y, middle_mcp.y) + 0.05
        thumb_vertical = abs(thumb_tip.y - thumb_ip.y) > abs(thumb_tip.x - thumb_ip.x)

        if thumb and other_folded and thumb_vertical and thumb_high:
            return "Thumbs Up"
        if thumb and other_folded and thumb_vertical and thumb_low:
            return "Thumbs Down"

        if all(fingers):
            return "Open Palm"
        if not any(fingers):
            return "Fist"
        if index and middle and not ring and not pinky:
            return "Peace"
        if thumb and not index and not middle and not ring and not pinky:
            return "Thumbs Up"
        if index and not middle and not ring and not pinky:
            return "Pointing"
        return "Custom/Unknown"

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gestures = []
        if results.multi_hand_landmarks:
            handedness_items = results.multi_handedness or []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks, start=1):
                hand_label = "Unknown"
                if idx - 1 < len(handedness_items):
                    hand_label = handedness_items[idx - 1].classification[0].label
                self.drawer.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self.styles.get_default_hand_landmarks_style(),
                    self.styles.get_default_hand_connections_style(),
                )
                gesture = self._classify(hand_landmarks, hand_label)
                gestures.append(gesture)

                height, width = frame.shape[:2]
                wrist = hand_landmarks.landmark[0]
                x = int(wrist.x * width)
                y = int(wrist.y * height)
                draw_label(frame, f"{hand_label} Hand: {gesture}", x, y, (180, 90, 20))

        headline = gestures[0] if gestures else "No hand detected"
        draw_label(frame, f"Gesture: {headline}", 10, 30, (120, 80, 20))
        return frame, {"gestures": gestures}
