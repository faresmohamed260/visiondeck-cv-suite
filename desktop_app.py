from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict

import cv2
import numpy as np
from PIL import Image, ImageTk

from common.camera import (
    CameraStream,
    ProcessedCameraStream,
    cache_ip_webcam_source,
    create_manual_ip_webcam_source,
    discover_all_cameras,
)
from face_detection_system.detector import FaceDetectionProject
from hand_gesture_recognition.recognizer import HandGestureProject
from real_time_object_detection.detector import ObjectDetectionProject


PROJECTS = {
    "Face Detection System": FaceDetectionProject,
    "Hand Gesture Recognition": HandGestureProject,
    "Real-Time Object Detection": ObjectDetectionProject,
}


class DashboardApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Computer Vision Final Project Dashboard")
        self.root.geometry("1320x820")
        self.root.minsize(1100, 720)
        self.root.configure(bg="#111827")

        self.sources = []
        self.source_map = {}
        self.processor_cache: Dict[str, object] = {}
        self.camera_stream: CameraStream | None = None
        self.processed_stream: ProcessedCameraStream | None = None
        self.current_image = None
        self.refresh_in_progress = False

        self.discovery_mode = tk.StringVar(value="Quick Scan")
        self.project_name = tk.StringVar(value="Face Detection System")
        self.selected_source = tk.StringVar(value="")
        self.manual_ip = tk.StringVar(value="")
        self.manual_port = tk.StringVar(value="8080")
        self.status_text = tk.StringVar(value="Ready. Refresh cameras to begin.")
        self.metric_title = tk.StringVar(value="Result")
        self.metric_value = tk.StringVar(value="-")
        self.metric_caption = tk.StringVar(value="No active stream.")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh_sources()
        self._schedule_frame_update()

    def _build_ui(self) -> None:
        title = tk.Label(
            self.root,
            text="Computer Vision Final Project Dashboard",
            font=("Segoe UI", 28, "bold"),
            fg="white",
            bg="#111827",
        )
        title.pack(anchor="w", padx=24, pady=(20, 8))

        subtitle = tk.Label(
            self.root,
            text="Run all three projects from one place using a laptop webcam or an Android phone running the IP Webcam app.",
            font=("Segoe UI", 11),
            fg="#d1d5db",
            bg="#111827",
        )
        subtitle.pack(anchor="w", padx=24, pady=(0, 16))

        content = tk.Frame(self.root, bg="#111827")
        content.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        content.grid_columnconfigure(0, weight=0)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        sidebar = tk.Frame(content, bg="#1f2937", width=320)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        sidebar.grid_propagate(False)

        main = tk.Frame(content, bg="#111827")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self._build_sidebar(sidebar)
        self._build_main(main)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        header = tk.Label(parent, text="Controls", font=("Segoe UI", 18, "bold"), fg="white", bg="#1f2937")
        header.pack(anchor="w", padx=18, pady=(18, 14))

        self._section_label(parent, "Discovery Mode")
        mode_combo = ttk.Combobox(
            parent,
            textvariable=self.discovery_mode,
            values=["Quick Scan", "Full Scan"],
            state="readonly",
        )
        mode_combo.pack(fill="x", padx=18, pady=(0, 10))

        refresh_btn = ttk.Button(parent, text="Refresh Cameras", command=self.refresh_sources)
        refresh_btn.pack(fill="x", padx=18, pady=(0, 18))

        self._section_label(parent, "Manual IP Webcam Entry")
        ip_entry = ttk.Entry(parent, textvariable=self.manual_ip)
        ip_entry.pack(fill="x", padx=18, pady=(0, 8))
        ip_entry.insert(0, "")
        ip_entry.configure()

        port_entry = ttk.Entry(parent, textvariable=self.manual_port)
        port_entry.pack(fill="x", padx=18, pady=(0, 8))

        add_btn = ttk.Button(parent, text="Add Manual Camera", command=self.add_manual_camera)
        add_btn.pack(fill="x", padx=18, pady=(0, 18))

        self._section_label(parent, "Camera Source")
        self.source_combo = ttk.Combobox(parent, textvariable=self.selected_source, state="readonly")
        self.source_combo.pack(fill="x", padx=18, pady=(0, 18))

        self._section_label(parent, "Project")
        project_combo = ttk.Combobox(
            parent,
            textvariable=self.project_name,
            values=list(PROJECTS.keys()),
            state="readonly",
        )
        project_combo.pack(fill="x", padx=18, pady=(0, 18))

        image_btn = ttk.Button(parent, text="Test Still Image", command=self.load_still_image)
        image_btn.pack(fill="x", padx=18, pady=(0, 10))

        btn_row = tk.Frame(parent, bg="#1f2937")
        btn_row.pack(fill="x", padx=18, pady=(0, 18))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        start_btn = ttk.Button(btn_row, text="Start", command=self.start_stream)
        start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        stop_btn = ttk.Button(btn_row, text="Stop", command=self.stop_stream)
        stop_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        status_title = tk.Label(parent, text="Status", font=("Segoe UI", 12, "bold"), fg="white", bg="#1f2937")
        status_title.pack(anchor="w", padx=18, pady=(6, 6))
        status_box = tk.Label(
            parent,
            textvariable=self.status_text,
            justify="left",
            wraplength=270,
            font=("Segoe UI", 10),
            fg="#d1d5db",
            bg="#111827",
            padx=12,
            pady=10,
        )
        status_box.pack(fill="x", padx=18, pady=(0, 18))

    def _build_main(self, parent: tk.Frame) -> None:
        video_card = tk.Frame(parent, bg="#0b1220", bd=0, highlightthickness=1, highlightbackground="#374151")
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        video_card.grid_rowconfigure(0, weight=1)
        video_card.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(
            video_card,
            bg="#0b1220",
            fg="#9ca3af",
            text="Choose a camera, select a project, and press Start.",
            font=("Segoe UI", 14),
            compound="center",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)

        side_panel = tk.Frame(parent, bg="#111827")
        side_panel.grid(row=0, column=1, sticky="nsew")

        results_card = tk.Frame(side_panel, bg="#1f2937", bd=0, highlightthickness=1, highlightbackground="#374151")
        results_card.pack(fill="x", pady=(0, 16))

        results_title = tk.Label(
            results_card,
            text="Live Results",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#1f2937",
        )
        results_title.pack(anchor="w", padx=18, pady=(18, 10))

        metric_title = tk.Label(results_card, textvariable=self.metric_title, font=("Segoe UI", 12), fg="#d1d5db", bg="#1f2937")
        metric_title.pack(anchor="w", padx=18)

        metric_value = tk.Label(results_card, textvariable=self.metric_value, font=("Segoe UI", 28, "bold"), fg="white", bg="#1f2937")
        metric_value.pack(anchor="w", padx=18, pady=(4, 8))

        metric_caption = tk.Label(
            results_card,
            textvariable=self.metric_caption,
            wraplength=250,
            justify="left",
            font=("Segoe UI", 10),
            fg="#9ca3af",
            bg="#1f2937",
        )
        metric_caption.pack(anchor="w", padx=18, pady=(0, 18))

        notes_card = tk.Frame(side_panel, bg="#1f2937", bd=0, highlightthickness=1, highlightbackground="#374151")
        notes_card.pack(fill="both", expand=True)

        notes_title = tk.Label(notes_card, text="Projects", font=("Segoe UI", 16, "bold"), fg="white", bg="#1f2937")
        notes_title.pack(anchor="w", padx=18, pady=(18, 12))

        notes_text = (
            "1. Face Detection System\n"
            "2. Hand Gesture Recognition\n"
            "3. Real-Time Object Detection\n\n"
            "Use Test Still Image to run any module on a local image without starting a camera."
        )
        notes = tk.Label(
            notes_card,
            text=notes_text,
            justify="left",
            wraplength=250,
            font=("Segoe UI", 11),
            fg="#d1d5db",
            bg="#1f2937",
        )
        notes.pack(anchor="w", padx=18, pady=(0, 18))

    @staticmethod
    def _section_label(parent: tk.Frame, text: str) -> None:
        label = tk.Label(parent, text=text, font=("Segoe UI", 11, "bold"), fg="white", bg="#1f2937")
        label.pack(anchor="w", padx=18, pady=(0, 6))

    def set_status(self, message: str) -> None:
        self.status_text.set(message)

    def refresh_sources(self) -> None:
        if self.refresh_in_progress:
            return

        self.refresh_in_progress = True
        self.set_status("Scanning for cameras...")

        def worker() -> None:
            try:
                scan_mode = "quick" if self.discovery_mode.get() == "Quick Scan" else "full"
                sources = discover_all_cameras(scan_mode=scan_mode)
                self.root.after(0, lambda: self._finish_refresh(sources))
            except Exception as exc:
                self.root.after(0, lambda: self._refresh_failed(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_refresh(self, sources) -> None:
        self.refresh_in_progress = False
        self.sources = sources
        self.source_map = {source.name: source for source in sources}
        names = list(self.source_map.keys())
        self.source_combo["values"] = names
        if names:
            if self.selected_source.get() not in self.source_map:
                self.selected_source.set(names[0])
            self.set_status(f"Found {len(names)} camera source(s).")
        else:
            self.selected_source.set("")
            self.set_status("No cameras found. Connect a webcam or add an IP Webcam address manually.")

    def _refresh_failed(self, error: str) -> None:
        self.refresh_in_progress = False
        self.set_status(f"Camera scan failed: {error}")

    def add_manual_camera(self) -> None:
        raw_port = self.manual_port.get().strip()
        try:
            port = int(raw_port)
        except ValueError:
            messagebox.showerror("Invalid Port", "Please enter a valid numeric port.")
            return

        source = create_manual_ip_webcam_source(self.manual_ip.get(), port)
        if source is None:
            messagebox.showerror(
                "Camera Not Found",
                "Could not connect to that IP Webcam address. Check the IP, port, and that IP Webcam is running.",
            )
            return

        self.source_map[source.name] = source
        self.sources = list(self.source_map.values())
        names = list(self.source_map.keys())
        self.source_combo["values"] = names
        self.selected_source.set(source.name)
        cache_ip_webcam_source(source)
        self.set_status(f"Added manual camera source: {source.name}")

    def get_processor(self, project_name: str):
        if project_name not in self.processor_cache:
            self.processor_cache[project_name] = PROJECTS[project_name]()
        return self.processor_cache[project_name]

    def start_stream(self) -> None:
        source_name = self.selected_source.get().strip()
        source = self.source_map.get(source_name)
        if source is None:
            messagebox.showwarning("No Camera Selected", "Choose a camera source first.")
            return

        try:
            self.stop_stream()
            processor = self.get_processor(self.project_name.get())
            stream = CameraStream(source)
            stream.start()
            processed_stream = ProcessedCameraStream(stream, processor, max_width=960)
            processed_stream.start()
            self.camera_stream = stream
            self.processed_stream = processed_stream
            cache_ip_webcam_source(source)
            self.current_image = None
            self.set_status(f"Streaming from {source.name} with {self.project_name.get()}.")
        except Exception as exc:
            self.stop_stream()
            messagebox.showerror("Stream Error", str(exc))
            self.set_status(f"Failed to start stream: {exc}")

    def stop_stream(self) -> None:
        if self.processed_stream is not None:
            self.processed_stream.stop()
            self.processed_stream = None
        if self.camera_stream is not None:
            self.camera_stream.stop()
            self.camera_stream = None

    def load_still_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Image Error", "Could not read the selected image.")
            return

        processor = self.get_processor(self.project_name.get())
        processed_frame, info = processor.process_frame(frame)
        self._display_frame(processed_frame)
        self._update_metrics(info)
        self.set_status(f"Processed still image: {Path(path).name}")

    def _schedule_frame_update(self) -> None:
        self._update_live_frame()
        self.root.after(40, self._schedule_frame_update)

    def _update_live_frame(self) -> None:
        if self.processed_stream is None:
            return

        frame, info = self.processed_stream.read()
        if frame is None:
            return

        self._display_frame(frame)
        self._update_metrics(info)

    def _display_frame(self, frame) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        max_width = max(self.video_label.winfo_width() - 20, 320)
        max_height = max(self.video_label.winfo_height() - 20, 240)
        image.thumbnail((max_width, max_height))

        photo = ImageTk.PhotoImage(image=image)
        self.current_image = photo
        self.video_label.configure(image=photo, text="")

    def _update_metrics(self, info: dict) -> None:
        if "faces_detected" in info:
            self.metric_title.set("Faces Detected")
            self.metric_value.set(str(info["faces_detected"]))
            self.metric_caption.set("Live face count from the current frame.")
        elif "gestures" in info:
            gestures = info["gestures"]
            headline = gestures[0] if gestures else "None"
            details = ", ".join(gestures) if gestures else "No hand detected."
            self.metric_title.set("Recognized Gesture")
            self.metric_value.set(headline)
            self.metric_caption.set(details)
        elif "objects_detected" in info:
            classes = ", ".join(info["classes"][:5]) if info["classes"] else "None"
            self.metric_title.set("Objects Detected")
            self.metric_value.set(str(info["objects_detected"]))
            self.metric_caption.set(f"Classes: {classes}")
        else:
            self.metric_title.set("Result")
            self.metric_value.set("-")
            self.metric_caption.set("No result data available.")

    def _on_close(self) -> None:
        self.stop_stream()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    app = DashboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
