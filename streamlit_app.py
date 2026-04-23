from __future__ import annotations

import time
from typing import Dict

import cv2
import numpy as np
import streamlit as st

from common.camera import CameraStream, ProcessedCameraStream, discover_all_cameras, ensure_live_feed_server
from face_detection_system.detector import FaceDetectionProject
from hand_gesture_recognition.recognizer import HandGestureProject
from real_time_object_detection.detector import ObjectDetectionProject


st.set_page_config(page_title="CV Final Project Dashboard", layout="wide")


PROJECTS = {
    "Face Detection System": FaceDetectionProject,
    "Hand Gesture Recognition": HandGestureProject,
    "Real-Time Object Detection": ObjectDetectionProject,
}


def init_state() -> None:
    st.session_state.setdefault("sources", [])
    st.session_state.setdefault("source_map", {})
    st.session_state.setdefault("selected_source", None)
    st.session_state.setdefault("selected_project", "Face Detection System")
    st.session_state.setdefault("camera_stream", None)
    st.session_state.setdefault("processor_cache", {})
    st.session_state.setdefault("camera_started", False)
    st.session_state.setdefault("processed_stream", None)
    st.session_state.setdefault("active_project", None)
    st.session_state.setdefault("active_source", None)
    st.session_state.setdefault("live_feed_server", None)


def refresh_sources() -> None:
    sources = discover_all_cameras()
    st.session_state.sources = sources
    st.session_state.source_map = {source.name: source for source in sources}
    if sources and st.session_state.selected_source not in st.session_state.source_map:
        st.session_state.selected_source = sources[0].name


def get_processor(project_name: str):
    cache: Dict[str, object] = st.session_state.processor_cache
    if project_name not in cache:
        cache[project_name] = PROJECTS[project_name]()
    return cache[project_name]


def stop_stream() -> None:
    live_server = st.session_state.live_feed_server
    if live_server is not None:
        live_server.unregister_stream("main")

    processed_stream = st.session_state.processed_stream
    if processed_stream is not None:
        processed_stream.stop()
    st.session_state.processed_stream = None

    stream = st.session_state.camera_stream
    if stream is not None:
        stream.stop()
    st.session_state.camera_stream = None
    st.session_state.camera_started = False


def start_stream() -> None:
    stop_stream()
    source_name = st.session_state.selected_source
    source = st.session_state.source_map.get(source_name)
    if source is None:
        raise RuntimeError("No camera source selected.")

    processor = get_processor(st.session_state.selected_project)
    stream = CameraStream(source)
    stream.start()
    processed_stream = ProcessedCameraStream(stream, processor, max_width=640)
    processed_stream.start()
    live_server = st.session_state.live_feed_server or ensure_live_feed_server()
    live_server.register_stream("main", processed_stream)
    st.session_state.camera_stream = stream
    st.session_state.processed_stream = processed_stream
    st.session_state.live_feed_server = live_server
    st.session_state.camera_started = True
    st.session_state.active_project = st.session_state.selected_project
    st.session_state.active_source = st.session_state.selected_source


def render_metrics(info: dict) -> None:
    if "faces_detected" in info:
        st.metric("Faces Detected", info["faces_detected"])
    elif "gestures" in info:
        gestures = ", ".join(info["gestures"]) if info["gestures"] else "None"
        st.metric("Recognized Gestures", gestures)
    elif "objects_detected" in info:
        classes = ", ".join(info["classes"][:5]) if info["classes"] else "None"
        st.metric("Objects Detected", info["objects_detected"])
        st.caption(f"Classes: {classes}")


if hasattr(st, "fragment"):
    @st.fragment(run_every="300ms")
    def render_live_metrics():
        processed_stream = st.session_state.processed_stream
        if processed_stream is None:
            st.info("Waiting for camera frames...")
            return

        _, info = processed_stream.read()
        if not info:
            st.info("Waiting for camera frames...")
            return

        st.subheader("Live Results")
        render_metrics(info)
else:
    def render_live_metrics():
        processed_stream = st.session_state.processed_stream
        if processed_stream is None:
            st.info("Waiting for camera frames...")
            return

        _, info = processed_stream.read()
        if not info:
            st.info("Waiting for camera frames...")
            return

        st.subheader("Live Results")
        render_metrics(info)


init_state()
if st.session_state.live_feed_server is None:
    st.session_state.live_feed_server = ensure_live_feed_server()

st.title("Computer Vision Final Project Dashboard")
st.write(
    "Run all three projects from one place using either a laptop webcam or an Android phone "
    "running the IP Webcam app."
)

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh Cameras", use_container_width=True):
        refresh_sources()

    if not st.session_state.sources:
        refresh_sources()

    source_names = list(st.session_state.source_map.keys())
    if source_names:
        selected_index = source_names.index(st.session_state.selected_source)
        st.selectbox(
            "Camera Source",
            options=source_names,
            index=selected_index,
            key="selected_source",
        )
    else:
        st.warning("No cameras found yet. Connect a webcam or start IP Webcam, then refresh.")

    st.selectbox(
        "Project",
        options=list(PROJECTS.keys()),
        key="selected_project",
    )

    uploaded_image = st.file_uploader(
        "Optional Image Input",
        type=["jpg", "jpeg", "png"],
        help="Use this to test a still image instead of the live camera feed.",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", use_container_width=True, disabled=not source_names):
            start_stream()
    with col2:
        if st.button("Stop", use_container_width=True):
            stop_stream()

    st.caption("IP Webcam discovery scans the local network automatically on refresh.")


feed_col, info_col = st.columns([3, 1])
frame_placeholder = feed_col.empty()
info_placeholder = info_col.container()
processor = get_processor(st.session_state.selected_project)

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    uploaded_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    processed_frame, info = processor.process_frame(uploaded_frame)
    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(
        rgb_frame,
        channels="RGB",
        caption=f"{st.session_state.selected_project} image result",
        use_container_width=True,
    )
    with info_placeholder:
        st.subheader("Image Results")
        render_metrics(info)

elif st.session_state.camera_started and st.session_state.camera_stream is not None:
    if (
        st.session_state.active_project != st.session_state.selected_project
        or st.session_state.active_source != st.session_state.selected_source
    ):
        start_stream()

    processed_stream = st.session_state.processed_stream
    frame, _ = processed_stream.read() if processed_stream is not None else (None, {})

    if frame is None:
        frame_placeholder.info("Waiting for camera frames...")
    else:
        frame_placeholder.markdown(
            f"""
            <div style="background:#111;border-radius:12px;padding:8px;">
              <img
                src="http://127.0.0.1:8765/stream/main"
                style="width:100%;display:block;border-radius:8px;"
              />
            </div>
            """,
            unsafe_allow_html=True,
        )
        with info_placeholder:
            render_live_metrics()
else:
    frame_placeholder.info("Choose a camera, select a project, and press Start.")
    with info_placeholder:
        st.subheader("Projects")
        st.write("1. Face Detection System")
        st.write("2. Hand Gesture Recognition")
        st.write("3. Real-Time Object Detection")
