from __future__ import annotations

import time
from typing import Dict

import cv2
import numpy as np
import streamlit as st

from common.camera import (
    CameraStream,
    ProcessedCameraStream,
    cache_ip_webcam_source,
    create_manual_ip_webcam_source,
    discover_all_cameras,
    ensure_live_feed_server,
)
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
    st.session_state.setdefault("sources_loaded", False)
    st.session_state.setdefault("scan_mode", "Quick Scan")
    st.session_state.setdefault("stream_version", 0)
    st.session_state.setdefault("manual_camera_ip", "")
    st.session_state.setdefault("manual_camera_port", 8080)


def refresh_sources() -> None:
    scan_mode = "quick" if st.session_state.scan_mode == "Quick Scan" else "full"
    sources = discover_all_cameras(scan_mode=scan_mode)
    st.session_state.sources = sources
    st.session_state.source_map = {source.name: source for source in sources}
    st.session_state.sources_loaded = True
    if sources and st.session_state.selected_source not in st.session_state.source_map:
        st.session_state.selected_source = sources[0].name


def add_manual_camera_source(ip_or_url: str, port: int) -> tuple[bool, str]:
    source = create_manual_ip_webcam_source(ip_or_url, port)
    if source is None:
        return False, "Could not connect to that IP Webcam address. Check the IP, port, and that IP Webcam is running."

    st.session_state.source_map[source.name] = source
    st.session_state.sources = list(st.session_state.source_map.values())
    st.session_state.selected_source = source.name
    st.session_state.sources_loaded = True
    cache_ip_webcam_source(source)
    return True, f"Added camera source: {source.name}"


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
    cache_ip_webcam_source(source)

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
    st.session_state.stream_version += 1


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
    @st.fragment(run_every="120ms")
    def render_live_feed():
        processed_stream = st.session_state.processed_stream
        if processed_stream is None:
            st.info("Waiting for camera frames...")
            return

        jpeg = processed_stream.read_jpeg()
        if jpeg is None:
            st.info("Waiting for camera frames...")
            return

        st.image(
            jpeg,
            caption=f"{st.session_state.selected_project} live feed",
            use_container_width=True,
        )

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
    def render_live_feed():
        processed_stream = st.session_state.processed_stream
        if processed_stream is None:
            st.info("Waiting for camera frames...")
            return

        jpeg = processed_stream.read_jpeg()
        if jpeg is None:
            st.info("Waiting for camera frames...")
            return

        st.image(
            jpeg,
            caption=f"{st.session_state.selected_project} live feed",
            use_container_width=True,
        )

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
    st.radio(
        "Discovery Mode",
        options=["Quick Scan", "Full Scan"],
        key="scan_mode",
        help=(
            "Quick Scan tries the last successful phone IP first and checks the most likely addresses. "
            "Full Scan checks the whole local subnet."
        ),
    )
    if st.button("Refresh Cameras", use_container_width=True):
        refresh_sources()

    st.markdown("**Manual IP Webcam Entry**")
    st.text_input(
        "Camera IP or URL",
        key="manual_camera_ip",
        placeholder="Example: 192.168.1.15 or http://192.168.1.15:8080",
        help="Enter the Android IP Webcam address manually if discovery does not find it.",
    )
    st.number_input(
        "Port",
        min_value=1,
        max_value=65535,
        key="manual_camera_port",
        step=1,
        help="Default IP Webcam port is usually 8080.",
    )
    if st.button("Add Manual Camera", use_container_width=True):
        ok, message = add_manual_camera_source(
            st.session_state.manual_camera_ip,
            int(st.session_state.manual_camera_port),
        )
        if ok:
            st.success(message)
        else:
            st.error(message)

    source_names = list(st.session_state.source_map.keys())
    if source_names and st.session_state.selected_source in source_names:
        selected_index = source_names.index(st.session_state.selected_source)
        st.selectbox(
            "Camera Source",
            options=source_names,
            index=selected_index,
            key="selected_source",
        )
    else:
        st.selectbox(
            "Camera Source",
            options=["No camera loaded yet"],
            index=0,
            disabled=True,
        )
        if st.session_state.sources_loaded:
            st.warning("No cameras found. Connect a webcam or start IP Webcam, then refresh.")
        else:
            st.caption("Click `Refresh Cameras` to scan for laptop and IP cameras.")

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

    if st.session_state.scan_mode == "Quick Scan":
        st.caption("Quick Scan tries the last working phone IP first, then scans likely local addresses.")
    else:
        st.caption("Full Scan checks the whole local subnet and may take longer.")


feed_col, info_col = st.columns([3, 1])
frame_placeholder = feed_col.empty()
info_placeholder = info_col.container()

if uploaded_image is not None:
    processor = get_processor(st.session_state.selected_project)
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

    with feed_col:
        render_live_feed()
    with info_col:
        render_live_metrics()
else:
    frame_placeholder.info("Choose a camera, select a project, and press Start.")
    with info_placeholder:
        st.subheader("Projects")
        st.write("1. Face Detection System")
        st.write("2. Hand Gesture Recognition")
        st.write("3. Real-Time Object Detection")
