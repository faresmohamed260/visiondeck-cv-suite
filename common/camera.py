from __future__ import annotations

import ipaddress
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


REQUEST_TIMEOUT = 0.35
DEFAULT_IP_WEBCAM_PORT = 8080
CACHE_FILE = Path(__file__).resolve().parent / "camera_cache.json"


@dataclass
class CameraSource:
    source_id: str
    name: str
    kind: str
    value: int | str
    base_url: Optional[str] = None
    preview_url: Optional[str] = None


def _probe_local_camera(index: int) -> Optional[CameraSource]:
    capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        return None

    ok, _ = capture.read()
    capture.release()
    if not ok:
        return None

    return CameraSource(
        source_id=f"local-{index}",
        name=f"Laptop Camera {index}",
        kind="local",
        value=index,
    )


def discover_local_cameras(max_index: int = 4) -> List[CameraSource]:
    cameras: List[CameraSource] = []
    for index in range(max_index + 1):
        source = _probe_local_camera(index)
        if source:
            cameras.append(source)
    return cameras


def _local_ipv4_networks() -> List[ipaddress.IPv4Network]:
    networks: list[ipaddress.IPv4Network] = []
    seen: set[str] = set()

    hostname = socket.gethostname()
    addresses = set()

    try:
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            addresses.add(info[4][0])
    except socket.gaierror:
        pass

    try:
        addresses.add(socket.gethostbyname(hostname))
    except socket.gaierror:
        pass

    for address in addresses:
        if address.startswith("127."):
            continue
        parts = address.split(".")
        if len(parts) != 4:
            continue
        network_str = ".".join(parts[:3]) + ".0/24"
        if network_str in seen:
            continue
        seen.add(network_str)
        networks.append(ipaddress.ip_network(network_str, strict=False))

    return networks


def _local_ipv4_addresses() -> List[str]:
    addresses: list[str] = []
    seen: set[str] = set()
    hostname = socket.gethostname()

    try:
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = info[4][0]
            if address.startswith("127.") or address in seen:
                continue
            seen.add(address)
            addresses.append(address)
    except socket.gaierror:
        pass

    try:
        address = socket.gethostbyname(hostname)
        if not address.startswith("127.") and address not in seen:
            addresses.append(address)
    except socket.gaierror:
        pass

    return addresses


def _read_camera_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_camera_cache(payload: dict) -> None:
    CACHE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _get_cached_ip_webcam() -> Optional[str]:
    payload = _read_camera_cache()
    cached_ip = payload.get("last_ip_webcam")
    if isinstance(cached_ip, str) and cached_ip:
        return cached_ip
    return None


def cache_ip_webcam_source(source: CameraSource) -> None:
    if source.kind != "ip_webcam" or not source.base_url:
        return

    try:
        cached_ip = urlparse(source.base_url).hostname
    except ValueError:
        return

    if not cached_ip:
        return

    payload = _read_camera_cache()
    payload["last_ip_webcam"] = cached_ip
    _write_camera_cache(payload)


def _is_ip_webcam_host(ip: str, port: int = DEFAULT_IP_WEBCAM_PORT) -> Optional[CameraSource]:
    base_url = f"http://{ip}:{port}"
    checks = [
        f"{base_url}/status.json",
        f"{base_url}/shot.jpg",
        base_url,
    ]

    for url in checks:
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            continue

        if response.status_code != 200:
            continue

        content_type = response.headers.get("content-type", "").lower()
        body = response.text[:300].lower() if "text" in content_type else ""

        if "ip webcam" in body or "json" in content_type or "image/jpeg" in content_type:
            return CameraSource(
                source_id=f"ip-{ip}",
                name=f"IP Webcam ({ip})",
                kind="ip_webcam",
                value=f"{base_url}/shot.jpg",
                base_url=base_url,
                preview_url=f"{base_url}/shot.jpg",
            )

    return None


def _build_ip_candidates(networks: List[ipaddress.IPv4Network], mode: str) -> List[str]:
    cached_ip = _get_cached_ip_webcam()
    candidates: list[str] = []
    seen: set[str] = set()
    local_addresses = _local_ipv4_addresses()

    def add_candidate(ip: str) -> None:
        if ip in seen:
            return
        seen.add(ip)
        candidates.append(ip)

    if cached_ip:
        add_candidate(cached_ip)

    for address in local_addresses:
        octets = address.split(".")
        if len(octets) != 4:
            continue
        prefix = ".".join(octets[:3])
        host_number = int(octets[3])

        if mode == "quick":
            likely_hosts = [host_number + offset for offset in range(-8, 9)]
            likely_hosts += list(range(100, 111))
            likely_hosts += list(range(120, 141))
            likely_hosts += list(range(150, 171))
            likely_hosts += list(range(180, 201))
            likely_hosts += list(range(2, 20))
            for host in likely_hosts:
                if 1 <= host <= 254:
                    add_candidate(f"{prefix}.{host}")

    for network in networks:
        hosts = [str(host) for host in network.hosts()]
        if mode == "quick":
            tail_hosts = hosts[-40:]
            head_hosts = hosts[:20]
            middle_hosts = hosts[99:140]
            for ip in tail_hosts + middle_hosts + head_hosts:
                add_candidate(ip)
        else:
            for ip in hosts[:254]:
                add_candidate(ip)

    return candidates


def discover_ip_webcams(mode: str = "quick") -> List[CameraSource]:
    networks = _local_ipv4_networks()
    if not networks:
        return []

    candidates = _build_ip_candidates(networks, mode=mode)

    results: list[CameraSource] = []
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(_is_ip_webcam_host, ip): ip for ip in candidates}
        for future in as_completed(futures):
            source = future.result()
            if source:
                cache_ip_webcam_source(source)
                results.append(source)

    results.sort(key=lambda item: item.name)
    return results


def discover_all_cameras(scan_mode: str = "quick") -> List[CameraSource]:
    sources = discover_local_cameras()
    known_ids = {source.source_id for source in sources}
    for source in discover_ip_webcams(mode=scan_mode):
        if source.source_id not in known_ids:
            sources.append(source)
    return sources


class CameraStream:
    def __init__(self, source: CameraSource):
        self.source = source
        self.capture: Optional[cv2.VideoCapture] = None
        self.latest_frame = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.session = requests.Session()

    def start(self) -> None:
        if self.running:
            return

        if self.source.kind == "local":
            self.capture = cv2.VideoCapture(self.source.value, cv2.CAP_MSMF)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(self.source.value, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                raise RuntimeError(f"Unable to open camera source: {self.source.name}")

            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)

            try:
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        self.running = True
        target = self._ip_webcam_loop if self.source.kind == "ip_webcam" else self._reader_loop
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def _reader_loop(self) -> None:
        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.05)

    def _ip_webcam_loop(self) -> None:
        snapshot_url = self.source.value
        while self.running:
            try:
                response = self.session.get(snapshot_url, timeout=REQUEST_TIMEOUT)
                if response.status_code != 200:
                    time.sleep(0.03)
                    continue

                data = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
            except requests.RequestException:
                time.sleep(0.05)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.session.close()


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x, max(0, y - 24)), (x + 220, y), color, -1)
    cv2.putText(frame, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


class ProcessedCameraStream:
    def __init__(self, camera_stream: CameraStream, processor, max_width: int = 640):
        self.camera_stream = camera_stream
        self.processor = processor
        self.max_width = max_width
        self.latest_frame = None
        self.latest_info = {}
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None

    def start(self) -> None:
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _resize_for_processing(self, frame):
        height, width = frame.shape[:2]
        if width <= self.max_width:
            return frame

        scale = self.max_width / float(width)
        return cv2.resize(frame, (int(width * scale), int(height * scale)))

    def _worker_loop(self) -> None:
        while self.running:
            frame = self.camera_stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = self._resize_for_processing(frame)
            processed_frame, info = self.processor.process_frame(frame)
            ok, encoded = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            with self.lock:
                self.latest_frame = processed_frame
                self.latest_info = info
                if ok:
                    self.latest_jpeg = encoded.tobytes()

    def read(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            info = dict(self.latest_info)
        return frame, info

    def read_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class _StreamHandler(BaseHTTPRequestHandler):
    server: "LiveFeedServer"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            payload = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        prefix = "/stream/"
        if not parsed.path.startswith(prefix):
            self.send_error(404)
            return

        stream_id = parsed.path[len(prefix):]
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            processed_stream = self.server.get_stream(stream_id)
            if processed_stream is None:
                time.sleep(0.1)
                continue

            jpeg = processed_stream.read_jpeg()
            if jpeg is None:
                time.sleep(0.01)
                continue

            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, format, *args):
        return


class LiveFeedServer(ThreadingHTTPServer):
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        super().__init__((host, port), _StreamHandler)
        self.streams: dict[str, ProcessedCameraStream] = {}
        self.lock = threading.Lock()

    def register_stream(self, stream_id: str, stream: ProcessedCameraStream) -> None:
        with self.lock:
            self.streams[stream_id] = stream

    def unregister_stream(self, stream_id: str) -> None:
        with self.lock:
            self.streams.pop(stream_id, None)

    def get_stream(self, stream_id: str) -> Optional[ProcessedCameraStream]:
        with self.lock:
            return self.streams.get(stream_id)


_live_feed_server: Optional[LiveFeedServer] = None
_live_feed_thread: Optional[threading.Thread] = None
_live_feed_lock = threading.Lock()


def ensure_live_feed_server(host: str = "127.0.0.1", port: int = 8765) -> LiveFeedServer:
    global _live_feed_server, _live_feed_thread

    with _live_feed_lock:
        if _live_feed_server is not None:
            return _live_feed_server

        _live_feed_server = LiveFeedServer(host=host, port=port)
        _live_feed_thread = threading.Thread(target=_live_feed_server.serve_forever, daemon=True)
        _live_feed_thread.start()
        return _live_feed_server
