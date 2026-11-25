"""WasteLens image workbench, adapted from the supplied Stitch design.

Run from the repository root: streamlit run app/app.py
"""

import hashlib
import base64
import csv
import io
import json
import random
import os
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from html import escape
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from wastelens import ops
from wastelens.classifier import Prediction, WasteClassifier
from wastelens.profiles import DEFAULT_MODEL, MODEL_PROFILES
from wastelens.workbench import PRESETS, PRESET_DESCRIPTIONS, assess_prediction, describe_operations, request_signature
from wastelens.streaming import DETECTOR_WEIGHTS, ObjectDetector, ObjectTracker, decode_frame, recognize_objects
from wastelens.video import (VideoSettings, make_sample_video, objects_csv, probe, process_video, read_frame,
                             records_csv, segment_agreement)

ROOT = Path(__file__).resolve().parents[1]
MAX_SIDE = 1024
LABELS_VI = {
    "cardboard": "Bìa carton", "glass": "Thủy tinh", "metal": "Kim loại", "paper": "Giấy",
    "plastic": "Nhựa", "trash": "Rác khác", "uncertain": "Không chắc",
}
MODEL_NAMES = {"baseline": "Baseline Lab 2", "improved": "MLP cải tiến", "mobilenet": "MobileNetV3 Large"}
CLASS_COLORS = {
    "cardboard": "#d97706", "glass": "#0284c7", "metal": "#64748b",
    "paper": "#6366f1", "plastic": "#059669", "trash": "#e11d48",
}
UPLOAD_ERROR = "Không đọc được ảnh. Vui lòng chọn lại tệp JPG hoặc PNG hợp lệ."
ASSESSMENT_LABELS = {"unlabeled": "Chưa có nhãn thật", "correct": "Đúng", "incorrect": "Sai", "uncertain": "Không chắc"}
STATELESS_HOST = os.environ.get("WASTELENS_STATELESS_HOST") == "1"
if STATELESS_HOST:
    from streamlit.components.v2 import component
    bridge_assets = Path(__file__).with_name("file_bridge")
    file_bridge = component("wastelens_files", html=(bridge_assets / "component.html").read_text(encoding="utf-8"),
                            css=(bridge_assets / "component.css").read_text(encoding="utf-8"),
                            js=(bridge_assets / "component.js").read_text(encoding="utf-8"))


def bridge_upload(key: str):
    """Send upload bytes over the session WebSocket, not an instance-local HTTP endpoint."""
    result = file_bridge(data={"mode": "upload", "max_bytes": 4 * 1024 * 1024}, key=key,
                         default={"file": None}, on_file_change=lambda: None)
    payload = result.file
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), str):
        return
    try:
        if len(payload["data"]) > 6 * 1024 * 1024:
            raise ValueError("Image too large")
        data = base64.b64decode(payload["data"], validate=True)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("Image too large")
    except ValueError:
        st.error("Ảnh không hợp lệ hoặc lớn hơn 4 MB. Vui lòng chọn lại.")
        return
    name = str(payload.get("name", "image.png"))
    marker = (hashlib.sha256(data).hexdigest(), name, key)
    if marker != st.session_state.last_upload:
        st.session_state.update(input_bytes=data, input_name=name, sample_label=None, sample_path=None,
                                last_upload=marker)
        st.rerun()


def show_image(image: np.ndarray, caption: str):
    """Display a BGR image; on the stateless host, embed it as a data URI instead of a media file."""
    if not STATELESS_HOST:
        st.image(image, channels="BGR", width="stretch", alt=caption)
        return
    ok, encoded = cv2.imencode(".png", image)
    if ok:
        data = base64.b64encode(encoded).decode("ascii")
        st.html(f'<div class="inline-image"><img src="data:image/png;base64,{data}" alt="{escape(caption)}"/>'
                f'<span>{escape(caption)}</span></div>')


@st.cache_resource(show_spinner="Đang nạp mô hình…")
def load_model(checkpoint: str, modified_ns: int) -> WasteClassifier:
    """Replacing a checkpoint invalidates its cached classifier."""
    return WasteClassifier(Path(checkpoint))


def profile_model(key: str) -> WasteClassifier:
    """Cached classifier for a model ID, keyed by the checkpoint's modification time."""
    path = MODEL_PROFILES[key].checkpoint
    return load_model(str(path), path.stat().st_mtime_ns)


@st.cache_resource(show_spinner="Đang nạp bộ phát hiện vật…")
def load_detector(checkpoint: str, modified_ns: int):
    """Cached COCO object detector for the webcam tab."""
    return ObjectDetector(Path(checkpoint))


@st.cache_resource
def camera_component(asset_revision: tuple[int, ...]):
    """Custom camera component (HTML/CSS/JS), rebuilt when its asset files change."""
    from streamlit.components.v2 import component
    assets = Path(__file__).with_name("camera")
    return component("wastelens_camera", **{kind: (assets / f"component.{kind}").read_text(encoding="utf-8")
                                           for kind in ("html", "css", "js")})


def render_webcam():
    """Webcam tab: the browser streams JPEG frames; each is detected, cropped and classified."""
    st.caption("SSDLite phát hiện vật → cắt vùng ảnh → mô hình bạn chọn phân loại thành 6 loại rác.")
    model_col, processing_col, settings_col = st.columns([4, 3, 2])
    key = model_col.selectbox("Mô hình phân loại", list(MODEL_NAMES), format_func=MODEL_NAMES.get,
                              index=list(MODEL_NAMES).index(DEFAULT_MODEL), key="camera_model")
    preset = processing_col.selectbox("Xử lý vùng ảnh", list(PRESETS), key="camera_preset")
    with settings_col.popover("Thiết lập stream", width="stretch"):
        fps = st.slider("Khung hình phân tích / giây", 1, 5, 2, key="camera_fps")
        analysis_size = st.selectbox("Độ phân giải phân tích (cạnh dài)", [640, 960, 1280], index=2,
                                     format_func=lambda size: f"{size} px", key="camera_analysis_size")
        threshold = st.slider("Ngưỡng phát hiện vật", .2, .9, .45, .05, key="camera_threshold")
        max_objects = st.slider("Số vật tối đa mỗi khung", 1, 5, 3, key="camera_max_objects")
        st.caption("Khung xem lấy độ phân giải cao theo camera thiết bị và giữ nguyên tỉ lệ. "
                   "Giảm độ phân giải phân tích nếu AI xử lý chậm; chất lượng khung xem vẫn giữ nguyên.")
    if not DETECTOR_WEIGHTS.is_file():
        st.info("Cần tải bộ phát hiện vật trước khi bật webcam.")
        st.code(".venv\\Scripts\\python.exe scripts/prepare_detector.py", language="powershell")
        return
    if not MODEL_PROFILES[key].checkpoint.is_file():
        st.error(f"Thiếu checkpoint: {MODEL_PROFILES[key].filename}")
        return
    # Webcam state survives Streamlit reruns: the last response sent to the browser, the recognized
    # objects, the browser session ID (changes on every camera start) and the selected object.
    state = st.session_state.setdefault("camera_state", {"response": None, "objects": [], "marker": None,
                                                         "session": None, "active": False, "selected": None})
    settings = (key, preset, threshold, max_objects)
    # New model or settings: drop results that were computed with the old ones.
    if state.get("settings") != settings:
        state.update(settings=settings, objects=[], selected=None)
        if state["response"]:
            state["response"] = {**state["response"], "objects": []}
    assets = Path(__file__).with_name("camera")
    revision = tuple((assets / f"component.{kind}").stat().st_mtime_ns for kind in ("html", "css", "js"))
    result = camera_component(revision)(data={"fps": fps, "analysisSize": analysis_size,
                                            "response": state["response"], "selected": state["selected"]},
                                key="live_camera", default={"frame": None, "event": None, "selection": None},
                                on_frame_change=lambda: None, on_event_change=lambda: None,
                                on_selection_change=lambda: None)
    event = result.event
    # Start/stop events from the browser reset the tracker so IDs never leak between camera sessions.
    if isinstance(event, dict) and event != state.get("last_event"):
        state["last_event"] = event
        state.update(active=event.get("active") is True, session=event.get("session"), objects=[], selected=None,
                     response=None, tracker=ObjectTracker())
    selection = result.selection
    if isinstance(selection, dict) and selection.get("session") == state["session"] and selection != state.get("selection_event"):
        state["selected"] = selection.get("id")
        state["selection_event"] = selection
    payload = result.frame
    # Only frames from the current camera session are processed, and each sequence number only once.
    if isinstance(payload, dict) and state["active"] and payload.get("session") == state["session"]:
        marker = (payload.get("session"), payload.get("seq"))
        if marker != state["marker"]:
            state["marker"] = marker
            response = {"session": marker[0], "seq": marker[1], "objects": [], "width": 1, "height": 1}
            try:
                frame = decode_frame(payload)
                detector = load_detector(str(DETECTOR_WEIGHTS), DETECTOR_WEIGHTS.stat().st_mtime_ns)
                objects = recognize_objects(frame, detector, profile_model(key), state["tracker"], PRESETS[preset],
                                            threshold=threshold, max_objects=max_objects)
                state["objects"] = objects
                response.update(width=frame.shape[1], height=frame.shape[0], objects=[
                    {"id": obj["id"], "box": obj["box"], "label": LABELS_VI[obj["prediction"].label],
                     "confidence": obj["prediction"].confidence} for obj in objects])
                if state["selected"] not in [obj["id"] for obj in objects]:
                    state["selected"] = objects[0]["id"] if objects else None
            except (ValueError, OSError, RuntimeError) as error:
                state["objects"] = []
                response["error"] = "Không xử lý được khung hình. Dừng rồi bật lại camera."
                state["error"] = str(error)
            state["response"] = response
            st.rerun()  # Rerun so the component receives the boxes and can send the next frame.
    objects = state["objects"] if state["active"] else []
    if objects:
        ids = [obj["id"] for obj in objects]
        selected = state["selected"] if state["selected"] in ids else ids[0]
        st.caption("Vật đang xem")
        for column, object_id in zip(st.columns(len(ids)), ids):
            if column.button(f"Vật #{object_id}", key=f"camera_object_{object_id}",
                             type="primary" if object_id == selected else "secondary", width="stretch"):
                state["selected"] = object_id
                st.rerun()
        obj = next(obj for obj in objects if obj["id"] == selected)
        prediction = obj["prediction"]
        st.markdown(f"**Vật #{selected} · {LABELS_VI[prediction.label]} · {prediction.confidence:.1%}**")
        st.caption(f"Phát hiện: {obj['detector_category']} ({obj['detector_confidence']:.0%}) · "
                   f"Phân loại: {MODEL_NAMES[key]} · Xử lý: {preset}")
        probability_table(prediction)
    else:
        st.info("Chưa có vật để phân loại." if state["active"] else "Bật camera để bắt đầu. Bạn có thể dừng bất cứ lúc nào.")
        probability_table(None)
    st.caption("Webcam không có Ground Truth. SSDLite nhận diện vật thuộc COCO, có thể bỏ sót rác không nằm trong các loại vật đã học. "
               "Nhãn rác được suy luận riêng trên từng vùng ảnh; đưa một vật rõ nét trước camera để thử.")


@st.cache_data
def sample_catalog() -> dict[str, list[Path]]:
    """Use labels from the versioned manifest; only held-out test images are offered."""
    manifest, image_root = ROOT / "data/splits/trashnet.csv", ROOT / "data/trashnet"
    if not manifest.is_file() or not image_root.is_dir():
        return {}
    samples = {}
    with manifest.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            path, label = image_root / row["path"], row["label"]
            if row["split"] == "test" and label in CLASS_COLORS and path.is_file():
                samples.setdefault(label, []).append(path)
    return samples


def to_bgr(image: Image.Image) -> np.ndarray:
    """Apply phone EXIF orientation and bound preview/filter cost without cropping."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if max(rgb.size) > MAX_SIDE:
        ratio = MAX_SIDE / max(rgb.size)
        rgb = rgb.resize((max(1, round(rgb.width * ratio)), max(1, round(rgb.height * ratio))),
                         Image.Resampling.BILINEAR)
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


@st.cache_data(show_spinner=False)
def decode_image(data: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    # Pixel decoding must be guarded too: truncated images may open successfully.
    """Decode uploaded bytes to a bounded BGR image and its original (EXIF-corrected) size."""
    with Image.open(io.BytesIO(data)) as image:
        original_size = ImageOps.exif_transpose(image).size
        return to_bgr(image), original_size


def init_state():
    """Create every session-state key the page reads, once per browser session."""
    defaults = {"input_bytes": None, "input_name": "", "sample_label": None, "sample_path": None,
                "upload_revision": 0,
                "last_upload": None, "ops_params": asdict(ops.OpsParams()), "processing_level": "Gốc",
                "result": None, "image_view": "Ảnh gốc"}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_image():
    """Remove the current image and its result; a new uploader key resets the widget."""
    st.session_state.update(input_bytes=None, input_name="", sample_label=None, sample_path=None,
                            upload_revision=st.session_state.upload_revision + 1, result=None)


def choose_sample(label: str, path: Path):
    """Load a labelled test image so ground truth and correctness can be shown."""
    st.session_state.update(input_bytes=path.read_bytes(), input_name=path.name, sample_label=label,
                            sample_path=path.relative_to(ROOT / "data/trashnet").as_posix(),
                            upload_revision=st.session_state.upload_revision + 1)
    st.session_state.dataset_browser = False


def choose_random_sample(label: str | None = None):
    """Sample within the requested class, or across the entire test catalog."""
    choices = [(name, path) for name, paths in sample_catalog().items()
               if label is None or name == label for path in paths]
    if not choices:
        return
    current_path = st.session_state.sample_path
    alternatives = [(name, path) for name, path in choices
                    if path.relative_to(ROOT / "data/trashnet").as_posix() != current_path]
    choose_sample(*random.choice(alternatives or choices))


def accept_upload(widget_key: str):
    """Uploader callback: store the new file's bytes and forget any dataset label."""
    file = st.session_state.get(widget_key)
    if file is not None:
        data = file.getvalue()
        st.session_state.update(input_bytes=data, input_name=file.name, sample_label=None, sample_path=None,
                                last_upload=(hashlib.sha256(data).hexdigest(), file.name, widget_key))


def file_picker():
    """Image uploader inside the change-image popover (or the session bridge on the stateless host)."""
    key = f"upload_{st.session_state.upload_revision}"
    if STATELESS_HOST:
        bridge_upload(key)
        return
    upload = st.file_uploader("Chọn ảnh JPG hoặc PNG", type=["jpg", "jpeg", "png"], key=key,
                              on_change=accept_upload, args=(key,), label_visibility="collapsed")
    if upload is not None:
        data = upload.getvalue()
        marker = (hashlib.sha256(data).hexdigest(), upload.name, key)
        # Also supports integrations that supply the file directly rather than via a callback.
        if marker != st.session_state.last_upload:
            st.session_state.update(input_bytes=data, input_name=upload.name, sample_label=None, sample_path=None,
                                    last_upload=marker)


def ground_truth() -> dict:
    """Reference labels are only for evaluation and are never classifier inputs."""
    dataset_label = st.session_state.sample_label
    return {"label": dataset_label, "source": "dataset" if dataset_label else None,
            "path": st.session_state.sample_path if dataset_label else None}


def render_dataset_label():
    """Show the manifest label of a dataset sample; uploads have no ground truth."""
    truth = ground_truth()
    if truth["source"] == "dataset":
        st.html(f'<div class="ground-truth-label">Nhãn thật: <strong>{LABELS_VI[truth["label"]]}</strong>'
                '<span>Từ nhãn dataset · tập test</span></div>')


def render_assessment(prediction: Prediction):
    """Ground-truth card: correct, incorrect or uncertain, for dataset samples only."""
    truth = ground_truth()
    if truth["source"] != "dataset":
        return
    assessment = assess_prediction(prediction.top_class, prediction.uncertain, truth["label"])
    status = assessment["status"]
    label = LABELS_VI[truth["label"]]
    st.html(f'<section class="assessment {status}"><div><div class="ground-truth-name">'
            f'Ground Truth: <strong>{label}</strong></div><span>Nhãn từ dataset · tập test</span></div>'
            f'<strong class="assessment-badge">{ASSESSMENT_LABELS[status]}</strong></section>')


def choose_preset(name: str):
    """Apply a Module 1 preset and switch the preview to the processed image."""
    st.session_state.ops_params = asdict(PRESETS[name])
    st.session_state.processing_level = name
    if st.session_state.get("image_view") != "So sánh":
        st.session_state.image_view = "Ảnh gốc" if name == "Gốc" else "Sau xử lý"


def prepare_editor():
    """Copy the active Module 1 settings into the editor's widget keys."""
    for key, value in st.session_state.ops_params.items():
        st.session_state[f"edit_{key}"] = value


def reset_editor():
    """Set every editor slider back to the identity (no-op) value."""
    for key, value in asdict(ops.OpsParams()).items():
        st.session_state[f"edit_{key}"] = value


def commit_editor():
    """Save the editor's values as the active Module 1 settings (the Apply button)."""
    st.session_state.ops_params = {
        key: st.session_state.get(f"edit_{key}", value) for key, value in asdict(ops.OpsParams()).items()
    }
    st.session_state.processing_level = "Tùy chỉnh"
    if st.session_state.get("image_view") != "So sánh":
        st.session_state.image_view = "Sau xử lý"


@st.dialog("Tùy chỉnh xử lý ảnh", width="large")
def custom_processing():
    """Dialog with every Module 1 control; changes apply only after Áp dụng."""
    st.caption("Module 1 · Chỉ những giá trị khác mặc định mới tác động lên ảnh. Bấm Áp dụng để lưu.")
    with st.form("processing_form"):
        color, geometry, smoothing = st.tabs(["Màu sắc", "Hình học", "Làm mượt & nhiễu"])
        with color:
            left, right = st.columns(2)
            with left:
                st.slider("Gamma", 0.2, 3.0, step=0.05, key="edit_gamma")
                st.slider("Tương phản", 0.2, 3.0, step=0.05, key="edit_contrast")
                st.slider("Độ sáng", -100, 100, step=1, key="edit_brightness")
                st.toggle("Ảnh xám", key="edit_grayscale")
            with right:
                st.slider("Dịch Hue", -90, 90, step=1, key="edit_hue_shift")
                st.slider("Hệ số Saturation", 0.0, 2.0, step=0.05, key="edit_saturation_scale")
                st.slider("Hệ số Value", 0.0, 2.0, step=0.05, key="edit_value_scale")
            st.slider("CLAHE clip limit (0 = tắt)", 0.0, 8.0, step=0.5, key="edit_clahe_clip")
            st.slider("CLAHE số ô lưới", 2, 16, step=1, key="edit_clahe_tiles")
        with geometry:
            st.slider("Góc xoay (độ)", -180, 180, step=1, key="edit_angle")
            st.slider("Tỉ lệ phóng", 0.5, 2.0, step=0.05, key="edit_scale")
            st.slider("Crop giữ lại phần trung tâm", 0.3, 1.0, step=0.05, key="edit_crop_keep")
            st.toggle("Lật ngang", key="edit_flip_horizontal")
            st.toggle("Lật dọc", key="edit_flip_vertical")
        with smoothing:
            st.slider("Gaussian σ (0 = tắt)", 0.0, 10.0, step=0.25, key="edit_gaussian_sigma")
            st.slider("Median kernel (1 = tắt)", 1, 15, step=2, key="edit_median_kernel")
            st.slider("Bilateral đường kính (0 = tắt)", 0, 15, step=1, key="edit_bilateral_diameter")
            st.slider("Bilateral σ màu", 10, 200, step=5, key="edit_bilateral_sigma_color")
            st.slider("Bilateral σ không gian", 10, 200, step=5, key="edit_bilateral_sigma_space")
            with st.expander("Nhiễu thử nghiệm"):
                st.slider("Nhiễu Gaussian σ", 0.0, 60.0, step=1.0, key="edit_noise_sigma")
                st.slider("Nhiễu muối tiêu", 0.0, 0.2, step=0.005, key="edit_salt_pepper")
        reset, submit = st.columns(2)
        reset.form_submit_button("Đặt lại", on_click=reset_editor, width="stretch")
        applied = submit.form_submit_button("Áp dụng", on_click=commit_editor, type="primary", width="stretch")
    if applied:
        st.rerun()


@st.dialog("Hướng dẫn sử dụng")
def show_help():
    """Short usage guide dialog."""
    st.markdown("1. **Chọn ảnh** hoặc thử một ảnh mẫu.\n"
                "2. Chọn **mô hình** và **mức xử lý ảnh** bên dưới vùng xem.\n"
                "3. Bấm **Phân loại** để xem kết quả và xác suất sáu lớp.\n"
                "4. Tải PNG/JSON hoặc so sánh ba mô hình trên cùng ảnh.")
    st.info("Mức xử lý là preset bộ lọc ảnh. Mức mạnh hơn không đảm bảo phân loại tốt hơn.")
    st.caption("Tab Video: tải video (hoặc tạo video mẫu), chọn mô hình và mức xử lý, bấm Xử lý video để nhận "
               "video có nhãn, biểu đồ xác suất theo thời gian, snapshot và CSV/JSON.")
    st.caption("Ảnh lớn được thu về tối đa 1.024 px ở cạnh dài. Chọn tab Webcam để phát hiện và phân loại vật trực tiếp; "
               "bấm Bật camera rồi cấp quyền, chọn mô hình và nhấp khung bao để xem xác suất. Bấm Dừng camera để kết thúc.")


@st.dialog("Thông tin mô hình", width="large")
def show_model_info():
    """Dialog describing the three models and how their confidence is computed."""
    for key, profile in MODEL_PROFILES.items():
        st.subheader(MODEL_NAMES[key])
        st.write(profile.description)
        st.caption(profile.training)
    st.info("MLP cải tiến hiệu chỉnh xác suất bằng temperature scaling T=2, ngưỡng không chắc 52,58%. "
            "Baseline và MobileNet dùng softmax chưa hiệu chỉnh.")
    st.caption("Suy luận hiện tại chạy PyTorch trên CPU. Dataset: TrashNet, sáu lớp.")


def render_header():
    """Brand bar with the model-information and help buttons."""
    with st.container(key="brand_header"):
        brand, info, help_col = st.columns([5, 2.5, 1.7], vertical_alignment="center")
        brand.html('<div class="brand"><span class="brand-icon" aria-hidden="true"></span>'
                   '<div><strong>WasteLens</strong>'
                   '<span class="brand-badge">VISION</span><small>Phân loại rác &amp; Xử lý ảnh số</small></div></div>')
        if info.button("Thông tin mô hình", icon=":material/info:", type="tertiary", width="stretch"):
            show_model_info()
        if help_col.button("Hướng dẫn", icon=":material/help:", type="tertiary", width="stretch"):
            show_help()


def render_viewer():
    """Input card first. Input bytes persist when the picker moves into the change-image popover."""
    image, original_size = None, None
    with st.container(key="image_card", border=True):
        with st.container(key="viewer_header"):
            navigation, change, delete = st.columns([6, 2, 0.7], vertical_alignment="center")
            with navigation:
                view = st.segmented_control("Chế độ xem", ["Ảnh gốc", "Sau xử lý", "So sánh"],
                                            required=True, key="image_view", label_visibility="collapsed")
            with change:
                with st.popover("Đổi ảnh", icon=":material/upload_file:", width="stretch"):
                    file_picker()
            delete.button("Xóa ảnh", icon=":material/delete:", type="tertiary", key="clear_image",
                          help="Xóa ảnh hiện tại", on_click=clear_image, disabled=not st.session_state.input_bytes,
                          width="stretch")
        if st.session_state.input_bytes is None:
            st.html('<div class="empty-view"><span class="empty-icon">↑</span>'
                    '<h2>Đưa một bức ảnh vào đây.</h2><p>Khám phá ảnh, thử các phép xử lý và phân loại rác.</p></div>')
            file_picker_empty()
        else:
            try:
                image, original_size = decode_image(st.session_state.input_bytes)
            except (OSError, ValueError, Image.DecompressionBombError):
                st.error(UPLOAD_ERROR)
                st.caption("Bấm Đổi ảnh để chọn lại. Các thiết lập xử lý của bạn vẫn được giữ.")
        params = ops.OpsParams(**st.session_state.ops_params)
        started = time.perf_counter()
        processed = ops.apply(image, params) if image is not None else None
        ops_ms = (time.perf_counter() - started) * 1000
        if image is not None:
            if view == "So sánh":
                before, after = st.columns(2)
                with before:
                    show_image(image, "Ảnh gốc")
                with after:
                    show_image(processed, "Sau xử lý")
            else:
                shown = processed if view == "Sau xử lý" else image
                show_image(shown, "Ảnh sau xử lý" if view == "Sau xử lý" else "Ảnh đầu vào")
            width, height = original_size
            metadata = (f'{escape(st.session_state.input_name)} <span>· {width} × {height} px'
                        f' · {len(st.session_state.input_bytes) / 1024:.1f} KB</span>')
            if max(original_size) > MAX_SIDE:
                metadata += f' <span>· Xử lý ở {image.shape[1]} × {image.shape[0]} px</span>'
            st.html(f'<div class="file-metadata">{metadata}</div>')
            render_dataset_label()
        samples = sample_catalog()
        if samples:
            with st.container(key="sample_strip"):
                caption, browse = st.columns([2, 2], vertical_alignment="center")
                caption.caption("THỬ ẢNH MẪU · TẬP TEST")
                with browse.popover("Chọn ảnh trong tập test", icon=":material/photo_library:", width="stretch",
                                    key="dataset_browser", on_change="rerun"):
                    choices = [(label, path) for label, paths in samples.items() for path in paths]
                    st.caption(f"{len(choices)} ảnh có nhãn thật từ manifest dataset.")
                    selected = st.selectbox("Ảnh trong tập test", choices, key="dataset_image",
                                            format_func=lambda item: f"{LABELS_VI[item[0]]} · {item[1].name}")
                    st.button("Dùng ảnh này", key="use_dataset_image", on_click=choose_sample,
                              args=selected, width="stretch", type="primary")
                with st.container(key="sample_buttons"):
                    columns = st.columns(len(samples) + 1)
                    for column, label in zip(columns[:-1], samples, strict=True):
                        column.button(LABELS_VI[label], key=f"sample_{label}", on_click=choose_random_sample,
                                      args=(label,), width="stretch", help=f"Ảnh ngẫu nhiên cùng loại: {LABELS_VI[label]}",
                                      type="primary" if st.session_state.sample_label == label else "secondary")
                    columns[-1].button("Random rác", key="sample_random", on_click=choose_random_sample,
                                       icon=":material/shuffle:", width="stretch",
                                       help="Chọn ngẫu nhiên một ảnh test từ cả 6 loại rác.")
    return image, processed, params, ops_ms


def file_picker_empty():
    # A separate widget key avoids two copies of the same uploader in the initial card.
    """Large uploader shown in the empty image card before any image is chosen."""
    key = f"empty_upload_{st.session_state.upload_revision}"
    if STATELESS_HOST:
        bridge_upload(key)
        return
    upload = st.file_uploader("Thả ảnh vào đây hoặc bấm để chọn", type=["jpg", "jpeg", "png"],
                              key=key, on_change=accept_upload, args=(key,))
    if upload is not None:
        marker = (hashlib.sha256(upload.getvalue()).hexdigest(), upload.name, key)
        if marker != st.session_state.last_upload:
            st.session_state.update(input_bytes=upload.getvalue(), input_name=upload.name, sample_label=None,
                                    sample_path=None, last_upload=marker)
            st.rerun()


def render_toolbar(has_image: bool):
    """Model, processing level and run controls; returns (model ID, run clicked, checkpoint present)."""
    key = st.session_state.get("model_key", DEFAULT_MODEL)
    level = st.session_state.processing_level
    with st.container(key="action_bar", border=True):
        model_col, processing_col, source_col, run_col = st.columns([3, 2.3, 1.1, 2], vertical_alignment="center")
        with model_col:
            with st.popover(f"Mô hình: {MODEL_NAMES[key]}", icon=":material/psychology:", width="stretch"):
                st.radio("Chọn mô hình", list(MODEL_PROFILES), key="model_key",
                         index=list(MODEL_PROFILES).index(DEFAULT_MODEL), format_func=lambda item: MODEL_NAMES[item])
                st.caption(MODEL_PROFILES[st.session_state.model_key].description)
        with processing_col:
            with st.popover(f"Xử lý ảnh: {level}", icon=":material/auto_fix_high:", width="stretch"):
                st.caption("CHỌN MỨC XỬ LÝ")
                for name in PRESETS:
                    st.button(name, key=f"preset_{name}", on_click=choose_preset, args=(name,),
                              help=PRESET_DESCRIPTIONS[name], width="stretch",
                              type="primary" if name == level else "secondary")
                if st.button("Tùy chỉnh chi tiết", icon=":material/tune:", width="stretch"):
                    prepare_editor()
                    custom_processing()
                st.caption("Mức mạnh hơn không đảm bảo accuracy cao hơn.")
        with source_col:
            with st.popover("Ảnh", icon=":material/image:", width="stretch"):
                st.write("Ảnh JPG / PNG")
                st.caption("Chuyển sang tab Video hoặc Webcam ở đầu trang để xử lý video.")
        key = st.session_state.get("model_key", DEFAULT_MODEL)
        available = MODEL_PROFILES[key].checkpoint.is_file()
        run = run_col.button("Phân loại" if st.session_state.result is None else "Phân loại lại",
                             key="run_inference", type="primary", icon=":material/bolt:",
                             width="stretch", disabled=not has_image or not available)
    params = ops.OpsParams(**st.session_state.ops_params)
    with st.container(key="active_operations"):
        summary, edit = st.columns([8, 1.5], vertical_alignment="center")
        summary.caption("●  " + describe_operations(params))
        if edit.button("Chỉnh sửa", type="tertiary", icon=":material/edit:", width="stretch"):
            prepare_editor()
            custom_processing()
    if not available:
        st.error(f"Thiếu checkpoint cho {MODEL_NAMES[key]}. Chọn mô hình khác hoặc nạp file {MODEL_PROFILES[key].filename}.")
    return key, run, available


def annotate(image: np.ndarray, prediction: Prediction, true_label: str | None = None) -> np.ndarray:
    """OpenCV's built-in font uses ASCII; metadata in JSON retains Vietnamese labels."""
    out = image.copy()
    text = (f"uncertain (closest: {prediction.top_class} {prediction.confidence:.0%})" if prediction.uncertain
            else f"{prediction.label} {prediction.confidence:.0%}")
    lines = [text]
    if true_label is not None:
        lines.append(f"truth: {true_label} | top1: {'correct' if prediction.top_class == true_label else 'incorrect'}")
    scale = max(0.3, min(out.shape[1] / max(max(map(len, lines)) * 12, 1), out.shape[0] / (60 * len(lines)), 1.0))
    thickness = max(1, round(scale * 2))
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness) for line in lines]
    pitch = max(height + baseline for (_, height), baseline in sizes) + 10
    color = ((0, 165, 255) if prediction.uncertain else (60, 60, 185)
             if true_label is not None and prediction.top_class != true_label else (95, 104, 0))
    cv2.rectangle(out, (0, 0), (max(size[0][0] for size in sizes) + 16, pitch * len(lines) + 8), color, -1)
    for index, (line, ((_, height), _)) in enumerate(zip(lines, sizes, strict=True)):
        cv2.putText(out, line, (8, height + 8 + index * pitch), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def run_inference(image, processed, params, ops_ms, model_key, signature):
    """Classify the processed image (and the original, if Module 1 changed it) and store the result."""
    classifier = profile_model(model_key)
    prediction = classifier.classify(processed)
    original = classifier.classify(image) if params.changed() else prediction
    st.session_state.result = {
        "signature": signature, "model_key": model_key, "prediction": prediction, "original": original,
        "params": asdict(params), "processed": processed.copy(), "name": st.session_state.input_name,
        "time": datetime.now().isoformat(timespec="seconds"), "ops_ms": ops_ms,
        "calibrated": classifier.calibrated, "threshold": classifier.threshold, "comparisons": {},
    }


def probability_table(prediction: Prediction | None):
    """Six-class probability table with proportional bars, highest first."""
    rows = []
    values = sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True) if prediction else [
        (name, None) for name in LABELS_VI if name != "uncertain"]
    for rank, (name, value) in enumerate(values, 1):
        top = prediction is not None and name == prediction.top_class
        color = "#00685f" if top else CLASS_COLORS[name]
        badge = '<span class="top-badge">TOP 1</span>' if top else ""
        percent = f"{value:.1%}" if value is not None else "—"
        width = 100 * value if value is not None else 0
        rows.append(
            f'<tr class="{"top-row" if top else ""}"><td><span class="class-dot" style="background:{CLASS_COLORS[name]}"></span>'
            f'{rank}. {LABELS_VI[name]} <span class="english">({name.capitalize()})</span> {badge}</td>'
            f'<td class="prob-value">{percent}</td><td><div class="prob-track"><div class="prob-fill" '
            f'style="width:{width:.6f}%;background:{color}"></div></div></td></tr>')
    st.html('<div class="distribution-title"><strong>PHÂN PHỐI XÁC SUẤT 6 LỚP TRASHNET</strong>'
            f'<span>{"Tổng chuẩn hóa: 100%" if prediction else "Chưa chạy phân loại"}</span></div>'
            '<div class="probability-scroll"><table class="probability-table"><thead><tr>'
            '<th>LỚP RÁC THẢI</th><th>XÁC SUẤT</th><th>PHÂN PHỐI</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def compare_models(bundle):
    """Run all three models on the same processed image and show their predictions side by side."""
    rows = []
    truth = ground_truth()["label"]
    for key, profile in MODEL_PROFILES.items():
        if not profile.checkpoint.is_file():
            st.warning(f"Thiếu checkpoint: {MODEL_NAMES[key]}")
            continue
        classifier = profile_model(key)
        version = profile.checkpoint.stat().st_mtime_ns
        cached = bundle["comparisons"].get(key)
        if cached is None or cached["version"] != version:
            prediction = bundle["prediction"] if key == bundle["model_key"] else classifier.classify(bundle["processed"])
            cached = {
                "version": version, "checkpoint": profile.filename, "training": profile.training,
                "label": prediction.label, "top_class": prediction.top_class, "confidence": prediction.confidence,
                "calibrated": classifier.calibrated, "uncertain_threshold": classifier.threshold,
                "probabilities": prediction.probabilities, "timings_ms": prediction.timings_ms,
            }
            bundle["comparisons"][key] = cached
        assessment = assess_prediction(cached["top_class"], cached["label"] == "uncertain", truth)
        verdict = ASSESSMENT_LABELS[assessment["status"]]
        predicted_label = LABELS_VI[cached["label"]]
        if cached["label"] == "uncertain":
            predicted_label += f' (gần nhất: {LABELS_VI[cached["top_class"]]})'
        row = {"Tên mô hình": MODEL_NAMES[key], "Dự đoán của mô hình": predicted_label}
        if truth is not None:
            row.update({"Nhãn thật": LABELS_VI[truth], "Đối chiếu": verdict})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Cả ba mô hình được đối chiếu với cùng nhãn thật trên cùng một ảnh sau Module 1." if truth is not None
               else "Dự đoán của ba mô hình trên cùng một ảnh sau Module 1.")


def render_result(signature: str | None):
    """Result card; shown only while the image, settings and model still match the stored result."""
    bundle = st.session_state.result
    with st.container(key="result_card", border=True):
        if bundle is None:
            st.html('<div class="result-placeholder"><strong>Sẵn sàng khi bạn sẵn sàng.</strong>'
                    '<p>Chọn ảnh và bấm Phân loại để xem kết quả.</p></div>')
            probability_table(None)
            return
        if bundle["signature"] != signature:
            st.warning("Ảnh, mô hình hoặc mức xử lý đã thay đổi. Bấm Phân loại lại để cập nhật kết quả.")
            return
        prediction, key = bundle["prediction"], bundle["model_key"]
        label = LABELS_VI[prediction.label]
        confidence_kind = "Đã hiệu chỉnh" if bundle["calibrated"] else "Softmax chưa hiệu chỉnh"
        st.html(
            f'<div class="prediction-summary {"uncertain" if prediction.uncertain else ""}">'
            f'<div class="prediction-symbol" style="background:{CLASS_COLORS[prediction.top_class]}">♻</div>'
            f'<div class="prediction-copy"><div>KẾT QUẢ DỰ ĐOÁN: <strong>{label}</strong></div>'
            f'<p>Mô hình: <strong>{MODEL_NAMES[key]}</strong> · Suy luận: {prediction.timings_ms["total"]:.1f} ms</p>'
            f'</div><div class="confidence"><strong>{prediction.confidence:.1%}</strong>'
            f'<span>{confidence_kind}</span></div></div>')
        if prediction.uncertain:
            st.caption(f"Lớp gần nhất: {LABELS_VI[prediction.top_class]} · Ngưỡng không chắc {bundle['threshold']:.1%}.")
        if ops.OpsParams(**bundle["params"]).changed():
            original = bundle["original"]
            st.caption(f"Trên ảnh gốc: {LABELS_VI[original.label]} {original.confidence:.1%} "
                       f"→ Sau xử lý: {label} {prediction.confidence:.1%}")
        render_assessment(prediction)
        probability_table(prediction)
        compare = st.toggle("So sánh cả 3 mô hình trên ảnh này", key="compare_models")
        if compare:
            with st.spinner("Đang so sánh mô hình…"):
                compare_models(bundle)
        render_exports(bundle, compare)


def render_exports(bundle, include_comparisons: bool):
    """Download buttons for the annotated PNG, the processed PNG and the JSON record."""
    prediction, original = bundle["prediction"], bundle["original"]
    key, profile = bundle["model_key"], MODEL_PROFILES[bundle["model_key"]]
    truth = ground_truth()
    comparisons = {
        model_key: {**result, "evaluation": assess_prediction(result["top_class"], result["label"] == "uncertain", truth["label"])}
        for model_key, result in bundle["comparisons"].items()
    } if include_comparisons else {}
    record = {
        "time": bundle["time"], "source": bundle["name"],
        "module1": ops.OpsParams(**bundle["params"]).changed(),
        "prediction": {"label": prediction.label, "top_class": prediction.top_class,
                       "confidence": prediction.confidence, "probabilities": prediction.probabilities,
                       "uncertain_threshold": bundle["threshold"], "calibrated": bundle["calibrated"]},
        "original_prediction": {"label": original.label, "confidence": original.confidence},
        "timings_ms": {"module1": bundle["ops_ms"], **prediction.timings_ms,
                       "pipeline_total": bundle["ops_ms"] + prediction.timings_ms["total"]},
        "model": profile.filename, "model_details": {"id": key, "name": profile.name, "training": profile.training},
        "ground_truth": truth,
        "evaluation": assess_prediction(prediction.top_class, prediction.uncertain, truth["label"]),
        "comparisons": comparisons,
    }
    stamp = bundle["time"].replace(":", "").replace("-", "")
    if STATELESS_HOST:
        ok, annotated = cv2.imencode(".png", annotate(bundle["processed"], prediction, truth["label"]))
        clean_ok, clean = cv2.imencode(".png", bundle["processed"])
        if ok and clean_ok:
            files = [
                {"label": "Tải ảnh kết quả", "name": f"wastelens-{key}-{stamp}.png", "mime": "image/png",
                 "data": base64.b64encode(annotated).decode("ascii")},
                {"label": "Tải JSON", "name": f"wastelens-{key}-{stamp}.json", "mime": "application/json",
                 "data": base64.b64encode(json.dumps(record, ensure_ascii=False, indent=2).encode()).decode("ascii")},
                {"label": "Tải ảnh sau xử lý", "name": f"wastelens-{key}-{stamp}-processed.png", "mime": "image/png",
                 "data": base64.b64encode(clean).decode("ascii")},
            ]
            file_bridge(data={"mode": "downloads", "files": files}, key="downloads_bridge")
        return
    with st.container(key="export_actions"):
        image_col, json_col, clean_col = st.columns(3)
        ok, png = cv2.imencode(".png", annotate(bundle["processed"], prediction, truth["label"]))
        image_col.download_button("Tải ảnh kết quả", png.tobytes() if ok else b"",
                                  f"wastelens-{key}-{stamp}.png", "image/png", disabled=not ok,
                                  icon=":material/download:", width="stretch", on_click="ignore")
        json_col.download_button("Tải JSON", json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
                                 f"wastelens-{key}-{stamp}.json", "application/json",
                                 icon=":material/data_object:", width="stretch", on_click="ignore")
        ok, clean = cv2.imencode(".png", bundle["processed"])
        clean_col.download_button("Tải ảnh sau xử lý", clean.tobytes() if ok else b"",
                                  f"wastelens-{key}-{stamp}-processed.png", "image/png", disabled=not ok,
                                  icon=":material/image:", width="stretch", on_click="ignore")


VIDEO_DIR = Path(tempfile.gettempdir()) / "wastelens-video"  # Uploaded and processed videos for this machine.
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]


def video_processing_options() -> dict[str, ops.OpsParams]:
    """The image presets, plus the custom Module 1 settings when they were edited in the image tab."""
    options = dict(PRESETS)
    custom = ops.OpsParams(**st.session_state.ops_params)
    if custom.changed() and st.session_state.processing_level == "Tùy chỉnh":
        options["Tùy chỉnh (từ tab Ảnh)"] = custom
    return options


def render_video():
    """Video tab: choose a video, model and Module 1 settings, then process it on request."""
    st.caption("Module 1 xử lý mọi khung hình → mô hình phân loại mỗi N khung → làm mượt xác suất theo thời gian "
               "→ video có nhãn, biểu đồ xác suất và file kết quả.")
    if STATELESS_HOST:
        st.info("Chế độ video cần chạy ứng dụng trên máy cục bộ; bản triển khai chỉ nhận tệp tối đa 4 MB.")
        return
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    upload_col, sample_col = st.columns([3, 1], vertical_alignment="bottom")
    upload = upload_col.file_uploader("Chọn video MP4, MOV, AVI, MKV hoặc WEBM", type=VIDEO_TYPES, key="video_upload")
    catalog = sample_catalog()
    if sample_col.button("Tạo video mẫu từ tập test", icon=":material/auto_awesome_motion:", width="stretch",
                         disabled=not catalog, help="Mỗi lớp một ảnh test, 2 giây mỗi ảnh, có chuyển động phóng chậm."):
        images = [(label, random.choice(catalog[label])) for label in CLASS_COLORS if label in catalog]
        path = VIDEO_DIR / f"sample-{datetime.now():%Y%m%d-%H%M%S}.mp4"
        segments = make_sample_video(images, path)
        st.session_state.video_source = {"path": path, "name": path.name, "segments": segments,
                                         "digest": hashlib.sha256(path.read_bytes()).hexdigest()}
    if upload is not None:
        data = upload.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        current = st.session_state.get("video_source")
        if current is None or current.get("upload_digest") != digest:
            path = VIDEO_DIR / f"input-{digest[:16]}{Path(upload.name).suffix.lower()}"
            if not path.exists():
                path.write_bytes(data)  # OpenCV and PyAV read from a file path, not from memory.
            st.session_state.video_source = {"path": path, "name": upload.name, "segments": None,
                                             "digest": digest, "upload_digest": digest}
    chosen = st.session_state.get("video_source")
    if chosen is None:
        st.info("Tải một video lên, hoặc tạo video mẫu từ ảnh test. Video dài chỉ xử lý 60 giây đầu (chỉnh trong Thiết lập).")
        return
    source, digest = chosen["path"], chosen["digest"]
    if chosen["segments"]:
        st.caption("Video mẫu · thứ tự nhãn thật: " + " → ".join(LABELS_VI[s["label"]] for s in chosen["segments"]))
    try:
        info = probe(source)
    except ValueError:
        st.error("Không đọc được video. Hãy thử tệp MP4 (H.264) khác.")
        return

    options = video_processing_options()
    model_col, processing_col, settings_col = st.columns([4, 3, 2])
    key = model_col.selectbox("Mô hình phân loại", list(MODEL_NAMES), format_func=MODEL_NAMES.get,
                              index=list(MODEL_NAMES).index(DEFAULT_MODEL), key="video_model")
    preset = processing_col.selectbox("Xử lý khung hình (Module 1)", list(options), key="video_preset")
    detect = st.toggle("Phát hiện từng vật (SSDLite) rồi phân loại từng vùng cắt", key="video_detect",
                       help="Tắt: phân loại cả khung hình như một ảnh. Bật: tìm tối đa N vật mỗi khung, "
                            "theo dõi ID và gắn nhãn rác riêng cho từng vật, giống tab Webcam.")
    with settings_col.popover("Thiết lập video", width="stretch"):
        every = st.slider("Phân loại mỗi N khung hình", 1, 30, max(1, round(info["fps"] / 5)), key="video_every",
                          help="N nhỏ chính xác theo thời gian hơn nhưng chậm hơn.")
        max_seconds = st.slider("Xử lý tối đa (giây đầu video)", 5, 120, 60, 5, key="video_max_seconds")
        max_side = st.selectbox("Độ phân giải đầu ra (cạnh dài)", [480, 720, 960, 1280], index=2,
                                format_func=lambda size: f"{size} px", key="video_max_side")
        smoothing = st.slider("Trọng số dự đoán mới (EMA)", 0.1, 1.0, 0.5, 0.1, key="video_smoothing",
                              help="1,0 = không làm mượt; nhỏ hơn = nhãn ổn định hơn nhưng đổi chậm hơn.")
        detector_threshold, max_objects = 0.45, 3
        if detect:
            detector_threshold = st.slider("Ngưỡng phát hiện vật", .2, .9, .45, .05, key="video_detector_threshold",
                                           help="Thấp hơn bắt được vật nhỏ hơn nhưng có thêm khung thừa.")
            max_objects = st.slider("Số vật tối đa mỗi khung", 1, 8, 3, key="video_max_objects")
    settings = VideoSettings(classify_every=every, max_seconds=float(max_seconds), max_side=max_side,
                             smoothing=smoothing, detect_objects=detect, detector_threshold=detector_threshold,
                             max_objects=max_objects)
    if detect and not DETECTOR_WEIGHTS.is_file():
        st.info("Cần tải bộ phát hiện vật trước khi bật phát hiện từng vật.")
        st.code(".venv\\Scripts\\python.exe scripts/prepare_detector.py", language="powershell")
        return
    params = options[preset]
    duration = info["duration_s"]
    st.caption(f"{chosen['name']} · {info['width']}×{info['height']} · {info['fps']:.2f} fps"
               + (f" · {duration:.1f} giây" if duration else "")
               + f" · Module 1: {describe_operations(params)}")
    if not MODEL_PROFILES[key].checkpoint.is_file():
        st.error(f"Thiếu checkpoint: {MODEL_PROFILES[key].filename}")
        return
    version = MODEL_PROFILES[key].checkpoint.stat().st_mtime_ns
    signature = hashlib.sha256(json.dumps(
        {"video": digest, "params": asdict(params), "model": key, "version": version, "settings": asdict(settings)},
        sort_keys=True).encode()).hexdigest()
    bundle = st.session_state.get("video_result")
    label = "Xử lý lại video" if bundle else "Xử lý video"
    if st.button(label, type="primary", icon=":material/movie:", key="video_run", width="stretch"):
        progress = st.progress(0.0, text="Đang xử lý video…")
        classifier = profile_model(key)
        destination = VIDEO_DIR / f"output-{signature[:16]}.mp4"
        detector = load_detector(str(DETECTOR_WEIGHTS), DETECTOR_WEIGHTS.stat().st_mtime_ns) if detect else None
        try:
            result = process_video(source, destination, classifier, params, settings,
                                   progress=lambda fraction, text: progress.progress(fraction, text=text),
                                   detector=detector)
        except (ValueError, RuntimeError, OSError):
            progress.empty()
            st.error("Không xử lý được video này. Hãy thử tệp MP4 (H.264) khác.")
            return
        st.session_state.video_result = {
            "signature": signature, "result": result, "model_key": key, "preset": preset,
            "params": asdict(params), "settings": asdict(settings), "name": chosen["name"], "source": source,
            "segments": chosen["segments"],
            "time": datetime.now().isoformat(timespec="seconds"), "calibrated": classifier.calibrated,
            "threshold": classifier.threshold, "classes": classifier.classes,
        }
        st.rerun()
    if bundle is None:
        st.video(str(source))
        return
    if bundle["signature"] != signature:
        st.warning("Video, mô hình hoặc thiết lập đã thay đổi. Bấm Xử lý lại video để cập nhật kết quả.")
        st.video(str(source))
        return
    render_video_result(bundle)


def render_video_players(bundle):
    """Input and annotated output side by side."""
    result = bundle["result"]
    before, after = st.columns(2)
    with before:
        st.caption("Video gốc")
        st.video(str(bundle["source"]))
    with after:
        st.caption(f"Sau xử lý và phân loại · {result.width}×{result.height}")
        st.video(str(result.output))
        if not result.browser_playable:
            st.caption("Trình duyệt có thể không phát được codec dự phòng; hãy tải video về để xem.")


def render_video_detection_result(bundle):
    """Detection-mode view: one row per tracked object, a per-frame snapshot and downloads."""
    result, key, classes = bundle["result"], bundle["model_key"], bundle["classes"]
    summary = result.summary()
    render_video_players(bundle)
    tracks = summary["tracks"]
    with st.container(key="video_summary", border=True):
        st.html(f'<div class="prediction-summary"><div class="prediction-symbol" style="background:#0f766e">♻</div>'
                f'<div class="prediction-copy"><div>ĐÃ THEO DÕI: <strong>{len(tracks)} vật</strong></div>'
                f'<p>Có vật ở {summary["frames_with_objects"]}/{summary["analysed_frames"]} khung được phân tích · '
                f'Mô hình: <strong>{MODEL_NAMES[key]}</strong> · Xử lý: {escape(bundle["preset"])}</p></div></div>')
        stats = st.columns(4)
        stats[0].metric("Khung hình đã xử lý", f"{result.frames_read}",
                        help="Đã dừng ở giới hạn thời gian." if result.truncated else None)
        stats[1].metric("Lượt phát hiện", f"{summary['detections']}")
        stats[2].metric("Tốc độ xử lý (khung/s)", f"{summary['processing_fps']:.0f}")
        stats[3].metric("Phân loại TB mỗi vật (ms)",
                        f"{summary['mean_inference_ms']:.1f}" if summary["mean_inference_ms"] is not None else "–")
        st.caption("SSDLite chỉ khoanh vùng (tên lớp COCO ở cột “Bộ phát hiện”); nhãn rác do mô hình 6 lớp quyết định "
                   "và được làm mượt EMA riêng cho từng ID. ID có thể đổi khi vật bị che hoặc di chuyển nhanh.")
    if not tracks:
        st.warning("Không phát hiện được vật nào. Thử hạ ngưỡng phát hiện hoặc tăng độ phân giải trong Thiết lập video.")
    else:
        st.markdown("**Từng vật được theo dõi**")
        st.dataframe(pd.DataFrame([{
            "ID": f"#{t['object_id']}", "Bộ phát hiện (COCO)": t["detector_category"],
            "Nhãn rác chiếm ưu thế": LABELS_VI[t["dominant_label"]], "Tỷ lệ lần thấy": f"{t['dominant_share']:.0%}",
            "Không chắc": f"{t['uncertain_share']:.0%}", "Confidence TB": f"{t['mean_confidence']:.0%}",
            "Xuất hiện (s)": f"{t['first_s']:.1f}–{t['last_s']:.1f}", "Số lần thấy": t["appearances"],
        } for t in tracks]), hide_index=True, width="stretch")

    snapshot, frame_index = None, 0
    frames = sorted({record.frame for record in result.objects})
    if frames:
        st.markdown("**Snapshot một khung hình**")
        chosen = st.select_slider("Thời điểm", options=frames, format_func=lambda f: f"{f / result.fps:.1f} s",
                                  key="video_detection_snapshot")
        frame_index = chosen
        snapshot = read_frame(result.output, chosen)
        left, right = st.columns([3, 2])
        if snapshot is not None:
            with left:
                show_image(snapshot, f"Khung {chosen} · {chosen / result.fps:.1f} s")
        with right:
            for record in [r for r in result.objects if r.frame == chosen]:
                st.markdown(f"**#{record.object_id} · {LABELS_VI[record.smoothed_label]} "
                            f"{record.smoothed_confidence:.1%}**")
                st.caption(f"Bộ phát hiện: {record.detector_category} ({record.detector_confidence:.0%}) · "
                           f"Khung này: {LABELS_VI[record.top_class]} {record.confidence:.1%}")

    stamp = bundle["time"].replace(":", "").replace("-", "")
    record_json = {
        "time": bundle["time"], "source": bundle["name"], "module1": ops.OpsParams(**bundle["params"]).changed(),
        "preset": bundle["preset"], "video_settings": bundle["settings"], "summary": summary,
        "model": MODEL_PROFILES[key].filename, "detector": DETECTOR_WEIGHTS.name,
        "calibrated": bundle["calibrated"], "uncertain_threshold": bundle["threshold"],
    }
    with st.container(key="video_exports"):
        video_col, csv_col, json_col, snap_col = st.columns(4)
        video_col.download_button("Tải video kết quả", result.output.read_bytes(), f"wastelens-{key}-{stamp}.mp4",
                                  "video/mp4", icon=":material/movie:", width="stretch", on_click="ignore")
        csv_col.download_button("Tải CSV theo vật", objects_csv(result, classes).encode("utf-8"),
                                f"wastelens-{key}-{stamp}-objects.csv", "text/csv", icon=":material/table:",
                                width="stretch", on_click="ignore")
        json_col.download_button("Tải JSON", json.dumps(record_json, ensure_ascii=False, indent=2).encode("utf-8"),
                                 f"wastelens-{key}-{stamp}.json", "application/json", icon=":material/data_object:",
                                 width="stretch", on_click="ignore")
        ok, png = cv2.imencode(".png", snapshot) if snapshot is not None else (False, None)
        snap_col.download_button("Tải snapshot", png.tobytes() if ok else b"",
                                 f"wastelens-{key}-{stamp}-t{frame_index / result.fps:.1f}s.png", "image/png",
                                 disabled=not ok, icon=":material/photo_camera:", width="stretch", on_click="ignore")


def render_video_result(bundle):
    """Processed-video view: players, summary, probability timeline, snapshot and downloads."""
    result, key, classes = bundle["result"], bundle["model_key"], bundle["classes"]
    if getattr(result, "detection", False):  # Results kept from before a code reload may lack the field.
        render_video_detection_result(bundle)
        return
    summary = result.summary()
    render_video_players(bundle)
    dominant = summary["dominant_confident_label"] or "uncertain"
    share = summary["label_shares"].get(dominant, 0.0)
    uncertain_note = (f" · Không chắc: {summary['uncertain_share']:.0%} số khung"
                      if summary["uncertain_share"] and dominant != "uncertain" else "")
    confidence_kind = "đã hiệu chỉnh" if bundle["calibrated"] else "softmax chưa hiệu chỉnh"
    with st.container(key="video_summary", border=True):
        st.html(f'<div class="prediction-summary"><div class="prediction-symbol" '
                f'style="background:{CLASS_COLORS.get(dominant, "#f59e0b")}">♻</div>'
                f'<div class="prediction-copy"><div>NHÃN CHIẾM ƯU THẾ: <strong>{LABELS_VI[dominant]}</strong></div>'
                f'<p>{share:.0%} số khung được phân loại{uncertain_note} · Mô hình: <strong>{MODEL_NAMES[key]}</strong> '
                f'· Xử lý: {escape(bundle["preset"])}</p></div></div>')
        stats = st.columns(4)
        stats[0].metric("Khung hình đã xử lý", f"{result.frames_read}",
                        help="Đã dừng ở giới hạn thời gian." if result.truncated else None)
        stats[1].metric("Khung được phân loại", f"{summary['classified_frames']}")
        stats[2].metric("Tốc độ xử lý (khung/s)", f"{summary['processing_fps']:.0f}")
        stats[3].metric("Suy luận TB (ms)", f"{summary['mean_inference_ms']:.1f}")
        per_frame = summary["mean_ms_per_frame"]
        st.caption("Thời gian trung bình mỗi khung (ms): " + " · ".join(
            f"{name} {per_frame[stage]:.1f}" for stage, name in
            (("decode", "Giải mã"), ("module1", "Module 1"), ("classify", "Phân loại"), ("encode", "Mã hóa"))))
        if result.truncated:
            st.caption(f"Video dài hơn giới hạn; chỉ {result.frames_read / result.fps:.0f} giây đầu được xử lý.")
        if bundle.get("segments"):
            agreement = segment_agreement(result, bundle["segments"])
            if agreement["frames"]:
                st.caption(f"Đối chiếu nhãn thật của video mẫu ({agreement['frames']} khung được phân loại). "
                           f"Top-1 từng khung: đúng {agreement['raw_top1']:.0%}. Nhãn hiển thị sau làm mượt: "
                           f"đúng {agreement['smoothed']:.0%} · không chắc {agreement['smoothed_uncertain']:.0%} · "
                           f"sai {agreement['smoothed_wrong']:.0%}. Ảnh test trên nền TrashNet, không phải video thực tế.")

    st.markdown("**Xác suất 6 lớp theo thời gian** (sau làm mượt EMA, " + confidence_kind + ")")
    frame = pd.DataFrame([{**{LABELS_VI[name]: 100 * record.smoothed_probabilities[name] for name in classes},
                           "Thời gian (s)": round(record.time_s, 2)} for record in result.records])
    st.line_chart(frame, x="Thời gian (s)", y=[LABELS_VI[name] for name in classes],
                  color=[CLASS_COLORS[name] for name in classes], y_label="Xác suất (%)", height=280)
    if bundle["calibrated"]:
        st.caption(f"Khung có xác suất cao nhất dưới {bundle['threshold']:.1%} được gắn nhãn “Không chắc”.")

    st.markdown("**Snapshot một khung hình**")
    times = [record.time_s for record in result.records]
    chosen = st.select_slider("Thời điểm", options=list(range(len(times))), format_func=lambda i: f"{times[i]:.1f} s",
                              key="video_snapshot")
    record = result.records[chosen]
    snapshot = read_frame(result.output, record.frame)
    left, right = st.columns([3, 2])
    if snapshot is not None:
        with left:
            show_image(snapshot, f"Khung {record.frame} · {record.time_s:.1f} s")
    with right:
        st.markdown(f"Khung này: **{LABELS_VI[record.top_class]}** {record.confidence:.1%} · "
                    f"sau làm mượt: **{LABELS_VI[record.smoothed_label]}** {record.smoothed_confidence:.1%}")
        probability_table(Prediction(label=record.smoothed_label, top_class=max(record.smoothed_probabilities,
                                     key=record.smoothed_probabilities.get), confidence=record.smoothed_confidence,
                                     probabilities=record.smoothed_probabilities))

    stamp = bundle["time"].replace(":", "").replace("-", "")
    record_json = {
        "time": bundle["time"], "source": bundle["name"], "module1": ops.OpsParams(**bundle["params"]).changed(),
        "preset": bundle["preset"], "video_settings": bundle["settings"], "summary": summary,
        "model": MODEL_PROFILES[key].filename, "model_details": {"id": key, "name": MODEL_PROFILES[key].name,
                                                                "training": MODEL_PROFILES[key].training},
        "calibrated": bundle["calibrated"], "uncertain_threshold": bundle["threshold"],
        "sample_segments": bundle.get("segments"),
        "sample_agreement": segment_agreement(result, bundle["segments"]) if bundle.get("segments") else None,
    }
    with st.container(key="video_exports"):
        video_col, csv_col, json_col, snap_col = st.columns(4)
        video_col.download_button("Tải video kết quả", result.output.read_bytes(), f"wastelens-{key}-{stamp}.mp4",
                                  "video/mp4", icon=":material/movie:", width="stretch", on_click="ignore")
        csv_col.download_button("Tải CSV theo khung", records_csv(result, classes).encode("utf-8"),
                                f"wastelens-{key}-{stamp}-frames.csv", "text/csv", icon=":material/table:",
                                width="stretch", on_click="ignore")
        json_col.download_button("Tải JSON", json.dumps(record_json, ensure_ascii=False, indent=2).encode("utf-8"),
                                 f"wastelens-{key}-{stamp}.json", "application/json", icon=":material/data_object:",
                                 width="stretch", on_click="ignore")
        ok, png = cv2.imencode(".png", snapshot) if snapshot is not None else (False, None)
        snap_col.download_button("Tải snapshot", png.tobytes() if ok else b"",
                                 f"wastelens-{key}-{stamp}-t{record.time_s:.1f}s.png", "image/png", disabled=not ok,
                                 icon=":material/photo_camera:", width="stretch", on_click="ignore")


def main():
    """Page layout: header, input source tabs, then the image, video or webcam workflow."""
    st.set_page_config(page_title="WasteLens · Phân loại rác", page_icon="♻️", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(Path(__file__).with_name("styles.css"))
    init_state()
    render_header()
    mode = st.segmented_control("Nguồn đầu vào", ["Ảnh", "Video", "Webcam"], default="Ảnh", required=True,
                                key="input_mode")
    if mode == "Video":
        render_video()
        return
    if mode == "Webcam":
        render_webcam()
        return
    if "camera_state" in st.session_state:
        del st.session_state["camera_state"]
    image, processed, params, ops_ms = render_viewer()
    key, run, available = render_toolbar(image is not None)
    signature = None
    if image is not None and available:
        signature = request_signature(st.session_state.input_bytes, params, key,
                                      MODEL_PROFILES[key].checkpoint.stat().st_mtime_ns)
        if run:
            with st.spinner("Đang phân loại ảnh…"):
                run_inference(image, processed, params, ops_ms, key, signature)
            st.rerun()
    render_result(signature)
    st.html('<footer class="workbench-footer"><span><strong>WasteLens</strong> · Đồ án Xử lý Ảnh &amp; Video số</span>'
            '<span>TrashNet · 6 lớp</span></footer>')


main()
