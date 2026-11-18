"""WasteLens image workbench, adapted from the supplied Stitch design.

Run from the repository root: streamlit run app/app.py
"""

import hashlib
import io
import json
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
from wastelens.workbench import PRESETS, PRESET_DESCRIPTIONS, describe_operations, request_signature

ROOT = Path(__file__).resolve().parents[1]
MAX_SIDE = 1024
LABELS_VI = {
    "cardboard": "Bìa carton", "glass": "Thủy tinh", "metal": "Kim loại", "paper": "Giấy",
    "plastic": "Nhựa", "trash": "Rác khác", "uncertain": "Không chắc",
}
BIN_VI = {
    "cardboard": "Thùng tái chế giấy", "paper": "Thùng tái chế giấy",
    "glass": "Thùng tái chế thủy tinh", "metal": "Thùng tái chế kim loại",
    "plastic": "Thùng tái chế nhựa", "trash": "Thùng rác thường",
}
MODEL_NAMES = {"baseline": "Baseline Lab 2", "improved": "MLP cải tiến", "mobilenet": "MobileNetV3 Large"}
CLASS_COLORS = {
    "cardboard": "#d97706", "glass": "#0284c7", "metal": "#64748b",
    "paper": "#6366f1", "plastic": "#059669", "trash": "#e11d48",
}
UPLOAD_ERROR = "Không đọc được ảnh. Vui lòng chọn lại tệp JPG hoặc PNG hợp lệ."


@st.cache_resource(show_spinner="Đang nạp mô hình…")
def load_model(checkpoint: str, modified_ns: int) -> WasteClassifier:
    """Replacing a checkpoint invalidates its cached classifier."""
    return WasteClassifier(Path(checkpoint))


def profile_model(key: str) -> WasteClassifier:
    path = MODEL_PROFILES[key].checkpoint
    return load_model(str(path), path.stat().st_mtime_ns)


@st.cache_data
def sample_images() -> dict[str, Path]:
    manifest, image_root = ROOT / "data/splits/trashnet.csv", ROOT / "data/trashnet"
    if not manifest.is_file() or not image_root.is_dir():
        return {}
    samples = {}
    for line in manifest.read_text(encoding="utf-8").splitlines()[1:]:
        path, label, _, split = line.split(",")[:4]
        if split == "test" and label not in samples and (image_root / path).is_file():
            samples[label] = image_root / path
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
    with Image.open(io.BytesIO(data)) as image:
        original_size = ImageOps.exif_transpose(image).size
        return to_bgr(image), original_size


def init_state():
    defaults = {"input_bytes": None, "input_name": "", "sample_label": None, "upload_revision": 0,
                "last_upload": None, "ops_params": asdict(ops.OpsParams()), "processing_level": "Gốc",
                "result": None, "image_view": "Ảnh gốc"}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_image():
    st.session_state.update(input_bytes=None, input_name="", sample_label=None,
                            upload_revision=st.session_state.upload_revision + 1, result=None)


def choose_sample(label: str, path: Path):
    st.session_state.update(input_bytes=path.read_bytes(), input_name=path.name, sample_label=label,
                            upload_revision=st.session_state.upload_revision + 1)


def accept_upload(widget_key: str):
    file = st.session_state.get(widget_key)
    if file is not None:
        data = file.getvalue()
        st.session_state.update(input_bytes=data, input_name=file.name, sample_label=None,
                                last_upload=(hashlib.sha256(data).hexdigest(), file.name, widget_key))


def file_picker():
    key = f"upload_{st.session_state.upload_revision}"
    upload = st.file_uploader("Chọn ảnh JPG hoặc PNG", type=["jpg", "jpeg", "png"], key=key,
                              on_change=accept_upload, args=(key,), label_visibility="collapsed")
    if upload is not None:
        data = upload.getvalue()
        marker = (hashlib.sha256(data).hexdigest(), upload.name, key)
        # Also supports integrations that supply the file directly rather than via a callback.
        if marker != st.session_state.last_upload:
            st.session_state.update(input_bytes=data, input_name=upload.name, sample_label=None, last_upload=marker)


def choose_preset(name: str):
    st.session_state.ops_params = asdict(PRESETS[name])
    st.session_state.processing_level = name
    if st.session_state.get("image_view") != "So sánh":
        st.session_state.image_view = "Ảnh gốc" if name == "Gốc" else "Sau xử lý"


def prepare_editor():
    for key, value in st.session_state.ops_params.items():
        st.session_state[f"edit_{key}"] = value


def reset_editor():
    for key, value in asdict(ops.OpsParams()).items():
        st.session_state[f"edit_{key}"] = value


def commit_editor():
    st.session_state.ops_params = {
        key: st.session_state.get(f"edit_{key}", value) for key, value in asdict(ops.OpsParams()).items()
    }
    st.session_state.processing_level = "Tùy chỉnh"
    if st.session_state.get("image_view") != "So sánh":
        st.session_state.image_view = "Sau xử lý"


@st.dialog("Tùy chỉnh xử lý ảnh", width="large")
def custom_processing():
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
    st.markdown("1. **Chọn ảnh** hoặc thử một ảnh mẫu.\n"
                "2. Chọn **mô hình** và **mức xử lý ảnh** bên dưới vùng xem.\n"
                "3. Bấm **Phân loại** để xem kết quả và xác suất sáu lớp.\n"
                "4. Tải PNG/JSON hoặc so sánh ba mô hình trên cùng ảnh.")
    st.info("Mức xử lý là preset bộ lọc ảnh. Mức mạnh hơn không đảm bảo phân loại tốt hơn.")
    st.caption("Ảnh lớn được thu về tối đa 1.024 px ở cạnh dài. Chế độ video và webcam chưa được triển khai.")


@st.dialog("Thông tin mô hình", width="large")
def show_model_info():
    for key, profile in MODEL_PROFILES.items():
        st.subheader(MODEL_NAMES[key])
        st.write(profile.description)
        st.caption(profile.training)
    st.info("MLP cải tiến hiệu chỉnh xác suất bằng temperature scaling T=2, ngưỡng không chắc 52,58%. "
            "Baseline và MobileNet dùng softmax chưa hiệu chỉnh.")
    st.caption("Suy luận hiện tại chạy PyTorch trên CPU. Dataset: TrashNet, sáu lớp.")


def render_header():
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
                before.image(image, channels="BGR", caption="Ảnh gốc", width="stretch")
                after.image(processed, channels="BGR", caption="Sau xử lý", width="stretch")
            else:
                shown = processed if view == "Sau xử lý" else image
                st.image(shown, channels="BGR", width="stretch",
                         alt="Ảnh sau xử lý" if view == "Sau xử lý" else "Ảnh đầu vào")
            width, height = original_size
            metadata = (f'{escape(st.session_state.input_name)} <span>· {width} × {height} px'
                        f' · {len(st.session_state.input_bytes) / 1024:.1f} KB</span>')
            if max(original_size) > MAX_SIDE:
                metadata += f' <span>· Xử lý ở {image.shape[1]} × {image.shape[0]} px</span>'
            st.html(f'<div class="file-metadata">{metadata}</div>')
        samples = sample_images()
        if samples:
            with st.container(key="sample_strip"):
                st.caption("THỬ ẢNH MẪU")
                columns = st.columns(len(samples))
                for column, (label, path) in zip(columns, samples.items(), strict=True):
                    column.button(LABELS_VI[label], key=f"sample_{label}", on_click=choose_sample,
                                  args=(label, path), width="stretch",
                                  type="primary" if st.session_state.sample_label == label else "secondary")
    return image, processed, params, ops_ms


def file_picker_empty():
    # A separate widget key avoids two copies of the same uploader in the initial card.
    key = f"empty_upload_{st.session_state.upload_revision}"
    upload = st.file_uploader("Thả ảnh vào đây hoặc bấm để chọn", type=["jpg", "jpeg", "png"],
                              key=key, on_change=accept_upload, args=(key,))
    if upload is not None:
        marker = (hashlib.sha256(upload.getvalue()).hexdigest(), upload.name, key)
        if marker != st.session_state.last_upload:
            st.session_state.update(input_bytes=upload.getvalue(), input_name=upload.name, sample_label=None,
                                    last_upload=marker)
            st.rerun()


def render_toolbar(has_image: bool):
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
                st.caption("Video và webcam chưa được triển khai.")
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


def annotate(image: np.ndarray, prediction: Prediction) -> np.ndarray:
    """OpenCV's built-in font uses ASCII; metadata in JSON retains Vietnamese labels."""
    out = image.copy()
    text = (f"uncertain (closest: {prediction.top_class} {prediction.confidence:.0%})" if prediction.uncertain
            else f"{prediction.label} {prediction.confidence:.0%}")
    scale = max(0.4, min(out.shape[1] / max(len(text) * 12, 1), out.shape[0] / 60, 1.0))
    thickness = max(1, round(scale * 2))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(out, (0, 0), (width + 16, height + baseline + 16),
                  (0, 165, 255) if prediction.uncertain else (95, 104, 0), -1)
    cv2.putText(out, text, (8, height + 8), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def run_inference(image, processed, params, ops_ms, model_key, signature):
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
    rows = []
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
        rows.append({"Mô hình": MODEL_NAMES[key], "Kết quả": LABELS_VI[cached["label"]],
                     "Lớp cao nhất": LABELS_VI[cached["top_class"]], "Confidence (%)": round(cached["confidence"] * 100, 2),
                     "Confidence": "Đã hiệu chỉnh" if cached["calibrated"] else "Softmax",
                     "Thời gian (ms)": round(cached["timings_ms"]["total"], 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Cùng một ảnh sau Module 1. Không chọn mô hình thắng bằng confidence vì các mô hình "
               "chưa được hiệu chỉnh giống nhau.")


def render_result(signature: str | None):
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
        suggestion = ("Hãy thử một ảnh rõ hơn." if prediction.uncertain else BIN_VI[prediction.label])
        st.html(
            f'<div class="prediction-summary {"uncertain" if prediction.uncertain else ""}">'
            f'<div class="prediction-symbol" style="background:{CLASS_COLORS[prediction.top_class]}">♻</div>'
            f'<div class="prediction-copy"><div>KẾT QUẢ DỰ ĐOÁN: <strong>{label}</strong></div>'
            f'<p>Mô hình: <strong>{MODEL_NAMES[key]}</strong> · Suy luận: {prediction.timings_ms["total"]:.1f} ms</p>'
            f'<p>Gợi ý: {suggestion}</p></div><div class="confidence"><strong>{prediction.confidence:.1%}</strong>'
            f'<span>{confidence_kind}</span></div></div>')
        if prediction.uncertain:
            st.caption(f"Lớp gần nhất: {LABELS_VI[prediction.top_class]} · Ngưỡng không chắc {bundle['threshold']:.1%}.")
        if ops.OpsParams(**bundle["params"]).changed():
            original = bundle["original"]
            st.caption(f"Trên ảnh gốc: {LABELS_VI[original.label]} {original.confidence:.1%} "
                       f"→ Sau xử lý: {label} {prediction.confidence:.1%}")
        with st.expander("Chi tiết thời gian xử lý"):
            st.caption(f"Module 1: {bundle['ops_ms']:.1f} ms · Chuẩn bị đầu vào: "
                       f"{prediction.timings_ms['features']:.1f} ms · Model: {prediction.timings_ms['model']:.1f} ms · "
                       f"Tổng: {bundle['ops_ms'] + prediction.timings_ms['total']:.1f} ms")
        probability_table(prediction)
        compare = st.toggle("So sánh cả 3 mô hình trên ảnh này", key="compare_models")
        if compare:
            with st.spinner("Đang so sánh mô hình…"):
                compare_models(bundle)
        render_exports(bundle, compare)


def render_exports(bundle, include_comparisons: bool):
    prediction, original = bundle["prediction"], bundle["original"]
    key, profile = bundle["model_key"], MODEL_PROFILES[bundle["model_key"]]
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
        "comparisons": bundle["comparisons"] if include_comparisons else {},
    }
    stamp = bundle["time"].replace(":", "").replace("-", "")
    with st.container(key="export_actions"):
        image_col, json_col, clean_col = st.columns(3)
        ok, png = cv2.imencode(".png", annotate(bundle["processed"], prediction))
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


def main():
    st.set_page_config(page_title="WasteLens · Phân loại rác", page_icon="♻️", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(Path(__file__).with_name("styles.css"))
    init_state()
    render_header()
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
