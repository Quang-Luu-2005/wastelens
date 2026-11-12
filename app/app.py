"""WasteLens Streamlit app, image mode: Module 1 (OpenCV operations) and Module 2 (classification).

Run from the repository root:  streamlit run app/app.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from wastelens import ops
from wastelens.classifier import DEFAULT_CHECKPOINT, Prediction, WasteClassifier

ROOT = Path(__file__).resolve().parents[1]
MAX_SIDE = 1024  # Larger uploads are downscaled so slow filters (bilateral) stay interactive.
LABELS_VI = {
    "cardboard": "Bìa carton", "glass": "Thủy tinh", "metal": "Kim loại", "paper": "Giấy",
    "plastic": "Nhựa", "trash": "Rác khác", "uncertain": "Không chắc",
}
BIN_VI = {
    "cardboard": "Thùng tái chế giấy", "paper": "Thùng tái chế giấy",
    "glass": "Thùng tái chế thủy tinh", "metal": "Thùng tái chế kim loại",
    "plastic": "Thùng tái chế nhựa", "trash": "Thùng rác thường",
}
# Slider keys; deleting them from session state restores the identity defaults.
PARAM_KEYS = [field for field in ops.OpsParams.__dataclass_fields__]


@st.cache_resource(show_spinner="Đang nạp model…")
def load_model() -> WasteClassifier:
    """Load the 10-MLP ensemble once per server process."""
    return WasteClassifier(DEFAULT_CHECKPOINT)


@st.cache_data
def sample_images() -> dict[str, Path]:
    """Two held-out test images per class, if the prepared dataset is present."""
    manifest, image_root = ROOT / "data/splits/trashnet.csv", ROOT / "data/trashnet"
    if not manifest.is_file() or not image_root.is_dir():
        return {}
    samples, counts = {}, {}
    for line in manifest.read_text(encoding="utf-8").splitlines()[1:]:
        path, label, _, split = line.split(",")[:4]
        if split == "test" and counts.get(label, 0) < 2 and (image_root / path).is_file():
            counts[label] = counts.get(label, 0) + 1
            samples[f"{LABELS_VI[label]} ({label}) · {Path(path).name}"] = image_root / path
    return samples


def to_bgr(image: Image.Image) -> np.ndarray:
    """PIL image -> BGR uint8, applying the EXIF orientation that phone photos rely on."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    longest = max(rgb.size)
    if longest > MAX_SIDE:
        rgb = rgb.resize((round(rgb.width * MAX_SIDE / longest), round(rgb.height * MAX_SIDE / longest)),
                         Image.Resampling.BILINEAR)
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def reset_params():
    for key in PARAM_KEYS:
        st.session_state.pop(key, None)


def module1_controls() -> ops.OpsParams:
    """Sidebar sliders for Module 1; ranges are chosen to stay meaningful on 512 x 384 photos."""
    sidebar = st.sidebar
    sidebar.header("Module 1 · Xử lý ảnh")
    sidebar.button("Đặt lại tất cả", on_click=reset_params, width="stretch")
    with sidebar.expander("Biến đổi màu", expanded=True):
        gamma = st.slider("Gamma", 0.2, 3.0, 1.0, 0.05, key="gamma", help=">1 làm sáng, <1 làm tối")
        contrast = st.slider("Tương phản (α)", 0.2, 3.0, 1.0, 0.05, key="contrast")
        brightness = st.slider("Độ sáng (β)", -100, 100, 0, 1, key="brightness")
        hue_shift = st.slider("Dịch hue (đơn vị OpenCV, 180 = một vòng)", -90, 90, 0, 1, key="hue_shift")
        saturation_scale = st.slider("Hệ số bão hòa", 0.0, 2.0, 1.0, 0.05, key="saturation_scale")
        value_scale = st.slider("Hệ số độ sáng HSV (V)", 0.0, 2.0, 1.0, 0.05, key="value_scale")
        clahe_clip = st.slider("CLAHE clip limit (0 = tắt)", 0.0, 8.0, 0.0, 0.5, key="clahe_clip")
        clahe_tiles = st.slider("CLAHE số ô lưới", 2, 16, 8, 1, key="clahe_tiles")
        grayscale = st.toggle("Ảnh xám", key="grayscale")
    with sidebar.expander("Biến đổi hình học"):
        angle = st.slider("Góc xoay (độ)", -180, 180, 0, 1, key="angle")
        scale = st.slider("Tỉ lệ phóng", 0.5, 2.0, 1.0, 0.05, key="scale")
        flip_horizontal = st.toggle("Lật ngang", key="flip_horizontal")
        flip_vertical = st.toggle("Lật dọc", key="flip_vertical")
        crop_keep = st.slider("Crop giữ lại phần trung tâm", 0.3, 1.0, 1.0, 0.05, key="crop_keep")
    with sidebar.expander("Làm mượt"):
        gaussian_sigma = st.slider("Gaussian σ (0 = tắt)", 0.0, 10.0, 0.0, 0.25, key="gaussian_sigma")
        median_kernel = st.slider("Median kernel (1 = tắt)", 1, 15, 1, 2, key="median_kernel")
        bilateral_diameter = st.slider("Bilateral đường kính d (0 = tắt)", 0, 15, 0, 1, key="bilateral_diameter",
                                       help="d lớn chạy chậm; d = 5 thường đủ")
        bilateral_sigma_color = st.slider("Bilateral σ màu", 10, 200, 75, 5, key="bilateral_sigma_color")
        bilateral_sigma_space = st.slider("Bilateral σ không gian", 10, 200, 75, 5, key="bilateral_sigma_space")
    with sidebar.expander("Nhiễu giả lập (để thử độ bền model)"):
        noise_sigma = st.slider("Nhiễu Gaussian σ", 0.0, 60.0, 0.0, 1.0, key="noise_sigma")
        salt_pepper = st.slider("Nhiễu muối tiêu (tỉ lệ điểm ảnh)", 0.0, 0.2, 0.0, 0.005, key="salt_pepper")
    return ops.OpsParams(
        gamma=gamma, contrast=contrast, brightness=brightness, hue_shift=hue_shift,
        saturation_scale=saturation_scale, value_scale=value_scale, clahe_clip=clahe_clip,
        clahe_tiles=clahe_tiles, grayscale=grayscale, angle=angle, scale=scale,
        flip_horizontal=flip_horizontal, flip_vertical=flip_vertical, crop_keep=crop_keep,
        gaussian_sigma=gaussian_sigma, median_kernel=median_kernel, bilateral_diameter=bilateral_diameter,
        bilateral_sigma_color=bilateral_sigma_color, bilateral_sigma_space=bilateral_sigma_space,
        noise_sigma=noise_sigma, salt_pepper=salt_pepper,
    )


def display_label(prediction: Prediction) -> str:
    return LABELS_VI[prediction.label]


def annotate(image: np.ndarray, prediction: Prediction) -> np.ndarray:
    """Draw the result on a copy of the image. ASCII text only: cv2.putText has no Vietnamese glyphs."""
    out = image.copy()
    text = (f"uncertain (closest: {prediction.top_class} {prediction.confidence:.0%})" if prediction.uncertain
            else f"{prediction.label} {prediction.confidence:.0%}")
    scale = max(0.6, out.shape[1] / 700)
    thickness = max(1, round(scale * 2))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    color = (0, 165, 255) if prediction.uncertain else (40, 160, 40)  # BGR: orange or green banner.
    cv2.rectangle(out, (0, 0), (width + 20, height + baseline + 20), color, -1)
    cv2.putText(out, text, (10, height + 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def probability_chart(prediction: Prediction, classes: list[str]):
    frame = pd.DataFrame({
        "Lớp": [f"{LABELS_VI[name]} ({name})" for name in classes],
        "Xác suất (%)": [100 * prediction.probabilities[name] for name in classes],
    })
    st.bar_chart(frame, x="Lớp", y="Xác suất (%)", horizontal=True, sort=False, height=260)


def main():
    st.set_page_config(page_title="WasteLens", page_icon="♻️", layout="wide")
    st.title("WasteLens · Phân loại rác 6 lớp")
    st.caption("Ensemble 10 MLP train từ đầu trên descriptor HOG + HSV + Lab và HSV · không dùng trọng số pretrained")

    if not DEFAULT_CHECKPOINT.is_file():
        st.error(f"Không tìm thấy model tại `{DEFAULT_CHECKPOINT}`. Chép `wastelens-ensemble.pt` vào thư mục `models/`.")
        st.stop()
    model = load_model()
    params = module1_controls()

    # ------------------------------------------------------------------ input
    samples = sample_images()
    sources = ["Tải ảnh lên"] + (["Ảnh mẫu (tập test TrashNet)"] if samples else [])
    source = st.segmented_control("Nguồn ảnh", sources, default=sources[0], required=True)
    if source == "Tải ảnh lên":
        upload = st.file_uploader("Chọn ảnh .jpg hoặc .png", type=["jpg", "jpeg", "png"])
        if upload is None:
            st.info("Tải một ảnh lên, hoặc chọn **Ảnh mẫu** để thử nhanh.")
            st.stop()
        name, image = upload.name, to_bgr(Image.open(upload))
    else:
        choice = st.selectbox("Ảnh mẫu", list(samples))
        name = samples[choice].name
        with Image.open(samples[choice]) as sample:
            image = to_bgr(sample)

    # --------------------------------------------------------------- Module 1
    started = time.perf_counter()
    processed = ops.apply(image, params)
    ops_ms = 1000 * (time.perf_counter() - started)
    left, right = st.columns(2)
    left.image(image, channels="BGR", caption=f"Ảnh gốc · {image.shape[1]}×{image.shape[0]}", width="stretch")
    right.image(processed, channels="BGR", width="stretch",
                caption=f"Sau xử lý · {processed.shape[1]}×{processed.shape[0]} · {ops_ms:.1f} ms")
    changed = params.changed()
    st.caption("Phép đang bật: " + (", ".join(f"`{k}={v}`" for k, v in changed.items()) if changed else "không có"))

    # --------------------------------------------------------------- Module 2
    st.header("Module 2 · Dự đoán")
    prediction = model.classify(processed)
    original = model.classify(image) if changed else prediction
    result, chart = st.columns([2, 3])
    with result:
        st.metric("Top-1", display_label(prediction),
                  help=f"Dưới ngưỡng {model.threshold:.0%} confidence thì hiển thị “Không chắc”")
        st.metric("Confidence (đã hiệu chỉnh)", f"{prediction.confidence:.1%}")
        if prediction.uncertain:
            st.warning(f"Gần nhất: **{LABELS_VI[prediction.top_class]}** ({prediction.confidence:.0%}). "
                       "Model không đủ chắc chắn; hãy chụp lại rõ hơn.")
        else:
            st.success(f"Gợi ý: **{BIN_VI[prediction.label]}**")
        # Total covers the whole pipeline: Module 1, descriptor extraction and the ensemble.
        timing = {"Module 1": ops_ms, "Đặc trưng": prediction.timings_ms["features"],
                  "Model": prediction.timings_ms["model"], "Tổng": ops_ms + prediction.timings_ms["total"]}
        st.caption("Thời gian xử lý (ms): " + " · ".join(f"{key} {value:.1f}" for key, value in timing.items()))
        if changed:
            st.caption(f"Trên ảnh gốc: {display_label(original)} {original.confidence:.0%}")
    with chart:
        probability_chart(prediction, model.classes)

    # --------------------------------------------------------- snapshot & export
    st.header("Snapshot & export")
    annotated = annotate(processed, prediction)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {
        "time": datetime.now().isoformat(timespec="seconds"), "source": name,
        "module1": changed, "prediction": {
            "label": prediction.label, "top_class": prediction.top_class, "confidence": prediction.confidence,
            "probabilities": prediction.probabilities, "uncertain_threshold": model.threshold,
        },
        "original_prediction": {"label": original.label, "confidence": original.confidence},
        "timings_ms": timing, "model": DEFAULT_CHECKPOINT.name,
    }
    preview, buttons = st.columns([3, 2])
    preview.image(annotated, channels="BGR", caption="Ảnh kết quả", width="stretch")
    ok, png = cv2.imencode(".png", annotated)
    buttons.download_button("Tải ảnh kết quả (PNG)", png.tobytes() if ok else b"", f"wastelens-{stamp}.png",
                            "image/png", width="stretch", disabled=not ok)
    buttons.download_button("Tải kết quả (JSON)", json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
                            f"wastelens-{stamp}.json", "application/json", width="stretch")
    ok, processed_png = cv2.imencode(".png", processed)
    buttons.download_button("Tải ảnh đã xử lý (không chú thích)", processed_png.tobytes() if ok else b"",
                            f"wastelens-{stamp}-processed.png", "image/png", width="stretch", disabled=not ok)


main()
