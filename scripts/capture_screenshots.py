"""Drive the running WasteLens app in Chrome and save one screenshot per feature for the report.

Start the app first (`run.bat` or `streamlit run app/app.py`), create the fake webcam clip with
`scripts/make_fake_camera.py`, then run this script from an environment that has Playwright
(`pip install playwright==1.49.1`; it uses the installed Google Chrome, so no browser download):

    python scripts/capture_screenshots.py docs/screenshots outputs/fake-camera.y4m
    python scripts/capture_screenshots.py docs/screenshots outputs/fake-camera.y4m --only webcam

The image tab uses the fixed test image plastic73.jpg so results are reproducible. The video tab
builds a sample from random test images, so its labels change between runs. The webcam tab reads
the fake camera clip; the camera picker is relabelled because Chrome names the fake device after
its local file path.
"""
import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

WIDTH, HEIGHT = 1440, 960
TEST_IMAGE = "plastic73.jpg"


class Shooter:
    """Small helpers around one Playwright page showing the Streamlit app."""

    def __init__(self, page, output: Path):
        self.page, self.output = page, output

    def idle(self, settle=0.8, timeout=240):
        """Wait until Streamlit has finished rerunning (no status widget, spinner or progress bar)."""
        time.sleep(settle)
        end = time.time() + timeout
        while time.time() < end:
            busy = self.page.locator('[data-testid="stStatusWidget"], [data-testid="stSpinner"], '
                                     '[data-testid="stProgress"]').count()
            if not busy:
                time.sleep(0.5)
                if not self.page.locator('[data-testid="stStatusWidget"]').count():
                    return
            time.sleep(0.3)
        print("  warning: the app was still busy after", timeout, "s")

    def full(self, name):
        """Whole-page screenshot: the app scrolls inside stMain, so grow the viewport to its height."""
        to_top = ("() => { const main = document.querySelector('[data-testid=\"stMain\"]');"
                  " main.scrollTop = 0; window.scrollTo(0, 0); return main.scrollHeight; }")
        height = self.page.evaluate(to_top)
        self.page.set_viewport_size({"width": WIDTH, "height": max(HEIGHT, height)})
        time.sleep(1.0)
        self.page.evaluate(to_top)
        time.sleep(0.3)
        self.save(name)
        self.page.set_viewport_size({"width": WIDTH, "height": HEIGHT})
        time.sleep(0.4)

    def save(self, name, **options):
        self.page.screenshot(path=self.output / f"{name}.png", **options)
        print("saved", name)

    def element(self, selector, name, pad=12):
        box = self.page.locator(selector).first.bounding_box()
        self.save(name, clip={"x": max(0, box["x"] - pad), "y": max(0, box["y"] - pad),
                              "width": box["width"] + 2 * pad, "height": box["height"] + 2 * pad})

    def segment(self, group, label):
        """Click one option of a Streamlit segmented control, found by its accessible group name."""
        self.page.locator(f'[role="radiogroup"][aria-label="{group}"] button').filter(has_text=label).first.click()
        self.idle()

    def button(self, name, exact=True):
        self.page.get_by_role("button", name=name, exact=exact).click()


def image_tab(shot: Shooter):
    page = shot.page
    shot.save("01-trang-chu")
    # A fixed held-out image chosen through the dataset browser, so the ground-truth card appears.
    shot.button("Chọn ảnh trong tập test")
    time.sleep(0.6)
    page.get_by_test_id("stSelectbox").last.click()
    page.keyboard.type(TEST_IMAGE)
    page.keyboard.press("Enter")
    shot.idle()
    shot.button("Dùng ảnh này")
    shot.idle()
    page.keyboard.press("Escape")
    shot.button("Phân loại")
    shot.idle(1.5)
    shot.full("02-anh-ket-qua")

    shot.button("Mô hình: MLP cải tiến")
    time.sleep(0.8)
    shot.save("03-chon-mo-hinh")
    page.keyboard.press("Escape")
    time.sleep(0.4)

    # Module 1: open the presets, apply "Mạnh", show original and processed side by side, re-run.
    shot.button("Xử lý ảnh: Gốc")
    time.sleep(0.6)
    shot.save("04-muc-xu-ly")
    shot.button("Mạnh")
    shot.idle()
    page.keyboard.press("Escape")
    time.sleep(0.3)
    shot.segment("Chế độ xem", "So sánh")
    shot.button("Phân loại lại")
    shot.idle(1.5)
    shot.full("05-module1-so-sanh")

    shot.button("Chỉnh sửa")
    shot.idle(1.2)
    shot.save("06-module1-tuy-chinh")
    page.keyboard.press("Escape")
    shot.idle()

    page.get_by_text("So sánh cả 3 mô hình trên ảnh này").click()
    shot.idle(2)
    shot.element(".st-key-result_card", "07-so-sanh-3-mo-hinh")

    shot.button("Thông tin mô hình")
    shot.idle(1.2)
    shot.save("08-thong-tin-mo-hinh")
    page.keyboard.press("Escape")
    shot.idle()


def video_tab(shot: Shooter):
    shot.segment("Nguồn đầu vào", "Video")
    shot.button("Tạo video mẫu từ tập test")
    shot.idle(2)
    shot.button("Xử lý video")
    shot.idle(3, timeout=600)
    shot.full("09-video-ket-qua")


def webcam_tab(shot: Shooter, model: str):
    page = shot.page
    shot.segment("Nguồn đầu vào", "Webcam")
    page.get_by_test_id("stSelectbox").first.click()
    page.get_by_role("option", name=model).click()
    shot.idle()
    shot.button("Bật camera")
    time.sleep(12)  # Let several analysed frames arrive so boxes and the probability table are filled.
    shot.idle(1)
    page.locator("option").filter(has_text=".y4m").evaluate_all(
        "els => els.forEach(e => e.textContent = 'Camera giả lập (video từ ảnh test)')")
    shot.full("10-webcam")


def main():
    parser = argparse.ArgumentParser(description="Capture WasteLens screenshots for the report.")
    parser.add_argument("output", type=Path)
    parser.add_argument("camera", type=Path, help="fake webcam clip from scripts/make_fake_camera.py")
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--only", nargs="*", choices=["image", "video", "webcam"])
    parser.add_argument("--webcam-model", default="MobileNetV3 Large",
                        help="classifier shown in the webcam screenshot")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parts = set(args.only or ["image", "video", "webcam"])
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={args.camera.resolve()}"])
        context = browser.new_context(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1.5,
                                      permissions=["camera"], locale="vi-VN")
        page = context.new_page()
        page.goto(args.url)
        page.get_by_text("WasteLens").first.wait_for(timeout=60000)
        shot = Shooter(page, args.output)
        shot.idle(2)
        if "image" in parts:
            image_tab(shot)
        if "video" in parts:
            video_tab(shot)
        if "webcam" in parts:
            webcam_tab(shot, args.webcam_model)
        browser.close()


if __name__ == "__main__":
    main()
