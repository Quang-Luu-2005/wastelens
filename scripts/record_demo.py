"""Record a captioned demo video of the running WasteLens app, without a screen recorder.

Three steps, because Playwright/edge-tts and PyAV usually live in different environments:

    # 0. (optional) Synthesize the narration in scripts/demo_narration.json with Edge TTS (needs internet).
    python scripts/record_demo.py voice outputs/demo-voice
    # 1. Drive the app in Chrome and capture frames through the DevTools screencast; each scene lasts
    #    at least as long as its narration.
    python scripts/record_demo.py record outputs/demo-frames --voice outputs/demo-voice
    # 2. Turn the frames into an H.264 MP4 at a constant frame rate and mix in the narration (needs PyAV).
    .venv\\Scripts\\python.exe scripts/record_demo.py assemble outputs/demo-frames outputs/wastelens-demo.mp4 --voice outputs/demo-voice

Start the app first (run.bat) and create the fake webcam clip with scripts/make_fake_camera.py.
Each scene shows a caption bar at the bottom of the page; a drawn cursor glides to every element
before it is clicked, so the viewer can follow what happens. The screencast only sends a frame when
the page changes, so the assembler repeats the last frame to fill pauses.
"""
import argparse
import base64
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1440, 900
TEST_IMAGE = "plastic73.jpg"
DETECT_VIDEO = ROOT / "outputs/test-videos/pexels-3585308-cans-pile.mp4"
NARRATION = Path(__file__).with_name("demo_narration.json")
MP3_BITRATE = 48000  # Edge TTS "audio-24khz-48kbitrate-mono-mp3": duration = bytes * 8 / bitrate.

# Injected once per page: a caption bar, a full-screen title card and a cursor that follows the mouse.
OVERLAY_JS = """
(() => {
  if (window.__demo) return;
  const css = document.createElement('style');
  css.textContent = `
    #demo-caption{position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:2147483646;
      max-width:80%;padding:12px 26px;border-radius:12px;background:rgba(15,23,42,.88);color:#fff;
      font:600 22px/1.35 "Segoe UI",Arial,sans-serif;text-align:center;box-shadow:0 6px 24px rgba(0,0,0,.25);
      transition:opacity .35s;opacity:0;pointer-events:none}
    #demo-caption small{display:block;font-weight:400;font-size:17px;opacity:.85;margin-top:2px}
    #demo-card{position:fixed;inset:0;z-index:2147483647;display:flex;flex-direction:column;align-items:center;
      justify-content:center;background:linear-gradient(135deg,#0f3d3e,#14532d);color:#fff;text-align:center;
      font-family:"Segoe UI",Arial,sans-serif;transition:opacity .5s;opacity:0;pointer-events:none}
    #demo-card h1{font-size:54px;margin:0 0 12px} #demo-card p{font-size:24px;margin:6px;opacity:.92}
    #demo-cursor{position:fixed;z-index:2147483647;width:22px;height:22px;margin:-4px 0 0 -3px;pointer-events:none;
      transition:transform .12s}
    #demo-cursor.down{transform:scale(.8)}`;
  document.head.appendChild(css);
  const caption = Object.assign(document.createElement('div'), {id: 'demo-caption'});
  const card = Object.assign(document.createElement('div'), {id: 'demo-card'});
  const cursor = Object.assign(document.createElement('div'), {id: 'demo-cursor'});
  cursor.innerHTML = '<svg viewBox="0 0 24 24" width="22" height="22"><path d="M3 2l7 19 2.6-7.4L20 11z" ' +
    'fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  document.body.append(caption, card, cursor);
  addEventListener('mousemove', e => { cursor.style.left = e.clientX + 'px'; cursor.style.top = e.clientY + 'px'; }, true);
  addEventListener('mousedown', () => cursor.classList.add('down'), true);
  addEventListener('mouseup', () => cursor.classList.remove('down'), true);
  // Chrome names the fake webcam after its local file path; show a neutral name instead.
  const rename = () => document.querySelectorAll('option').forEach(o => {
    if (o.textContent.includes('.y4m')) o.textContent = 'Camera giả lập (video từ ảnh test)'; });
  new MutationObserver(rename).observe(document.body, {childList: true, subtree: true, characterData: true});
  window.__demo = {
    caption(title, detail) {
      caption.innerHTML = title ? title + (detail ? '<small>' + detail + '</small>' : '') : '';
      caption.style.opacity = title ? 1 : 0;
    },
    card(html) {
      card.innerHTML = html || ''; card.style.opacity = html ? 1 : 0; cursor.style.opacity = html ? 0 : 1;
    },
  };
})();
"""


RENAME_JS = """
(() => {
  // The camera component renders inside a shadow root, out of reach of document.querySelectorAll
  // and MutationObserver on the document, so walk every shadow root a few times per second.
  const walk = root => {
    root.querySelectorAll('option').forEach(o => {
      if (o.textContent.includes('.y4m')) o.textContent = 'Camera giả lập (video từ ảnh test)'; });
    root.querySelectorAll('*').forEach(e => { if (e.shadowRoot) walk(e.shadowRoot); });
  };
  setInterval(() => walk(document), 150);
})();
"""


class Recorder:
    """Drives one page and stores every screencast frame with its timestamp."""

    def __init__(self, page, frames: Path, durations: dict | None = None):
        self.page, self.frames = page, frames
        self.index, self.log = 0, []
        # Narration: seconds per line, when each line starts, and when the current line ends.
        self.durations, self.spoken, self.deadline = durations or {}, [], 0.0
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._frame)

    def _frame(self, event):
        path = self.frames / f"{self.index:06d}.jpg"
        path.write_bytes(base64.b64decode(event["data"]))
        self.log.append({"file": path.name, "t": event["metadata"]["timestamp"]})
        self.index += 1
        self.cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

    def start(self):
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92,
                                               "maxWidth": WIDTH, "maxHeight": HEIGHT, "everyNthFrame": 1})

    def stop(self):
        self.until_quiet()
        self.wait(0.5)
        self.cdp.send("Page.stopScreencast")
        (self.frames / "frames.json").write_text(json.dumps(self.log), encoding="utf-8")
        (self.frames / "narration.json").write_text(json.dumps(self.spoken), encoding="utf-8")
        print(f"{len(self.log)} frames, {len(self.spoken)} narration lines")

    # --- pacing -----------------------------------------------------------------------------
    def wait(self, seconds):
        """Sleep while still letting Playwright deliver screencast events (time.sleep would block them)."""
        self.page.wait_for_timeout(seconds * 1000)

    def idle(self, settle=0.8, timeout=300):
        """Wait until Streamlit has finished rerunning."""
        self.wait(settle)
        end = time.time() + timeout
        while time.time() < end:
            busy = self.page.locator('[data-testid="stStatusWidget"], [data-testid="stSpinner"], '
                                     '[data-testid="stProgress"]').count()
            if not busy:
                self.wait(0.5)
                if not self.page.locator('[data-testid="stStatusWidget"]').count():
                    return
            self.wait(0.3)
        print("  warning: still busy after", timeout, "s")

    def until_quiet(self):
        """Wait until the narration line that is playing has finished."""
        remaining = self.deadline - time.time()
        if remaining > 0:
            self.wait(remaining)

    def say(self, key):
        """Start a narration line now (after the previous one ends) and note its start time."""
        if not key or key not in self.durations:
            return
        self.until_quiet()
        start = time.time()
        self.spoken.append({"key": key, "t": start})
        self.deadline = start + self.durations[key] + 0.5

    # --- overlay ----------------------------------------------------------------------------
    def overlay(self):
        self.page.evaluate(OVERLAY_JS)

    def caption(self, title, detail="", hold=0.0, say=None):
        self.say(say)
        self.overlay()
        self.page.evaluate("([t, d]) => window.__demo.caption(t, d)", [title, detail])
        self.wait(hold)

    def card(self, html, hold, say=None):
        self.say(say)
        self.overlay()
        self.page.evaluate("h => window.__demo.card(h)", html)
        self.wait(hold)
        self.until_quiet()
        self.page.evaluate("() => window.__demo.card('')")
        self.wait(0.6)

    # --- interaction ------------------------------------------------------------------------
    def point(self, locator):
        """Scroll the element into view and glide the drawn cursor onto its centre."""
        locator.evaluate("e => e.scrollIntoView({behavior: 'smooth', block: 'center'})")
        self.wait(0.7)
        box = locator.bounding_box()
        self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=25)
        self.wait(0.35)

    def click(self, locator, after=0.6):
        self.point(locator)
        locator.click()
        self.wait(after)

    def button(self, name, exact=True, after=0.6):
        self.click(self.page.get_by_role("button", name=name, exact=exact).first, after)

    def segment(self, group, label):
        self.click(self.page.locator(f'[role="radiogroup"][aria-label="{group}"] button').filter(has_text=label).first)
        self.idle()

    def show(self, locator, hold):
        """Scroll an element to the middle of the screen and keep it there for `hold` seconds."""
        locator.first.evaluate("e => e.scrollIntoView({behavior: 'smooth', block: 'center'})")
        self.wait(hold)

    def scroll_main(self, top, hold):
        self.page.evaluate("y => document.querySelector('[data-testid=\"stMain\"]').scrollTo({top: y, behavior: 'smooth'})",
                           top)
        self.wait(hold)


def scenes(r: Recorder, student: str, student_id: str):
    page = r.page
    r.card(f"<h1>WasteLens</h1><p>Phân loại rác TrashNet 6 lớp · Xử lý ảnh và video</p>"
           f"<p style='margin-top:28px'>Lab 03 · Xử lý ảnh và video số</p><p>{student} · {student_id}</p>", 5, say="intro")

    # Image mode and Module 2.
    r.caption("1. Chế độ ảnh", "Chọn một ảnh trong tập Test (hoặc tải ảnh JPG/PNG lên)", 2.5, say="image")
    r.button("Chọn ảnh trong tập test")
    r.click(page.get_by_test_id("stSelectbox").last)
    page.keyboard.type(TEST_IMAGE, delay=60)
    page.keyboard.press("Enter")
    r.idle()
    r.button("Dùng ảnh này")
    r.idle()
    page.keyboard.press("Escape")
    r.caption("2. Module 2: phân loại", "Nhãn Top-1, confidence (%), thời gian xử lý và xác suất của 6 lớp", say="classify")
    r.button("Phân loại")
    r.idle(1.5)
    r.show(page.locator(".st-key-result_card"), 5)
    r.caption("Ba mô hình để chọn", "Baseline Lab02 · MLP cải tiến (mặc định) · MobileNetV3 pretrained", say="models")
    r.button("Mô hình: MLP cải tiến")
    r.wait(3)
    page.keyboard.press("Escape")
    r.wait(0.5)

    # Module 1.
    r.caption("3. Module 1: xử lý ảnh OpenCV", "Chọn nhanh một mức xử lý: Gốc, Nhẹ, Vừa, Mạnh", say="module1")
    r.button("Xử lý ảnh: Gốc")
    r.wait(1.5)
    r.button("Mạnh")
    r.idle()
    page.keyboard.press("Escape")
    r.caption("Ảnh gốc và ảnh sau xử lý", "Chế độ So sánh: dự đoán trên cả hai ảnh", say="compare")
    r.segment("Chế độ xem", "So sánh")
    r.button("Phân loại lại")
    r.idle(1.5)
    r.scroll_main(0, 1)
    r.wait(4)
    r.caption("Tùy chỉnh từng tham số", "Biến đổi màu · biến đổi hình học · làm mượt · nhiễu thử", say="custom")
    r.button("Chỉnh sửa")
    r.idle(1.2)
    sliders = page.get_by_role("slider")
    if sliders.count():
        # The slider track sits on top of the thumb, so focus the thumb and move it with the keyboard.
        r.point(sliders.nth(1))
        sliders.nth(1).focus()
        for _ in range(6):
            page.keyboard.press("ArrowRight")
            r.wait(0.25)
    r.wait(2)
    for tab in ("Hình học", "Làm mượt"):
        tabs = page.get_by_role("tab").filter(has_text=tab)
        if tabs.count():
            r.click(tabs.first, after=2)
    page.keyboard.press("Escape")
    r.idle()

    # Compare the three models and export.
    r.caption("So sánh cả ba mô hình", "Cùng một ảnh sau Module 1, đối chiếu với nhãn thật", say="three")
    r.click(page.get_by_text("So sánh cả 3 mô hình trên ảnh này"))
    r.idle(2)
    r.show(page.locator(".st-key-result_card"), 5)
    r.caption("Snapshot & export", "Tải ảnh kết quả có chú thích, ảnh sau xử lý và file JSON", say="export")
    downloads = page.get_by_test_id("stDownloadButton")
    for i in range(min(downloads.count(), 3)):
        r.point(downloads.nth(i))
        r.wait(0.8)
    r.wait(2)

    # Video mode: whole-frame classification on a labelled sample, then per-object detection on a real clip.
    r.caption("4. Chế độ video", "Video mẫu ghép từ 6 ảnh Test; mỗi N khung phân loại một lần, làm mượt theo thời gian", say="video")
    r.segment("Nguồn đầu vào", "Video")
    r.button("Tạo video mẫu từ tập test")
    r.idle(2)
    r.button("Xử lý video")
    r.idle(3, timeout=600)
    r.scroll_main(0, 3)
    charts = page.locator('[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"]')
    if charts.count():
        r.caption("Xác suất 6 lớp theo thời gian", "Video có nhãn, CSV theo khung, JSON tóm tắt và snapshot đều tải được", say="chart")
        r.show(charts, 5)
    if DETECT_VIDEO.is_file():
        r.caption("Phát hiện từng vật trong video thật", "SSDLite tìm vật, theo dõi ID, phân loại từng vùng cắt", say="detect")
        r.scroll_main(0, 1)
        page.locator('input[type="file"]').first.set_input_files(str(DETECT_VIDEO))
        r.idle(3)
        r.click(page.get_by_test_id("stSelectbox").first)
        page.get_by_role("option", name="MobileNetV3 Large").click()
        r.idle()
        r.click(page.get_by_text("Phát hiện từng vật (SSDLite)"))
        r.idle()
        # After the first run the button reads "Xử lý lại video".
        r.click(page.get_by_role("button", name=re.compile(r"^Xử lý( lại)? video$")).first)
        r.idle(3, timeout=600)
        r.scroll_main(0, 4)
        tables = page.get_by_test_id("stDataFrame")
        if tables.count():
            r.caption("Bảng các vật được theo dõi", "Nhãn rác chiếm ưu thế của từng vật, thời điểm xuất hiện", say="tracks")
            r.show(tables, 5)

    # Webcam (bonus).
    r.caption("5. Webcam thời gian thực (bonus)", "Camera giả lập phát ảnh Test; phát hiện, theo dõi và phân loại từng vật", say="webcam")
    r.segment("Nguồn đầu vào", "Webcam")
    r.click(page.get_by_test_id("stSelectbox").first)
    page.get_by_role("option", name="MobileNetV3 Large").click()
    r.idle()
    r.button("Bật camera")
    r.wait(10)
    r.caption("Xác suất 6 lớp của vật đang chọn", "Nhấp vào khung của một vật để xem bảng xác suất của vật đó", say="probs")
    r.scroll_main(700, 6)
    r.caption("")
    r.card("<h1>WasteLens</h1><p>Mã nguồn: github.com/Quang-Luu-2005/wastelens</p>"
           f"<p style='margin-top:28px'>{student} · {student_id}</p>", 4, say="outro")


def record(args):
    from playwright.sync_api import sync_playwright
    args.frames.mkdir(parents=True, exist_ok=True)
    for old in args.frames.glob("*.jpg"):
        old.unlink()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={args.camera.resolve()}"])
        context = browser.new_context(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1,
                                      permissions=["camera"], locale="vi-VN", accept_downloads=True)
        # Runs in every frame, including the camera component's iframe, before its own scripts.
        context.add_init_script(RENAME_JS)
        page = context.new_page()
        page.goto(args.url)
        page.get_by_text("WasteLens").first.wait_for(timeout=60000)
        durations = {}
        if args.voice:
            durations = json.loads((args.voice / "voice.json").read_text(encoding="utf-8"))["durations"]
        recorder = Recorder(page, args.frames, durations)
        recorder.idle(2)
        recorder.overlay()
        recorder.start()
        try:
            scenes(recorder, args.student, args.student_id)
        except Exception:
            page.screenshot(path=str(args.frames / "error.png"), full_page=True)
            raise
        finally:
            recorder.stop()
            browser.close()


def voice(args):
    """Synthesize one MP3 per narration line with Edge TTS and store the durations."""
    import asyncio

    import edge_tts
    script = json.loads(NARRATION.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    async def synthesize(key, text):
        audio = b""
        async for chunk in edge_tts.Communicate(text, script["voice"], rate=script.get("rate", "+0%")).stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
        (args.output / f"{key}.mp3").write_bytes(audio)
        return len(audio) * 8 / MP3_BITRATE

    durations = {}
    for key, text in script["lines"].items():
        for attempt in range(4):
            try:
                durations[key] = asyncio.run(synthesize(key, text))
                break
            except edge_tts.exceptions.NoAudioReceived:
                # The service drops a request now and then; wait a little and ask again.
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        print(f"{key:<10} {durations[key]:5.1f} s")
    (args.output / "voice.json").write_text(json.dumps({"voice": script["voice"], "durations": durations}, indent=2),
                                            encoding="utf-8")
    print(f"total {sum(durations.values()):.1f} s")


def narration_track(frames: Path, voice_dir: Path, start: float, rate: int = 48000):
    """Place every spoken line at its recorded start time on one stereo float track."""
    import av
    import numpy as np
    spoken = json.loads((frames / "narration.json").read_text(encoding="utf-8"))
    pieces = []
    for line in spoken:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        chunks = []
        with av.open(str(voice_dir / f"{line['key']}.mp3")) as container:
            for frame in container.decode(audio=0):
                chunks += [f.to_ndarray() for f in resampler.resample(frame)]
        chunks += [f.to_ndarray() for f in resampler.resample(None)]
        pieces.append((int((line["t"] - start) * rate), np.concatenate(chunks, axis=1)))
    length = max(offset + samples.shape[1] for offset, samples in pieces) if pieces else 0
    track = np.zeros((2, length), np.float32)
    for offset, samples in pieces:
        track[:, offset:offset + samples.shape[1]] += samples
    return np.clip(track, -1, 1), rate


def assemble(args):
    import av
    import numpy as np
    from PIL import Image
    log = json.loads((args.frames / "frames.json").read_text(encoding="utf-8"))
    if not log:
        raise SystemExit("No frames recorded")
    start, end = log[0]["t"], log[-1]["t"] + 1.0
    track, rate = narration_track(args.frames, args.voice, start) if args.voice else (None, 48000)
    if track is not None:
        end = max(end, start + track.shape[1] / rate + 0.5)
    with av.open(str(args.output), "w") as container:
        stream = container.add_stream("libx264", rate=args.fps)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        audio = None
        if track is not None:
            # Every stream must exist before the first packet is written (the header is written then).
            audio = container.add_stream("aac", rate=rate)
            audio.layout = "stereo"
            audio.bit_rate = 128000
        current, cursor, frame = None, 0, None
        for tick in range(int((end - start) * args.fps)):
            moment = start + tick / args.fps
            while cursor < len(log) and log[cursor]["t"] <= moment:
                current, cursor = log[cursor]["file"], cursor + 1
                image = Image.open(args.frames / current).convert("RGB")
                if image.size != (WIDTH, HEIGHT):
                    image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
                frame = np.asarray(image)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio is not None:
            for offset in range(0, track.shape[1], 1024):
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(track[:, offset:offset + 1024]),
                                                   format="fltp", layout="stereo")
                frame.sample_rate, frame.pts = rate, offset
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    print(f"{args.output}: {end - start:.1f} s at {args.fps} fps" + (", with narration" if track is not None else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="drive the app and capture frames")
    rec.add_argument("frames", type=Path)
    rec.add_argument("--camera", type=Path, default=ROOT / "outputs/fake-camera.y4m")
    rec.add_argument("--url", default="http://localhost:8501")
    rec.add_argument("--student", default="Lưu Huy Minh Quang")
    rec.add_argument("--student-id", default="23127016")
    rec.add_argument("--voice", type=Path, help="folder from the voice step; scenes wait for each line")
    vo = sub.add_parser("voice", help="synthesize the narration with Edge TTS")
    vo.add_argument("output", type=Path)
    asm = sub.add_parser("assemble", help="build an MP4 from captured frames")
    asm.add_argument("frames", type=Path)
    asm.add_argument("output", type=Path)
    asm.add_argument("--fps", type=int, default=25)
    asm.add_argument("--voice", type=Path, help="folder from the voice step, to mix the narration in")
    args = parser.parse_args()
    {"record": record, "voice": voice, "assemble": assemble}[args.command](args)


if __name__ == "__main__":
    main()
