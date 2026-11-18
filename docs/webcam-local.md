# Webcam local

Run from the repository root (`D:\XLAV\Lab_3\source`):

```powershell
.venv\Scripts\python.exe scripts/prepare_detector.py
.venv\Scripts\python.exe -m streamlit run app/app.py
```

Open http://localhost:8501, choose **Webcam**, then **Bật camera** and grant camera permission. The browser uses the camera of the device opening the page. Audio is never requested. **Dừng camera**, switching to the image tab, or leaving the page releases its video tracks.

Choose Baseline Lab 2, MLP cải tiến, or MobileNetV3 Large for recognition. Click a box or select **Vật đang xem** to inspect that object's six-class probability table. Module 1 presets process each detected crop before recognition; detection coordinates stay in the original frame. Camera images have no dataset Ground Truth.

The pipeline uses the official torchvision **SSDLite320 MobileNetV3 Large COCO** detector to locate objects, skips people, adds a small bounded crop margin, and runs the selected existing TrashNet classifier on each crop. The detector is pretrained; this does not change the training provenance of the three recognition models. Its COCO class is shown separately from the predicted waste class. A short IoU tracker associates boxes between analyzed frames; IDs can change after occlusion or substantial motion.

The preview uses the browser's live camera stream, requesting native high resolution and then the largest width/height ideals advertised by its track capabilities. Resolution is negotiated first; a second request asks for the highest advertised frame rate while keeping those dimensions fixed, so seeking more FPS cannot lower preview resolution. The actual mode is negotiated by the browser, driver and hardware; capability maxima are not a guarantee that every width/height/FPS combination is available. If renegotiation fails, the working stream is retained. The viewer follows the actual video aspect ratio, including portrait cameras. The quality line displays actual source dimensions and the track's reported FPS; it is not a measured playback FPS counter. Overlay backing resolution follows device pixel ratio, with redraws only when boxes, selection or display size change.

Only analysis frames are downscaled, preserving their aspect ratio and never upscaling smaller camera frames. Choose a longest side of 640, 960 or 1280 pixels in **Thiết lập stream** (default 1280). This does not reduce preview quality. JPEG frames travel over the Streamlit session WebSocket at a configurable 1–5 frames per second, with at most one inference request in flight. The default is 2 analyzed frames per second and at most 3 objects. Boxes expire visually after 2.5 seconds without an updated result. No frames are recorded to disk. Detector weights are downloaded once from PyTorch, hash checked, and stored in the ignored `models/` directory.

COCO is not a dedicated waste-detection dataset: cardboard and irregular trash may not be detected, while ordinary objects can still be assigned one of the six waste classes. TrashNet uses controlled backgrounds, so its held-out image accuracy is not webcam accuracy. Use a clear object and simple background for the demo. Localhost supports camera access; another device normally needs HTTPS.

This feature is being developed locally; it does not require deploying or pushing to GitHub.
