# WasteLens container deployment

`Dockerfile.vercel` packages the existing Streamlit app, three CPU classifiers and
the 377 labeled test images. Vercel's Container preset detects this file. No
training runs at build or request time.

## Build and run locally

```powershell
docker build -f Dockerfile.vercel -t wastelens-vercel:local .
docker run --rm -p 8502:80 wastelens-vercel:local
```

Open `http://localhost:8502`, select an image, then click **Phân loại**. Test model
selection, sample labels, Module 1, comparison, file upload and PNG/JSON downloads.

## Assets

Builds download the fixed assets from the GitHub `runtime-v1` release. The
`assets.json` file records SHA-256 and size; the downloader verifies hashes before
installing any checkpoint or extracting test images. The image archive is limited
to manifest test rows, and archive paths are checked before extraction.

TrashNet attribution: Gary Thung and Mindy Yang,
<https://github.com/garythung/trashnet>. Use the upstream license and dataset
description in `data/splits/trashnet-summary.json` and the main README.

## Vercel

Use the **Container** framework preset for the `wastelens` project with the
repository root as its Root Directory. Deploy a reviewed Git commit to Preview
first. The server listens on port 80 (or the configured `PORT`). This path uses
Vercel's Container Runtime beta, not the standard Python function builder.

The container uses CPU-only PyTorch and headless OpenCV. Upload size is limited to
4 MB in this container to stay within Vercel Function request limits. Local
Streamlit settings are unchanged. Instance memory and WebSocket duration are
subject to the project's Vercel plan. Sessions may reset on reconnect or scale-in;
download a result if you need to retain it. Uploaded user images are not saved to
the repository or release.

`WASTELENS_STATELESS_HOST=1` enables a small bundled file component: uploads travel
through the existing session WebSocket, preview images use inline data URLs and
downloads use browser Blob URLs. This avoids Streamlit's instance-local HTTP
upload/media store when Vercel routes HTTP requests to different instances. The
local app keeps its normal Streamlit file picker and download controls.

Never treat a successful build as inference verification: check the deployed
browser's WebSocket, upload request, model output and downloaded JSON before
claiming the deployment works. Deployment protection is retained by default.
