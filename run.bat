@echo off
rem WasteLens one-command setup and launch for Windows (CPU only, Python 3.12).
rem   run.bat          install what is missing, then open the app in the browser
rem   run.bat setup    install only (virtual environment, packages, dataset, webcam detector)
rem   run.bat test     install what is missing, then run the unit tests
rem Every "exit /b" sits outside parentheses: inside a block, cmd /c can lose the exit code.
setlocal EnableExtensions
rem Always work from the folder that contains this file, wherever it is started from;
rem Streamlit also reads .streamlit\config.toml (the light theme) from here.
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set "MODE=%~1"

rem 1. Virtual environment with Python 3.12 (the py launcher if present, otherwise python on PATH).
if exist "%PY%" goto :check_python
echo [1/5] Creating the virtual environment .venv ...
where py >nul 2>nul && (py -3.12 -m venv .venv) || (python -m venv .venv)
if not exist "%PY%" goto :no_python
:check_python
"%PY%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)"
if errorlevel 1 goto :wrong_python

rem 2. Packages: CPU-only PyTorch first, so the requirements do not pull a large CUDA build.
"%PY%" -c "import torch, torchvision, cv2, streamlit, av, wastelens" >nul 2>nul
if not errorlevel 1 (
    echo [2/5] Packages already installed.
    goto :models
)
call :check_path_length || goto :path_too_long
echo [2/5] Installing PyTorch CPU and the pinned requirements; this downloads about 300 MB ...
"%PY%" -m pip install --upgrade pip || goto :install_failed
"%PY%" -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu || goto :install_failed
"%PY%" -m pip install -r requirements.txt || goto :install_failed
"%PY%" -m pip install --no-deps -e . || goto :install_failed

rem 3. Model checkpoints are not in Git; the app cannot classify without them.
:models
set "MISSING="
for %%M in (wastelens-baseline.pt wastelens-ensemble.pt wastelens-mobilenetv3.pt) do (
    if not exist "models\%%M" set "MISSING=1" & echo      Missing models\%%M
)
if defined MISSING (
    echo [3/5] Copy the three .pt files from the project's Google Drive folder into models\ ,
    echo       or run: %PY% scripts\prepare_app_models.py   after downloading the experiment outputs.
) else (
    echo [3/5] Model checkpoints found.
)

rem 4. Optional downloads: TrashNet test samples and the webcam detector. Failures do not stop the app.
if exist "data\trashnet\cardboard" (
    echo [4/5] TrashNet samples found.
) else (
    echo [4/5] Downloading and verifying TrashNet for the sample images ...
    "%PY%" scripts\prepare_trashnet.py || echo      Dataset download failed; the app still works with your own images.
)
if exist "models\ssdlite320_mobilenet_v3_large_coco-a79551df.pth" (
    echo [5/5] Webcam detector found.
) else (
    echo [5/5] Downloading the webcam object detector ...
    "%PY%" scripts\prepare_detector.py || echo      Detector download failed; image and video modes still work.
)

if /i "%MODE%"=="setup" goto :setup_done
if /i "%MODE%"=="test" goto :run_tests
if defined MISSING goto :missing_models
echo Starting WasteLens; the browser opens at the address printed below. Press Ctrl+C here to stop.
"%PY%" -m streamlit run app\app.py --browser.gatherUsageStats false
exit /b %errorlevel%

:setup_done
echo Setup finished. Start the app with: run.bat
exit /b 0

:run_tests
"%PY%" -m unittest discover -s tests
exit /b %errorlevel%

:missing_models
echo Cannot start: model checkpoints are missing, see step 3.
exit /b 1

:no_python
echo Could not create .venv. Install Python 3.12 from https://www.python.org/downloads/ and retry.
exit /b 1

:wrong_python
echo .venv does not use Python 3.12. Delete the .venv folder, install Python 3.12 and run again.
exit /b 1

:path_too_long
echo The folder path is too long for PyTorch on this Windows installation:
echo   %~dp0
echo Move the project to a short path such as C:\WasteLens and run run.bat again,
echo or enable Windows long paths: https://pip.pypa.io/warnings/enable-long-paths
exit /b 1

:install_failed
echo Installation failed. Check the internet connection and the messages above, then run again.
echo If pip mentions "Long Path support", move this folder to a short path such as C:\WasteLens.
exit /b 1

:check_path_length
rem PyTorch installs files about 145 characters deep inside .venv, and Windows limits paths to
rem 260 characters unless long paths are enabled. Stop early instead of failing mid-install.
reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2>nul | find "0x1" >nul
if not errorlevel 1 exit /b 0
set "HERE=%~dp0"
rem Too long when the folder path still has a character at position 100 (0-based).
if "%HERE:~100,1%"=="" exit /b 0
exit /b 1
