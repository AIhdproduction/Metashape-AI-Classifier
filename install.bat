@echo off
setlocal
cd /d "%~dp0"
set "MSAM3_NOPAUSE=0"
if /I "%~1"=="--no-pause" set "MSAM3_NOPAUSE=1"
set "NICEGUI_STORAGE_PATH=%~dp0work\.nicegui"
set "HF_HUB_DISABLE_TELEMETRY=1"
echo Installing AIhdproduction Metashape AI Classifier (FlurSeg and SAM 3)...
echo Running it again after every git pull updates an existing installation.
where git >nul 2>nul
if errorlevel 1 (
  echo ERROR: Install Git for Windows from https://git-scm.com/download/win first.
  goto :failed
)
py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>nul
if errorlevel 1 (
  echo ERROR: Install 64-bit Python 3.12 including the Python launcher from python.org.
  goto :failed
)
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Existing .venv must use 64-bit Python 3.12'"
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install --upgrade "pip>=25,<27" "setuptools>=75,<81" wheel
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install "triton-windows>=3.3,<3.4"
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -e ".[sam]"
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :failed
echo.
echo Checking the app...
".venv\Scripts\python.exe" -m msam3.install_check
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -c "import nicegui; import msam3.app.screen; print('App verified.')"
if errorlevel 1 goto :failed
echo Checking the FlurSeg code...
".venv\Scripts\python.exe" -c "from transformers import EomtDinov3ForUniversalSegmentation; import msam3.segmentation.flurseg_backend; print('FlurSeg code verified.')"
if errorlevel 1 goto :failed
echo Checking the SAM 3 code...
".venv\Scripts\python.exe" -c "from sam3 import build_sam3_image_model; print('SAM 3 code verified.')"
if errorlevel 1 goto :failed
echo Checking CUDA...
".venv\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: update NVIDIA driver and check your GPU'; print('GPU:', torch.cuda.get_device_name(0))"
if errorlevel 1 goto :failed
if not exist ".venv\Scripts\hf.exe" (
  echo ERROR: .venv\Scripts\hf.exe is missing. It belongs to huggingface_hub.
  goto :failed
)
".venv\Scripts\python.exe" -m msam3 classes --model flurseg
if errorlevel 1 goto :failed
echo.
echo ======================================================================
echo Installation complete. Weights are not downloaded automatically.
echo.
echo 1. FlurSeg, the default model. Target folder:
echo      %~dp0models\flurseg\v1
echo    with config.json, model.safetensors and preprocessor_config.json.
echo    Skip this step if the files are already there. Otherwise run in this folder:
echo.
echo .venv\Scripts\hf.exe download AIhdproduction/Flurseg --revision 5df53dbbf721778f1ecd4903369e9465926ae2fd --include "v1/*" --local-dir models/flurseg
echo.
echo    Browser route: https://huggingface.co/AIhdproduction/Flurseg/tree/main/v1
echo.
echo 2. Optional, only for SAM 3 with your own classes:
echo    Request access at https://huggingface.co/facebook/sam3
echo    and save the checkpoint as models\sam3.pt.
echo.
echo Then in Metashape: Tools, Run Script, start_in_metashape.py
echo ======================================================================
if "%MSAM3_NOPAUSE%"=="0" pause
exit /b 0
:failed
echo.
echo Installation failed. Read the error above, fix it, and run install.bat again.
if "%MSAM3_NOPAUSE%"=="0" pause
exit /b 1
