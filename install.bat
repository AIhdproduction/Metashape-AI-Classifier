@echo off
setlocal
cd /d "%~dp0"
set "MSAM3_NOPAUSE=0"
if /I "%~1"=="--no-pause" set "MSAM3_NOPAUSE=1"
echo Installing AIhdproduction Metashape SAM 3 Classifier...
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
".venv\Scripts\python.exe" -c "import torch; from sam3 import build_sam3_image_model; assert torch.cuda.is_available(), 'CUDA unavailable: update NVIDIA driver and check your GPU'; print('GPU:', torch.cuda.get_device_name(0)); print('Installation verified.')"
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m msam3 classes
if errorlevel 1 goto :failed
echo.
echo Installation complete. Weights are not downloaded automatically.
echo Request access: https://huggingface.co/facebook/sam3
echo Download the checkpoint and save it as models\sam3.pt.
echo Next: .venv\Scripts\activate.bat
echo Then: python -m msam3 validate --project "D:\jobs\site\site.psx"
if "%MSAM3_NOPAUSE%"=="0" pause
exit /b 0
:failed
echo.
echo Installation failed. Read the error above, fix it, and run install.bat again.
if "%MSAM3_NOPAUSE%"=="0" pause
exit /b 1
