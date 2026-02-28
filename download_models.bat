@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Download Pocket TTS ONNX models from Hugging Face
rem Usage: download_models.bat [output_dir]

set "MODEL_ID=KevinAHM/pocket-tts-onnx"
set "OUTPUT_DIR=%~1"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=models"

set "PYTHON_CMD="
where py >nul 2>&1 && set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
    where python >nul 2>&1 && set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo ERROR: Python 3 was not found. Please install Python first.
    exit /b 1
)

set "VENV_DIR=%TEMP%\pocket_tts_hf_venv_%RANDOM%%RANDOM%"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

echo === Pocket TTS Model Downloader ===
echo Model: %MODEL_ID%
echo Output: %OUTPUT_DIR%
echo Temp venv: %VENV_DIR%
echo.

echo Creating temporary venv...
%PYTHON_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 goto :fail

echo Installing huggingface_hub in temporary venv...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%VENV_PYTHON%" -m pip install --upgrade huggingface_hub
if errorlevel 1 goto :fail

if not exist "%OUTPUT_DIR%\onnx" mkdir "%OUTPUT_DIR%\onnx"

echo Downloading models...
for %%F in (
    "onnx/mimi_encoder.onnx"
    "onnx/text_conditioner.onnx"
    "onnx/flow_lm_main_int8.onnx"
    "onnx/flow_lm_flow_int8.onnx"
    "onnx/mimi_decoder_int8.onnx"
) do (
    call :download_file "%%~F"
    if errorlevel 1 goto :fail
)

call :download_file "tokenizer.model"
if errorlevel 1 goto :fail

call :download_file "reference_sample.wav"
if errorlevel 1 goto :fail

echo.
echo === Download Complete ===
echo.
echo Directory structure:
dir "%OUTPUT_DIR%" /s /b /a-d
echo.

call :cleanup
exit /b 0

:download_file
set "FILE=%~1"
echo Downloading !FILE!...
set "HF_REPO=%MODEL_ID%"
set "HF_FILE=!FILE!"
set "HF_OUT=%OUTPUT_DIR%"
"%VENV_PYTHON%" -c "import os, shutil; from huggingface_hub import hf_hub_download; repo=os.environ['HF_REPO']; file=os.environ['HF_FILE']; out=os.environ['HF_OUT']; cached=hf_hub_download(repo_id=repo, filename=file); dst=os.path.join(out, *file.split('/')); d=os.path.dirname(dst); d and os.makedirs(d, exist_ok=True); shutil.copy2(cached, dst)"
exit /b %ERRORLEVEL%

:fail
echo.
echo ERROR: Download failed.
call :cleanup
exit /b 1

:cleanup
if exist "%VENV_DIR%" (
    echo Cleaning temporary venv...
    rmdir /s /q "%VENV_DIR%"
)
exit /b 0
