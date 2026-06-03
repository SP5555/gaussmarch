@echo off

:: Verify nvcc
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found. Is CUDA Toolkit installed and on PATH?
    exit /b 1
)

:: Check OptiX
if "%OPTIX_INSTALL_DIR%"=="" (
    echo ERROR: OPTIX_INSTALL_DIR is not set.
    echo   Command Prompt:  set OPTIX_INSTALL_DIR=C:\path\to\OptiX
    echo   PowerShell:      $env:OPTIX_INSTALL_DIR = "C:\path\to\OptiX"
    echo   NOTE: In PowerShell, 'set VAR=...' creates a PS variable, not an env var.
    echo   Or pass it directly: cmake .. -DOptiX_INSTALL_DIR=C:\path\to\OptiX
    exit /b 1
)

for /f "tokens=*" %%i in ('where nvcc') do set NVCC_PATH=%%i
echo Using nvcc: %NVCC_PATH%
echo OptiX: %OPTIX_INSTALL_DIR%

if not exist build mkdir build
cd build

cmake .. ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_CUDA_COMPILER="%NVCC_PATH%" ^
    -DOptiX_INSTALL_DIR="%OPTIX_INSTALL_DIR%"

cmake --build . --config Release --parallel
