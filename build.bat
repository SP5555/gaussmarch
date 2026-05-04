@echo off

:: Verify nvcc
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found. Is CUDA Toolkit installed and on PATH?
    exit /b 1
)

for /f "tokens=*" %%i in ('where nvcc') do set NVCC_PATH=%%i
echo Using nvcc: %NVCC_PATH%

if not exist build mkdir build
cd build

cmake .. ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    "-DCMAKE_CUDA_ARCHITECTURES=86;89" ^
    -DCMAKE_CUDA_COMPILER="%NVCC_PATH%"

cmake --build . --config Release --parallel