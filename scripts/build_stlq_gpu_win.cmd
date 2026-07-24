@echo off
setlocal enableextensions enabledelayedexpansion

REM Configure + build stlq_gpu on Windows (Ninja + CUDA)

set "VSDEV=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
set "ONEAPI=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

call "%VSDEV%" -arch=x64 || exit /b 1
call "%ONEAPI%" intel64 || exit /b 1

"%CMAKE%" -S stlq_gpu -B stlq_gpu\cmake-build-win -G Ninja ^
  -DSTLQ_ENABLE_CUDA=ON ^
  -DSTLQ_USE_MKL=ON ^
  -DMKLROOT="C:/Program Files (x86)/Intel/oneAPI/mkl/latest" ^
  -DHDF5_ROOT="C:/Program Files/HDF_Group/HDF5/2.0.0" ^
  -DSTLQ_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe" ^
  -DCMAKE_BUILD_TYPE=Release || exit /b 1

"%CMAKE%" --build stlq_gpu\cmake-build-win -j 18 || exit /b 1

endlocal
