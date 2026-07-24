@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul 2>&1
cd /d d:\code\vq\stlq\stlq_gpu\cmake-build-win
"C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" --build . --target stlq_main 2>&1
if %ERRORLEVEL% == 0 (echo BUILD_OK) else (echo BUILD_FAIL)
