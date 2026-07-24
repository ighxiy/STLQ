@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul 2>&1
cd /d d:\code\vq\stlq\stlq_lsq_qps\cmake-build-win-rel
"C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" --build . --target lsq_qps_lib 2>&1
if %ERRORLEVEL% == 0 (echo BUILD_OK) else (echo BUILD_FAIL)
