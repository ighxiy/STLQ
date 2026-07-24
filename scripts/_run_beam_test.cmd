@echo off
setlocal enableextensions

REM Set up environment (VS + oneAPI for MKL/CUDA DLLs) then run the beam test.

set "VSDEV=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
set "ONEAPI=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

call "%VSDEV%" -arch=x64 >nul 2>&1
call "%ONEAPI%" intel64 >nul 2>&1

set STLQ_CUDA_BEAM_SYNC=1

cd /d "d:\code\vq\stlq\stlq_gpu\cmake-build-win"

echo === Running test_beam_large_root_best_err_matches_cost.exe ===
echo STLQ_CUDA_BEAM_SYNC=%STLQ_CUDA_BEAM_SYNC%
test_beam_large_root_best_err_matches_cost.exe
echo Exit code: %ERRORLEVEL%

endlocal
