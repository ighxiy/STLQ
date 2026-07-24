@echo off
setlocal enableextensions

set "VSDEV=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
set "ONEAPI=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

call "%VSDEV%" -arch=x64 >nul 2>&1
call "%ONEAPI%" intel64 >nul 2>&1

cd /d "d:\code\vq\stlq\stlq_gpu\cmake-build-win"

echo.
echo === test_beam_large_root_best_err_matches_cost ===
test_beam_large_root_best_err_matches_cost.exe
echo Exit code: %ERRORLEVEL%

echo.
echo === test_streaming_encode_base ===
test_streaming_encode_base.exe
echo Exit code: %ERRORLEVEL%

echo.
echo === test_linkage_encode_forced_root_device_xc_from_init ===
test_linkage_encode_forced_root_device_xc_from_init.exe
echo Exit code: %ERRORLEVEL%

echo.
echo === test_icm_cuda_strict ===
test_icm_cuda_strict.exe
echo Exit code: %ERRORLEVEL%

echo.
echo === test_icm_cuda_large_root_strict ===
test_icm_cuda_large_root_strict.exe
echo Exit code: %ERRORLEVEL%

endlocal
