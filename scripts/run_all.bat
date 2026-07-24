@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "d:\code\vq\stlq"
if errorlevel 1 goto :fail

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
if errorlevel 1 goto :fail

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
if errorlevel 1 goto :fail

cd /d "d:\code\vq\stlq\stlq_gpu\cmake-build-win"
if errorlevel 1 goto :fail

set "PATH=C:\Program Files\HDF_Group\HDF5\2.0.0\bin;%PATH%"

set "PROGRAM=stlq_main.exe"
set "BASE_CFG=..\configs\basic_desktop_win.cfg"

set "DRY_RUN=0"
set "CONTINUE_ON_ERROR=1"

set "TRAIN_INIT_LINKAGE_MODES=legacy_root_only"
set "TRAIN_LINKAGE_KNN_KS=10"
set "BASE_LINKAGE_KNN_KS=15"

rem linked groups: train_num_layers, base_num_layers, train_depth_k, base_depth_k
set "GROUP_COUNT=1"
set "GROUP_1=14,16,10,15"
if not exist "%PROGRAM%" (
    echo PROGRAM NOT FOUND: %PROGRAM%
    goto :fail
)

if not exist "%BASE_CFG%" (
    echo CONFIG NOT FOUND: %BASE_CFG%
    goto :fail
)

set /a RUN_ID=0

for %%M in (%TRAIN_INIT_LINKAGE_MODES%) do (
    for %%B in (%BASE_LINKAGE_KNN_KS%) do (
        for %%T in (%TRAIN_LINKAGE_KNN_KS%) do (
            for /L %%I in (1,1,%GROUP_COUNT%) do (
                set "G=!GROUP_%%I!"
                for /f "tokens=1,2,3,4 delims=," %%a in ("!G!") do (
                    set /a RUN_ID+=1
                    echo ==========================================
                    echo RUN !RUN_ID!
                    echo mode=%%M
                    echo base.linkage.knn_k=%%B
                    echo train.linkage.knn_k=%%T
                    echo train.linkage.num_layers=%%a
                    echo base.linkage.num_layers=%%b
                    echo train.linkage.depth_k=%%c
                    echo base.linkage.depth_k=%%d
                    echo ==========================================

                    if "!DRY_RUN!"=="0" (
                        "%PROGRAM%" --config "%BASE_CFG%" ^
                            --set "train.init_linkage_mode=%%M" ^
                            --set "base.linkage.knn_k=%%B" ^
                            --set "train.linkage.knn_k=%%T" ^
                            --set "train.linkage.num_layers=%%a" ^
                            --set "base.linkage.num_layers=%%b" ^
                            --set "train.linkage.depth_k=%%c" ^
                            --set "base.linkage.depth_k=%%d"

                        set "RC=!ERRORLEVEL!"
                        if not "!RC!"=="0" (
                            echo FAILED: !RC!
                            if "!CONTINUE_ON_ERROR!"=="0" goto :fail
                        )
                    )
                )
            )
        )
    )
)

echo ALL DONE
pause
exit /b 0

:fail
echo SCRIPT FAILED
pause
exit /b 1