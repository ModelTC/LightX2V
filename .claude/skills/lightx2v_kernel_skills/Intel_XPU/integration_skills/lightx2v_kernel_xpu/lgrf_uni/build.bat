@echo off
REM Build SDP ESIMD DLL for PTL-H (Panther Lake) with doubleGRF
REM Generates: esimd.unify.lgrf.dll + esimd.unify.lgrf.lib

cd /d "%~dp0"

if not defined SETVARS_COMPLETED (
    call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
)

echo Building ESIMD SDP DLL for PTL-H...
icpx sdp_kernels.cpp -shared -o esimd.unify.lgrf.dll ^
    -DBUILD_ESIMD_KERNEL_LIB ^
    -fsycl -fsycl-targets=spir64_gen ^
    -Xs "-device ptl-h -options -doubleGRF" ^
    -O3

if errorlevel 1 (
    echo ERROR: DLL build failed
    exit /b 1
)

echo DLL build succeeded: esimd.unify.lgrf.dll
