@echo off
REM ======================================================================
REM  build.bat
REM  Same flow as build_la.bat:
REM    1. Build lgrf_uni\esimd.unify.lgrf.dll  (icpx, PTL-H doubleGRF)
REM    2. cmake direct build → _cmake_build\   → _ext.*.pyd
REM    3. Copy pyd + dll into python\sycl_kernels\
REM    4. Smoke test  (test\test_sdp.py)
REM    5. pip wheel . --no-build-isolation → dist\*.whl
REM  Dumps full log to stdout at the end.
REM ======================================================================
setlocal

set "PROJ=%~dp0"
set "LOGFILE=%PROJ%build.log"
set "ERRFILE=%PROJ%build.err"

echo === Build start === > "%LOGFILE%"
echo %DATE% %TIME% >> "%LOGFILE%"
echo. > "%ERRFILE%"

REM ── Detect Python ──────────────────────────────────────────────────────────
if defined CONDA_PREFIX (
    set "PYEXE=%CONDA_PREFIX%\python.exe"
) else (
    for /f "tokens=*" %%p in ('where python 2^>nul') do if not defined PYEXE set "PYEXE=%%p"
)
if not exist "%PYEXE%" (
    echo ERROR: Python not found. Activate your conda/venv first.
    echo ERROR: Python not found >> "%LOGFILE%"
    goto :dump_and_fail
)
echo PYEXE=%PYEXE% >> "%LOGFILE%"

REM ── Detect cmake ───────────────────────────────────────────────────────────
"%PYEXE%" -c "import cmake,os; open('_tmp.txt','w').write(os.path.join(os.path.dirname(cmake.__file__),'data','bin','cmake.exe'))" 2>nul
if exist _tmp.txt (set /p CMAKE_EXE=<_tmp.txt & del _tmp.txt 2>nul)
if not defined CMAKE_EXE (
    for /f "tokens=*" %%c in ('where cmake 2^>nul') do if not defined CMAKE_EXE set "CMAKE_EXE=%%c"
)
if not defined CMAKE_EXE (
    echo ERROR: cmake not found. Run: conda install cmake
    echo ERROR: cmake not found >> "%LOGFILE%"
    goto :dump_and_fail
)
echo CMAKE_EXE=%CMAKE_EXE% >> "%LOGFILE%"

REM ── Detect ninja ───────────────────────────────────────────────────────────
for /f "tokens=*" %%n in ('where ninja 2^>nul') do if not defined NINJA_EXE set "NINJA_EXE=%%n"
if not defined NINJA_EXE (
    echo ERROR: ninja not found. Run: conda install ninja
    echo ERROR: ninja not found >> "%LOGFILE%"
    goto :dump_and_fail
)
echo NINJA_EXE=%NINJA_EXE% >> "%LOGFILE%"

REM ── Detect MSVC via vswhere ────────────────────────────────────────────────
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found.
    echo ERROR: vswhere not found >> "%LOGFILE%"
    goto :dump_and_fail
)
for /f "usebackq tokens=*" %%v in (`"%VSWHERE%" -latest -property installationPath`) do set "VS_INSTALL=%%v"
if not defined VS_INSTALL (
    echo ERROR: No Visual Studio installation found.
    echo ERROR: VS not found >> "%LOGFILE%"
    goto :dump_and_fail
)
call "%VS_INSTALL%\VC\Auxiliary\Build\vcvarsall.bat" x64 >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo vcvarsall FAILED >> "%LOGFILE%" & goto :dump_and_fail)
echo vcvarsall OK >> "%LOGFILE%"

REM ── Activate Intel oneAPI ─────────────────────────────────────────────────
if defined ONEAPI_ROOT (
    set "SETVARS=%ONEAPI_ROOT%\setvars.bat"
) else if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
    set "SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
) else if exist "C:\Program Files\Intel\oneAPI\setvars.bat" (
    set "SETVARS=C:\Program Files\Intel\oneAPI\setvars.bat"
) else (
    echo ERROR: Intel oneAPI setvars.bat not found.
    echo ERROR: oneAPI setvars not found >> "%LOGFILE%"
    goto :dump_and_fail
)
call "%SETVARS%" --force
where icpx >nul 2>&1
if errorlevel 1 (echo setvars FAILED -- icpx not in PATH >> "%LOGFILE%" & goto :dump_and_fail)
echo setvars OK >> "%LOGFILE%"

REM ── Detect torch root ─────────────────────────────────────────────────────
"%PYEXE%" -c "import os,torch; open('_tmp.txt','w').write(os.path.dirname(torch.__file__))" 2>> "%ERRFILE%"
if errorlevel 1 (
    echo ERROR: torch not importable from %PYEXE%
    echo ERROR: torch not importable >> "%LOGFILE%"
    goto :dump_and_fail
)
set /p torch_root=<_tmp.txt
del _tmp.txt 2>nul
set "torch_root=%torch_root:\=/%"
echo torch_root=%torch_root% >> "%LOGFILE%"

REM ══════════════════════════════════════════════════════════════════════════
REM  Step 1: Build ESIMD DLL
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo === Step 1: Build ESIMD DLL ===
echo === Step 1: Build ESIMD DLL === >> "%LOGFILE%"
cd /d "%PROJ%lgrf_uni"

if exist esimd.unify.lgrf.dll (
    del /f esimd.unify.lgrf.dll 2>nul
    if errorlevel 1 (
        echo ERROR: esimd.unify.lgrf.dll is locked.
        echo ERROR: DLL locked >> "%LOGFILE%"
        goto :dump_and_fail
    )
)

icpx sdp_kernels.cpp -shared -o esimd.unify.lgrf.dll ^
    -DBUILD_ESIMD_KERNEL_LIB ^
    -fsycl -fsycl-targets=spir64_gen ^
    -Xs "-device ptl-h -options -doubleGRF" ^
    -O3 >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo DLL BUILD FAILED >> "%LOGFILE%" & goto :dump_and_fail)
echo DLL build OK >> "%LOGFILE%"
echo DLL build OK

REM ══════════════════════════════════════════════════════════════════════════
REM  Step 2: cmake build → _cmake_build\ → _ext.*.pyd
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo === Step 2: cmake build (.pyd) ===
echo === Step 2: cmake build (.pyd) === >> "%LOGFILE%"
cd /d "%PROJ%"

if exist _cmake_build rmdir /s /q _cmake_build

"%CMAKE_EXE%" -GNinja ^
    "-DCMAKE_MAKE_PROGRAM=%NINJA_EXE%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER=icx ^
    -DCMAKE_CXX_STANDARD=20 ^
    "-DCMAKE_PREFIX_PATH=%torch_root%" ^
    "-DPython_EXECUTABLE=%PYEXE%" ^
    -B _cmake_build -S . >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo CMAKE CONFIGURE FAILED >> "%LOGFILE%" & goto :dump_and_fail)

"%CMAKE_EXE%" --build _cmake_build --config Release -j >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo CMAKE BUILD FAILED >> "%LOGFILE%" & goto :dump_and_fail)
echo cmake build OK >> "%LOGFILE%"
echo cmake build OK

REM ══════════════════════════════════════════════════════════════════════════
REM  Step 3: Copy artifacts → python\sycl_kernels\
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo === Step 3: Copy artifacts ===
echo === Step 3: Copy artifacts === >> "%LOGFILE%"
cd /d "%PROJ%"

for /f "tokens=*" %%f in ('dir /b "_cmake_build\_ext*.pyd" 2^>nul') do (
    copy /y "_cmake_build\%%f" "python\sycl_kernels\" >> "%LOGFILE%" 2>> "%ERRFILE%"
    echo Copied %%f >> "%LOGFILE%"
)
copy /y "lgrf_uni\esimd.unify.lgrf.dll" "python\sycl_kernels\" >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo COPY FAILED >> "%LOGFILE%" & goto :dump_and_fail)
echo Artifacts copied >> "%LOGFILE%"
echo Artifacts copied to python\sycl_kernels\

REM ══════════════════════════════════════════════════════════════════════════
REM  Step 4: Smoke test
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo === Step 4: test_sdp.py ===
echo === Step 4: test_sdp.py === >> "%LOGFILE%"
cd /d "%PROJ%"
set "PYTHONPATH=%PROJ%python;%PYTHONPATH%"

if exist test\test_sdp.py (
    "%PYEXE%" test\test_sdp.py >> "%LOGFILE%" 2>> "%ERRFILE%"
    if errorlevel 1 (echo TEST FAILED >> "%LOGFILE%" & echo TEST FAILED) else (echo TEST PASSED >> "%LOGFILE%" & echo TEST PASSED)
) else (
    echo test\test_sdp.py not found >> "%LOGFILE%"
)

REM ══════════════════════════════════════════════════════════════════════════
REM  Step 5: Build wheel  (pip wheel --no-build-isolation)
REM  NOTE: python -m build --no-isolation fails when cmake is only a system
REM  binary (not pip-installed): scikit-build-core injects "cmake>=3.22" as
REM  a dynamic build requirement and the `build` package rejects it.
REM  pip wheel --no-build-isolation skips that pre-check.
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo === Step 5: Build wheel ===
echo === Step 5: Build wheel === >> "%LOGFILE%"
cd /d "%PROJ%"

if exist dist rmdir /s /q dist
set "CMAKE_PREFIX_PATH=%torch_root%"

"%PYEXE%" -m pip wheel . --no-build-isolation -w dist >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo WHEEL BUILD FAILED >> "%LOGFILE%" & goto :dump_and_fail)
echo Wheel build OK >> "%LOGFILE%"

for /f "tokens=*" %%w in ('dir /b "dist\*.whl" 2^>nul') do (
    echo Wheel: dist\%%w
    echo Wheel: dist\%%w >> "%LOGFILE%"
)

echo.
echo === FINISHED ===
echo   pyd   : python\sycl_kernels\
echo   wheel : dist\
echo   log   : %LOGFILE%
echo.
echo %DATE% %TIME% >> "%LOGFILE%"
echo ===FINISHED=== >> "%LOGFILE%"
goto :dump_log

:dump_and_fail
echo ===FAILED=== >> "%LOGFILE%"
:dump_log
type "%LOGFILE%"
endlocal
exit /b 0
