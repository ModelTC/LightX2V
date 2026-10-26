# Windows build and non-interactive shell notes

Read this when the normal Windows build fails during environment setup, tool
discovery or invocation from Git Bash/MSYS2. These are historical failure modes,
not requirements to replace a working environment. The current
[build.bat](../../../../../../../lightx2v_kernel/intel_xpu/build.bat) already discovers
MSVC and the real CMake executable, but still calls `setvars.bat` directly.

## oneAPI initialization and paths containing parentheses

Earlier piped `cmd.exe` sessions reported `'vars.bat' is not recognized` under
`Program Files (x86)`, sometimes with a successful `setvars.bat` exit status but
no compiler in PATH. Check the log and `where icpx`; an exit code alone is not
enough. If that failure is reproduced, use full paths to the required installed
components rather than a relative `call vars.bat`. For example, in a temporary
launcher adapted to the installed toolkit:

```bat
set "BUILD_ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
call "%BUILD_ONEAPI_ROOT%\compiler\latest\env\vars.bat"
where icpx
if errorlevel 1 exit /b 1
```

Initialize other required components from their actual full paths. Do not assume
every toolkit uses `latest`, or that this compiler-only example sets up oneDNN
and MSVC. The full-path workaround is retained from the earlier guide; it was
not reproduced during this documentation review.

## CMake launcher exits without output

Some earlier environments resolved `Scripts\cmake.exe` to a failing Python
launcher. Inspect `where cmake` and the selected Python before reinstalling.
When CMake is installed as a Python package, locate its actual binary with:

```cmd
python -c "import cmake,os; print(os.path.join(os.path.dirname(cmake.__file__),'data','bin','cmake.exe'))"
```

Set `CMAKE_EXE` to that result in the temporary launcher. The repository build
script already attempts this lookup; retain a working system CMake when the
Python package is absent.

## Git Bash/MSYS2 invoking cmd.exe

Prefer a normal activated cmd.exe environment first. If activation blocks only
in a piped/non-interactive session, the earlier guide used a separate `.bat`
launcher with an explicitly selected environment:

```bat
@echo off
setlocal
set "BUILD_ENV_PREFIX=C:\path\to\existing\conda-env"
set "CONDA_PREFIX=%BUILD_ENV_PREFIX%"
set "PATH=%BUILD_ENV_PREFIX%;%BUILD_ENV_PREFIX%\Scripts;%BUILD_ENV_PREFIX%\Library\bin;%PATH%"
cd /d D:\path\to\project
python -c "import sys,torch; print(sys.executable, torch.__version__)"
if errorlevel 1 exit /b 1
call build.bat
exit /b %errorlevel%
```

This bypasses activation hooks, so use it only when the selected environment's
requirements are known and validated. Preserve compiler setup and any necessary
DLL search paths; capture stdout/stderr in the task's temporary directory.
Both the current and historical `build.bat` can return 0 after failure, so also
inspect `build.log` for `===FAILED===` / `===FINISHED===`, check individual test
results and confirm a fresh wheel with the expected contents. Forwarding the
batch exit code alone does not establish success.

## Packaging and UR header differences

- The maintained project uses `python -m pip wheel . --no-build-isolation -w dist`;
  here `-w` selects the wheel output directory. In `python -m build`, `-w` means
  `--wheel`, and the output directory instead uses `-o dist`. Do not transfer
  flags between these tools.
- Inspect the active compiler's include tree before generating the `ur_api.h`
  shim. Both CMake projects generate `${SYCL_INCLUDE_DIR}/ur_api.h` in a separate
  build-directory shim. The historical checked-in header retains an absolute
  machine-specific path with an extra `/sycl/` segment; do not copy it over the
  generated shim.
- Load the packaged ESIMD DLL before importing `_ext`, preserving the DLL search
  directory handles for the required lifetime. Use the maintained Python loader
  as the implementation reference.
