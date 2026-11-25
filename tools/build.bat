@echo off
rem Configure and build TensorLens with MSVC + Ninja. Usage: tools\build.bat [extra cmake args]
setlocal
set "HERE=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VS=%%i"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined IDASDK set "IDASDK=%HERE%\..\deps\ida-sdk"
if not defined IDABIN set "IDABIN=%HERE%\build\idabin"
cmake -S "%HERE%" -B "%HERE%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DIDABIN="%IDABIN%" %* || exit /b 1
cmake --build "%HERE%\build" || exit /b 1
