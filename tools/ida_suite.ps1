# Build, install the plugin into the IDA user folder, and run it headless over a list of binaries.
# Usage: powershell -File tools\ida_suite.ps1 [-NoBuild] [names...]
param([switch]$NoBuild, [string[]]$Only = @())
$ErrorActionPreference = 'Continue'
$Only = @($Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$repo = (Resolve-Path "$PSScriptRoot\..").Path
$comp = (Resolve-Path "$repo\..").Path
if (-not $NoBuild) {
  cmd /c "`"$repo\tools\build.bat`" -DTENSORLENS_PLUGIN=ON" 2>&1 | Select-String -Pattern '\): error|FAILED|CMake Error' | ForEach-Object { $_.Line }
  if ($LASTEXITCODE -ne 0) { 'BUILD FAILED'; exit 1 }
}
$plugins = Join-Path $env:APPDATA 'Hex-Rays\IDA Pro\plugins'
New-Item -ItemType Directory -Force $plugins | Out-Null
Copy-Item "$repo\build\idabin\plugins\tensorlens.dll" $plugins -Force
$c = "$comp\corpus"
$suite = [ordered]@{
  synthetic = "$c\synthetic\out\libtensorlens_embedded_x86_64_stripped.so"
  tflite216 = "$c\extracted\tensorflow-lite-2.16.1-aar\jni\x86_64\libtensorflowlite_jni.so"
  whobird   = "$c\extracted\fdroid-org.woheller69.whobird_55\lib\x86_64\libtensorflowlite_jni.so"
  tflite214 = "$c\extracted\tflite-runtime-2.14-linux\tflite_runtime\_pywrap_tensorflow_interpreter_wrapper.so"
  seeneva   = "$c\extracted\fdroid-app.seeneva.reader_4\lib\x86_64\libseeneva.so"
  dfn       = "$c\extracted\deepfilternet-0.5.6\libdeep_filter_ladspa-0.5.6-x86_64-unknown-linux-gnu.so"
}
foreach ($k in $suite.Keys) {
  if ($Only.Count -and -not ($Only -contains $k)) { continue }
  "=== $k"
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\tools\ida_run.ps1" $suite[$k] -Work "$comp\out\ida\$k" -TimeoutMin 25
}
