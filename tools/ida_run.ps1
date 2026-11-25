# Run TensorLens headless inside IDA on one binary.
# IDA Free has no batch (-B) or autonomous (-A) mode, so this starts IDA normally with
# -Otensorlens:<report.json>, accepts the "Load a new file" dialog with Enter, and lets the
# plugin write its report and close IDA when auto-analysis ends.
# Usage: powershell -File tools\ida_run.ps1 <binary> [-Work <dir>] [-TimeoutMin 20] [-Show]
#   -Show keeps IDA open with the TensorLens windows (for screenshots).
param(
  [Parameter(Mandatory = $true)][string]$Binary,
  [string]$Work = '',
  [int]$TimeoutMin = 20,
  [switch]$Show,
  [switch]$Reuse  # open the existing .i64 instead of analysing from scratch
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Windows.Forms
if (-not $Work) { $Work = Join-Path $PSScriptRoot '..\..\out\ida' }
New-Item -ItemType Directory -Force $Work | Out-Null
$Work = (Resolve-Path $Work).Path
$Binary = (Resolve-Path $Binary).Path
$ida = 'C:\Program Files\IDA Free 9.5\ida.exe'
$name = [IO.Path]::GetFileName($Binary)
$dir = Join-Path $Work ([IO.Path]::GetFileNameWithoutExtension($name))
New-Item -ItemType Directory -Force $dir | Out-Null
$target = Join-Path $dir $name
if (-not (Test-Path $target)) { Copy-Item $Binary $target }
$db = "$target.i64"
$json = Join-Path $dir 'report.json'
$log = Join-Path $dir 'ida.log'
Remove-Item -ErrorAction SilentlyContinue $json, $log
if ($Show) { $env:TENSORLENS_SHOW = '1' } else { Remove-Item Env:TENSORLENS_SHOW -ErrorAction SilentlyContinue }
Get-ChildItem $dir -Filter "$name.*" | Where-Object { $_.Extension -in '.id0','.id1','.id2','.nam','.til' } | Remove-Item -Force
if (-not $Reuse) { Remove-Item -ErrorAction SilentlyContinue $db }
$open = if (Test-Path $db) { $db } else { $target }
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $ida -ArgumentList @("-Otensorlens:$json", "-Opdb:off", "-L$log", "`"$open`"") -PassThru -WorkingDirectory $dir

# Accept dialogs that block a fresh load. They are top-level windows of the IDA process.
# Preferred button per dialog: start fresh rather than reuse half-written databases.
$prefer = @('Overwrite', 'OK', 'Yes', 'Continue')
$root = [Windows.Automation.AutomationElement]::RootElement
$deadline = (Get-Date).AddMinutes($TimeoutMin)
$accepted = 0
$btnCond = New-Object Windows.Automation.PropertyCondition([Windows.Automation.AutomationElement]::ControlTypeProperty, [Windows.Automation.ControlType]::Button)
while (-not $p.HasExited -and (Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 700
  $cond = New-Object Windows.Automation.PropertyCondition([Windows.Automation.AutomationElement]::ProcessIdProperty, $p.Id)
  $wins = @()
  try { $wins = @($root.FindAll([Windows.Automation.TreeScope]::Children, $cond)) } catch { }
  foreach ($w in $wins) {
    $title = $w.Current.Name
    if ($title -match '^IDA - ' -or $title -match '^IDA Free') { continue }  # the main window
    try {
      $buttons = @($w.FindAll([Windows.Automation.TreeScope]::Descendants, $btnCond))
      $choices = $prefer
      # PDB prompts: "Yes" opens a file browser that blocks forever, so decline them.
      $allText = @($w.FindAll([Windows.Automation.TreeScope]::Descendants, [Windows.Automation.Condition]::TrueCondition) | ForEach-Object { $_.Current.Name }) -join ' '
      if ($allText -match 'pdb|Symbol Server') { $choices = @('No', 'Cancel') }
      foreach ($want in $choices) {
        $b = $buttons | Where-Object { $_.Current.Name -eq $want } | Select-Object -First 1
        if ($b) {
          $texts = @($w.FindAll([Windows.Automation.TreeScope]::Descendants,
            (New-Object Windows.Automation.PropertyCondition([Windows.Automation.AutomationElement]::ControlTypeProperty, [Windows.Automation.ControlType]::Text))) | ForEach-Object { $_.Current.Name }) -join ' | '
          "dialog '$title' [$texts] -> $want"
          $b.GetCurrentPattern([Windows.Automation.InvokePattern]::Pattern).Invoke()
          $accepted++
          break
        }
      }
    } catch { }
  }
  if ($Show -and (Test-Path $json)) { break }
}
if (-not $p.HasExited -and -not $Show) {
  if ((Get-Date) -ge $deadline) { "TIMEOUT after $TimeoutMin min" }
  Stop-Process -Id $p.Id -Force
}
$code = if ($p.HasExited) { $p.ExitCode } else { 'running' }
"exit $code after $([int]$sw.Elapsed.TotalSeconds) s, $accepted dialog(s) accepted"
if (Test-Path $log) { Get-Content $log | Select-String -Pattern 'TensorLens|rror|crash|exception' | Select-Object -Last 40 }
if (Test-Path $json) { "report: $json ($((Get-Item $json).Length) bytes)" } else { 'NO REPORT' }
