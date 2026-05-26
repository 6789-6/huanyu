$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Frontend = Join-Path $Root "runtime\frontend\frontweb"
$Backend = Join-Path $Root "runtime\backend\algorithm_service"
$ShouyuRoot = Join-Path $Root "ml\shouyu_project"

function Stop-HuanyuProcess {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -like "python*" -and
            $_.CommandLine -and
            (
                $_.CommandLine -like "*$Backend*app.py*" -or
                $_.CommandLine -like "*http.server 8080*" -and $_.CommandLine -like "*$Frontend*"
            )
        } |
        ForEach-Object {
            Write-Host "Stopping old process $($_.ProcessId)"
            Stop-Process -Id $_.ProcessId -Force
        }
}

function Wait-Http {
    param(
        [string]$Url,
        [int]$Seconds = 12
    )
    $deadline = (Get-Date).AddSeconds($Seconds)
    do {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            return $response.StatusCode
        } catch {
            Start-Sleep -Milliseconds 500
        }
    } while ((Get-Date) -lt $deadline)
    throw "Service did not become ready: $Url"
}

Write-Host "Huanyu root: $Root"
Write-Host "Backend:     $Backend"
Write-Host "Frontend:    $Frontend"
Write-Host "Models:      $ShouyuRoot"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating .venv..."
    py -3 -m venv (Join-Path $Root ".venv")
}

Write-Host "Checking Python dependencies..."
& $VenvPython -c "import flask, flask_cors, numpy, cv2, mediapipe, torch; print('deps ok')"

Stop-HuanyuProcess

$env:SHOUYU_ROOT = $ShouyuRoot

Write-Host "Starting Flask backend..."
Start-Process -FilePath $VenvPython -ArgumentList "app.py" -WorkingDirectory $Backend -WindowStyle Hidden
$backendCode = Wait-Http "http://127.0.0.1:5000/api/health"

Write-Host "Starting frontend..."
Start-Process -FilePath $VenvPython -ArgumentList "-m http.server 8080" -WorkingDirectory $Frontend -WindowStyle Hidden
$frontendCode = Wait-Http "http://127.0.0.1:8080/login.html"

Write-Host ""
Write-Host "Backend health:  $backendCode  http://127.0.0.1:5000/api/health"
Write-Host "Frontend health: $frontendCode http://127.0.0.1:8080/login.html"
Write-Host "Open: http://127.0.0.1:8080/login.html"
Write-Host "Open: http://127.0.0.1:8080/translate.html"
Write-Host "Open: http://127.0.0.1:8080/paths/index.html"
