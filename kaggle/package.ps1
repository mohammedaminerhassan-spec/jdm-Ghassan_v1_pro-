# kaggle/package.ps1 — Windows PowerShell packaging for Kaggle upload
param(
    [string]$OutFile = "ghassan-pro-src.zip"
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $PSScriptRoot
Set-Location $RepoDir

$OutPath = Join-Path $RepoDir $OutFile
if (Test-Path $OutPath) {
    Remove-Item -Path $OutPath -Force
}

Write-Host "Creating Kaggle package: $OutFile ..." -ForegroundColor Cyan

# Files and directories to exclude
$ExcludePatterns = @(
    "build",
    "artifacts",
    ".git",
    ".vscode",
    "*.obj",
    "*.o",
    "*.graw",
    "*.ckpt.bin",
    "*.zip"
)

$TempDir = Join-Path $env:TEMP ("ghassan_pkg_" + [System.Guid]::NewGuid().ToString().Substring(0,8))
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

try {
    # Copy files while skipping excluded patterns
    Get-ChildItem -Path $RepoDir -Recurse | ForEach-Object {
        $rel = $_.FullName.Substring($RepoDir.Length + 1)
        $skip = $false
        foreach ($pat in $ExcludePatterns) {
            if ($rel -like "$pat*" -or $rel -like "*\$pat*") {
                $skip = $true
                break
            }
        }
        if (-not $skip) {
            $dest = Join-Path $TempDir $rel
            if ($_.PSIsContainer) {
                New-Item -ItemType Directory -Path $dest -Force | Out-Null
            } else {
                $destParent = Split-Path -Parent $dest
                if (-not (Test-Path $destParent)) {
                    New-Item -ItemType Directory -Path $destParent -Force | Out-Null
                }
                Copy-Item -Path $_.FullName -Destination $dest -Force
            }
        }
    }

    # Create zip
    Compress-Archive -Path "$TempDir\*" -DestinationPath $OutPath -Force
    $size = (Get-Item $OutPath).Length / 1MB
    Write-Host "[package] Success! Created $OutPath ($([Math]::Round($size, 2)) MB)" -ForegroundColor Green
    Write-Host "[package] Upload this zip to Kaggle, unzip it, and run: bash kaggle/setup.sh" -ForegroundColor Yellow
}
finally {
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
