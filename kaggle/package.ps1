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

# Files and directories to exclude.
# Paths here use FORWARD slashes; $rel is normalized to forward slashes too,
# because Get-ChildItem hands back backslash paths and `-like "artifacts/x*"`
# never matched `artifacts\x`. That silently packed the whole 1 GB parquet lake
# plus every checkpoint into the zip (and the `.git` pattern stripped
# .gitattributes, which is what keeps the .sh files LF on Linux).
$ExcludePatterns = @(
    "build",
    "build_test",
    "build_cpu",
    "out",
    "artifacts/shards*",
    "artifacts/checkpoints",
    "artifacts/*.gguf",
    "artifacts/synth",
    "artifacts/corpus",
    "dataset/english_parquet",
    "dataset/qa_darija",
    "dataset/parquet",
    "dataset/dataset-main",
    "dataset/model_speak",
    "english_parquet",
    "kaggle_upload",
    "*.parquet",
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
    Get-ChildItem -Path $RepoDir -Recurse -Force | ForEach-Object {
        $rel = $_.FullName.Substring($RepoDir.Length + 1) -replace '\\','/'
        $skip = $false
        foreach ($pat in $ExcludePatterns) {
            # A directory pattern matches the entry itself and everything under
            # it; a wildcard pattern is a plain -like test.
            if ($rel -eq $pat -or $rel -like "$pat/*" -or $rel -like $pat) {
                $skip = $true
                break
            }
        }
        if (-not $skip) {
            $dest = Join-Path $TempDir ($rel -replace '/','\')
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
    $kept = (Get-ChildItem -Path $TempDir -Recurse -File |
             Measure-Object -Property Length -Sum)
    Write-Host ("[package] staged {0} files, {1:N1} MB" -f `
                $kept.Count, ($kept.Sum / 1MB))

    # Create zip with FORWARD-slash entry names. Compress-Archive on Windows
    # PowerShell 5.1 stores backslashes, which is legal in a zip but means
    # `unzip` on Linux produces one odd file name instead of a directory tree.
    if (Get-Command Compress-Archive -ErrorAction SilentlyContinue) {
        Add-Type -AssemblyName System.IO.Compression | Out-Null
        Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null
        if (Test-Path $OutPath) { Remove-Item -Path $OutPath -Force }
        $zip = [System.IO.Compression.ZipFile]::Open($OutPath, 'Create')
        try {
            Get-ChildItem -Path $TempDir -Recurse -File | ForEach-Object {
                $entryName = $_.FullName.Substring($TempDir.Length + 1) -replace '\\','/'
                [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                    $zip, $_.FullName, $entryName,
                    [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
            }
        }
        finally { $zip.Dispose() }
    }
    else {
        Compress-Archive -Path "$TempDir\*" -DestinationPath $OutPath -Force
    }
    $size = (Get-Item $OutPath).Length / 1MB
    Write-Host "[package] Success! Created $OutPath ($([Math]::Round($size, 2)) MB)" -ForegroundColor Green
    Write-Host "[package] Upload this zip to Kaggle, unzip it, and run: bash kaggle/setup.sh" -ForegroundColor Yellow
}
finally {
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
