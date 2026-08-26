param(
    [string]$Configuration = "release"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dist = Join-Path $Root "dist\PhotoCleaner"

if (!(Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "cargo was not found; the current development environment cannot build Release."
}

Push-Location $Root
try {
    cargo build --release
    if (Test-Path $Dist) {
        Remove-Item -LiteralPath $Dist -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Dist | Out-Null
    foreach ($dir in @("config", "data\indexes", "data\operations", "cache\thumbnails", "models", "runtime\onnx", "runtime\media", "codecs", "logs", "LICENSES")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $Dist $dir) | Out-Null
    }
    Copy-Item -LiteralPath (Join-Path $Root "target\release\PhotoCleaner.exe") -Destination (Join-Path $Dist "PhotoCleaner.exe")
    if (Test-Path (Join-Path $Root "models\dinov2_vits14.onnx")) {
        Copy-Item -LiteralPath (Join-Path $Root "models\dinov2_vits14.onnx") -Destination (Join-Path $Dist "models\dinov2_vits14.onnx")
    }
    if (Test-Path (Join-Path $Root "runtime\onnx")) {
        Copy-Item -Path (Join-Path $Root "runtime\onnx\*") -Destination (Join-Path $Dist "runtime\onnx") -Recurse -Force
    }
    if (Test-Path (Join-Path $Root "runtime\media")) {
        Copy-Item -Path (Join-Path $Root "runtime\media\*") -Destination (Join-Path $Dist "runtime\media") -Recurse -Force
    }
    Copy-Item -LiteralPath (Join-Path $Root "packaging\README.txt") -Destination (Join-Path $Dist "README.txt")
    Copy-Item -LiteralPath (Join-Path $Root "packaging\VERSIONS.txt") -Destination (Join-Path $Dist "VERSIONS.txt")
    if (Test-Path (Join-Path $Root "PERFORMANCE_REPORT.md")) {
        Copy-Item -LiteralPath (Join-Path $Root "PERFORMANCE_REPORT.md") -Destination (Join-Path $Dist "PERFORMANCE_REPORT.md")
    }
    if (Test-Path (Join-Path $Root "CACHE_BEHAVIOR.md")) {
        Copy-Item -LiteralPath (Join-Path $Root "CACHE_BEHAVIOR.md") -Destination (Join-Path $Dist "CACHE_BEHAVIOR.md")
    }

    $sizeLines = @()
    foreach ($item in Get-ChildItem -LiteralPath $Dist) {
        $bytes = if ($item.PSIsContainer) {
            (Get-ChildItem -LiteralPath $item.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        } else {
            $item.Length
        }
        $sizeLines += "{0}`t{1:N0} bytes" -f $item.Name, ($bytes -as [int64])
    }
    $sizeLines | Set-Content -LiteralPath (Join-Path $Dist "SIZE_REPORT.txt") -Encoding UTF8
}
finally {
    Pop-Location
}
