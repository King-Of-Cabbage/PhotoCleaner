<#
  PhotoCleaner - 模块验证脚本 (由 Claude 生成)

  用法:
      powershell -ExecutionPolicy Bypass -File D:\photocleaner\scripts\ci.ps1
      ... -Label "M03-split-near-duplicate"   给这次运行起个名字
      ... -SkipRelease                        只跑 fmt + test
      ... -BootstrapToolchain                 没有工具链时自动下载安装 stable-msvc
      ... -CargoPath "X:\...\cargo.exe"       手动指定

  步骤: cargo fmt --check -> cargo test --all-targets -> cargo build --release
  输出: work\ci_last.log (UTF-8)，末行 RESULT 汇总。
  不 commit / 不 push / 不碰任何照片或数据。
#>
param(
    [string]$Label = "",
    [switch]$SkipRelease,
    [switch]$BootstrapToolchain,
    [string]$CargoPath = ""
)

$ErrorActionPreference = "Continue"
$Root    = Split-Path -Parent $PSScriptRoot
$WorkDir = Join-Path $Root "work"
if (!(Test-Path $WorkDir)) { New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null }
$Log = Join-Path $WorkDir "ci_last.log"

$OutputEncoding    = [System.Text.Encoding]::UTF8
$env:RUST_BACKTRACE = "1"

$lines = New-Object System.Collections.Generic.List[string]
function Emit([string]$t) { $lines.Add($t) | Out-Null; Write-Host $t }
function Save() { $lines | Set-Content -LiteralPath $Log -Encoding UTF8 }
function Run([string]$exe, [string[]]$a) { return (& $exe @a 2>&1 | ForEach-Object { $_.ToString() }) }

# ---------------------------------------------------------- 便携工具链环境
# 这台机器上 rust 是便携安装的：CARGO_HOME 在仓库的 work\tools\cargo
$PortableCargoHome  = Join-Path $WorkDir "tools\cargo"
$PortableRustupHome = Join-Path $WorkDir "tools\rustup"

if (Test-Path $PortableCargoHome)  { $env:CARGO_HOME  = $PortableCargoHome }
if (-not $env:RUSTUP_HOME) {
    if (Test-Path (Join-Path $PortableRustupHome "toolchains")) {
        $env:RUSTUP_HOME = $PortableRustupHome
    } elseif (Test-Path (Join-Path $env:USERPROFILE ".rustup\toolchains")) {
        $env:RUSTUP_HOME = Join-Path $env:USERPROFILE ".rustup"
    } else {
        $env:RUSTUP_HOME = $PortableRustupHome
    }
}

function Find-Exe([string]$name) {
    if ($env:CARGO_HOME) {
        $p = Join-Path $env:CARGO_HOME "bin\$name"
        if (Test-Path $p) { return $p }
    }
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in @(
        (Join-Path $env:USERPROFILE ".cargo\bin\$name"),
        "C:\ProgramData\chocolatey\bin\$name"
    )) { if (Test-Path $p) { return $p } }
    return $null
}

function Installed-Toolchains([string]$rustup) {
    if (-not $rustup) { return @() }
    $out = Run $rustup @("toolchain", "list")
    return @($out | Where-Object { $_ -and $_ -notmatch 'no installed toolchains' -and $_ -notmatch '^error' })
}

function Invoke-Step([string]$Name, [string]$Exe, [string[]]$A) {
    Emit ""
    Emit "==================== STEP $Name ===================="
    Emit "> $Exe $($A -join ' ')"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $out = Run $Exe $A
    $code = $LASTEXITCODE
    $sw.Stop()
    foreach ($l in $out) { $lines.Add($l) | Out-Null; Write-Host $l }
    Emit "---- $Name exit=$code  elapsed=$([int]$sw.Elapsed.TotalSeconds)s ----"
    Save
    return $code
}

Push-Location $Root
try {
    Emit "PhotoCleaner CI"
    Emit "time        : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Emit "root        : $Root"
    Emit "label       : $(if ($Label) { $Label } else { '(none)' })"
    Emit "CARGO_HOME  : $env:CARGO_HOME"
    Emit "RUSTUP_HOME : $env:RUSTUP_HOME"

    $cargo  = if ($CargoPath -and (Test-Path $CargoPath)) { $CargoPath } else { Find-Exe "cargo.exe" }
    $rustup = Find-Exe "rustup.exe"
    Emit "cargo       : $cargo"
    Emit "rustup      : $rustup"

    if (-not $cargo) {
        Emit ""
        Emit "FATAL: cargo.exe not found"
        Emit "RESULT fmt=SKIP test=SKIP build=SKIP overall=FAIL reason=cargo-not-found label=$Label"
        Save; exit 1
    }

    # -------------------------------------------------- 工具链检查 / 引导
    $toolchains = Installed-Toolchains $rustup
    Emit "toolchains  : $(if ($toolchains.Count) { $toolchains -join ' | ' } else { '(none)' })"

    if ($toolchains.Count -eq 0) {
        if (-not $BootstrapToolchain) {
            Emit ""
            Emit "FATAL: rustup has no installed toolchain."
            Emit ""
            Emit "cargo.exe here is only a rustup shim; the actual compiler is missing."
            Emit "The registry cache (crates already downloaded) is intact, so only the"
            Emit "toolchain itself needs to be fetched (roughly 300-500 MB, one time)."
            Emit ""
            Emit "Re-run with:"
            Emit "  powershell -ExecutionPolicy Bypass -File D:\photocleaner\scripts\ci.ps1 -BootstrapToolchain"
            Emit ""
            Emit "It will install stable-x86_64-pc-windows-msvc into:"
            Emit "  $env:RUSTUP_HOME"
            Emit "(that path is under work\ which .gitignore already excludes)"
            Emit ""
            Emit "RESULT fmt=SKIP test=SKIP build=SKIP overall=FAIL reason=no-toolchain label=$Label"
            Save; exit 1
        }

        Emit ""
        Emit "==================== BOOTSTRAP ===================="
        Emit "installing stable-x86_64-pc-windows-msvc into $env:RUSTUP_HOME ..."
        Save
        $bs = Invoke-Step "TOOLCHAIN-INSTALL" $rustup @("toolchain", "install", "stable-x86_64-pc-windows-msvc", "--profile", "default", "--no-self-update")
        if ($bs -ne 0) {
            Emit "RESULT fmt=SKIP test=SKIP build=SKIP overall=FAIL reason=toolchain-install-failed label=$Label"
            Save; exit 1
        }
        $bs = Invoke-Step "TOOLCHAIN-DEFAULT" $rustup @("default", "stable-x86_64-pc-windows-msvc")
        if ($bs -ne 0) {
            Emit "RESULT fmt=SKIP test=SKIP build=SKIP overall=FAIL reason=toolchain-default-failed label=$Label"
            Save; exit 1
        }
        $toolchains = Installed-Toolchains $rustup
        Emit "toolchains  : $($toolchains -join ' | ')"
    }

    $ver = Run $cargo @("--version")
    Emit "cargo ver   : $($ver -join ' ')"
    Emit "head        : $((Run 'git' @('rev-parse','HEAD')) -join '')"
    Emit "branch      : $((Run 'git' @('rev-parse','--abbrev-ref','HEAD')) -join '')"
    Save

    $fmt   = Invoke-Step "FMT"   $cargo @("fmt", "--all", "--", "--check")
    $test  = Invoke-Step "TEST"  $cargo @("test", "--all-targets")
    $build = if ($SkipRelease) { 0 } else { Invoke-Step "BUILD" $cargo @("build", "--release") }

    Emit ""
    Emit "==================== GIT ===================="
    foreach ($l in (Run 'git' @('status','--short')))               { $lines.Add($l) | Out-Null }
    Emit "---- git log ----"
    foreach ($l in (Run 'git' @('log','--oneline','--decorate','-5'))) { $lines.Add($l) | Out-Null }

    $fmtS   = if ($fmt  -eq 0) { "PASS" } else { "FAIL" }
    $testS  = if ($test -eq 0) { "PASS" } else { "FAIL" }
    $buildS = if ($SkipRelease) { "SKIP" } elseif ($build -eq 0) { "PASS" } else { "FAIL" }
    $overall = if ($fmt -eq 0 -and $test -eq 0 -and ($SkipRelease -or $build -eq 0)) { "PASS" } else { "FAIL" }

    Emit ""
    Emit "RESULT fmt=$fmtS test=$testS build=$buildS overall=$overall label=$Label"
    Save
    Write-Host ""
    Write-Host "full log: $Log"
    if ($overall -eq "PASS") { exit 0 } else { exit 1 }
}
finally { Pop-Location }
