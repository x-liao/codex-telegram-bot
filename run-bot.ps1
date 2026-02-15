$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BotFile = Join-Path $Root "bot.py"
$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir "bot.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Resolve-Python {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Command = "python"; Args = @() }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ Command = "py"; Args = @("-3") }
    }
    throw "Python not found in PATH."
}

if (-not (Test-Path $BotFile)) {
    throw "bot.py not found at $BotFile"
}

$py = Resolve-Python

while ($true) {
    $start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "$start [runner] starting bot"

    try {
        & $py.Command @($py.Args + @($BotFile)) *>> $LogFile
        $exitCode = $LASTEXITCODE
    }
    catch {
        $exitCode = -1
        $err = $_ | Out-String
        Add-Content -Path $LogFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [runner] launch error: $err"
    }

    $end = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "$end [runner] bot exited with code=$exitCode; restart in 3s"
    Start-Sleep -Seconds 3
}
