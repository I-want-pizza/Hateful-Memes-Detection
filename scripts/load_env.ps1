# load_env.ps1 - Read .env and set variables in caller scope.
# Usage: . "$PSScriptRoot\load_env.ps1"  (dot-sourcing required)
# Expects .env to be in the current working directory (code\).

param([string]$EnvPath = ".env")

if (-not (Test-Path $EnvPath)) {
    Write-Error "ERROR: $EnvPath not found. Run scripts from the code\ folder."
    exit 1
}

Write-Host "  [env] $((Resolve-Path $EnvPath).Path)" -ForegroundColor DarkGray

Get-Content $EnvPath | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    $idx = $line.IndexOf("=")
    if ($idx -lt 0) { return }
    $name  = $line.Substring(0, $idx).Trim()
    $value = $line.Substring($idx + 1).Trim()
    Set-Variable -Name $name -Value $value -Scope Script
}
