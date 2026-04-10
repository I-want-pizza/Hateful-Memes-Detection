# watch_logs.ps1 - Live-tail training logs from remote.
# Run from code\ folder:
#   powershell -ExecutionPolicy Bypass -File scripts\watch_logs.ps1
#
# Press Ctrl+C to stop.

. "$PSScriptRoot\load_env.ps1" -EnvPath ".env"

if (-not $REMOTE_IP) { Write-Error "REMOTE_IP not set in .env"; exit 1 }

$port  = if ($REMOTE_PORT) { $REMOTE_PORT } else { "22" }
$rpath = if ($REMOTE_PATH) { $REMOTE_PATH } else { "~/hateful_memes" }

Write-Host "=== Watching logs on ${REMOTE_USER}@${REMOTE_IP}:${port} ===" -ForegroundColor Cyan
Write-Host "=== Press Ctrl+C to stop ===" -ForegroundColor Yellow
Write-Host ""

ssh -p $port "${REMOTE_USER}@${REMOTE_IP}" "tail -f ${rpath}/run_all.log"
