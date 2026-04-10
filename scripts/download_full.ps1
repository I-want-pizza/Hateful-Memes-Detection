# download_full.ps1 - Download EVERYTHING from remote (including .pt and wandb).
# Warning: this can be large (several GB).
# Run from code\ folder:
#   powershell -ExecutionPolicy Bypass -File scripts\download_full.ps1

. "$PSScriptRoot\load_env.ps1" -EnvPath ".env"

if (-not $REMOTE_IP) { Write-Error "REMOTE_IP not set in .env"; exit 1 }

$port      = if ($REMOTE_PORT)       { $REMOTE_PORT }       else { "22" }
$localDest = if ($LOCAL_RESULTS_DIR) { $LOCAL_RESULTS_DIR } else { "results" }
$rpath     = if ($REMOTE_PATH)       { $REMOTE_PATH }       else { "~/hateful_memes" }
$archive   = "$env:TEMP\hateful_memes_full.tar.gz"

New-Item -ItemType Directory -Force -Path $localDest | Out-Null

Write-Host "=== Checking size on remote ===" -ForegroundColor Cyan
ssh -p $port "${REMOTE_USER}@${REMOTE_IP}" "du -sh $rpath/outputs"

Write-Host ""
Write-Host "=== Packing full outputs on remote (may take a while) ===" -ForegroundColor Cyan

ssh -p $port "${REMOTE_USER}@${REMOTE_IP}" @"
cd $rpath
tar -czf /tmp/results_full.tar.gz \
    --exclude="outputs/*/wandb" \
    outputs/*/best.pt run_all.log results_summary.csv ablation_results.csv 2>/dev/null || \
tar -czf /tmp/results_full.tar.gz \
    --exclude="outputs/*/wandb" \
    outputs/*/best.pt run_all.log 2>/dev/null
du -sh /tmp/results_full.tar.gz
echo "Packed!"
"@

Write-Host ""
Write-Host "=== Downloading archive ===" -ForegroundColor Cyan
scp -P $port -o ServerAliveInterval=20 -o ServerAliveCountMax=10 `
    "${REMOTE_USER}@${REMOTE_IP}:/tmp/results_full.tar.gz" "$archive"

$sizeMB = [math]::Round((Get-Item $archive).Length / 1MB, 0)
Write-Host "  Downloaded: ~${sizeMB} MB"

Write-Host ""
Write-Host "=== Extracting to $localDest ===" -ForegroundColor Cyan
tar -xzf $archive -C $localDest

Remove-Item $archive -Force
ssh -p $port "${REMOTE_USER}@${REMOTE_IP}" "rm -f /tmp/results_full.tar.gz"

Write-Host ""
Write-Host "=== Done -> $localDest ===" -ForegroundColor Green
