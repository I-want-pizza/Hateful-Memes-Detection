# download_results.ps1 - Download training results from remote as single archive.
# Run from code\ folder:
#   powershell -ExecutionPolicy Bypass -File scripts\download_results.ps1

. "$PSScriptRoot\load_env.ps1" -EnvPath ".env"

if (-not $REMOTE_IP) { Write-Error "REMOTE_IP not set in .env"; exit 1 }

$port      = if ($REMOTE_PORT)       { $REMOTE_PORT }       else { "22" }
$localDest = if ($LOCAL_RESULTS_DIR) { $LOCAL_RESULTS_DIR } else { "results" }
$rpath     = if ($REMOTE_PATH)       { $REMOTE_PATH }       else { "~/hateful_memes" }
$archive   = "$env:TEMP\hateful_memes_results.tar.gz"

New-Item -ItemType Directory -Force -Path $localDest | Out-Null

Write-Host "=== Packing results on remote ===" -ForegroundColor Cyan
Write-Host "  Remote : ${REMOTE_USER}@${REMOTE_IP}:${rpath}  (port $port)"

ssh -p $port "${REMOTE_USER}@${REMOTE_IP}" @"
cd $rpath
tar -czf /tmp/results.tar.gz \
    --exclude="outputs/*/wandb" \
    --exclude="outputs/*/*.pt" \
    outputs \
    analysis_outputs \
    run_all.log \
    results_summary.csv \
    ablation_results.csv 2>/dev/null || \
tar -czf /tmp/results.tar.gz \
    --exclude="outputs/*/wandb" \
    --exclude="outputs/*/*.pt" \
    outputs run_all.log 2>/dev/null
echo "Packed!"
"@

Write-Host ""
Write-Host "=== Downloading archive ===" -ForegroundColor Cyan
scp -P $port -o ServerAliveInterval=20 -o ServerAliveCountMax=10 `
    "${REMOTE_USER}@${REMOTE_IP}:/tmp/results.tar.gz" "$archive"

Write-Host ""
Write-Host "=== Extracting to $localDest ===" -ForegroundColor Cyan
tar -xzf $archive -C $localDest

# Cleanup
Remove-Item $archive -Force
ssh -p $port "${REMOTE_USER}@${REMOTE_IP}" "rm -f /tmp/results.tar.gz"

Write-Host ""
Write-Host "=== Done -> $localDest ===" -ForegroundColor Green
