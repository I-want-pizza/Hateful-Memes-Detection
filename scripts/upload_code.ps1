# upload_code.ps1 - Upload project code to remote via single tar.gz archive.
# Run from code\ folder:
#   powershell -ExecutionPolicy Bypass -File scripts\upload_code.ps1

. "$PSScriptRoot\load_env.ps1" -EnvPath ".env"

if (-not $REMOTE_IP) { Write-Error "REMOTE_IP not set in .env"; exit 1 }

$port  = if ($REMOTE_PORT) { $REMOTE_PORT } else { "22" }
$rpath = if ($REMOTE_PATH) { $REMOTE_PATH } else { "~/hateful_memes" }

$archive = "$env:TEMP\hateful_memes_code.tar.gz"

Write-Host "=== Packing code ===" -ForegroundColor Cyan

# Pack everything except data, outputs, wandb, __pycache__, .env, *.pt
tar -czf $archive `
    --exclude="./data" `
    --exclude="./outputs" `
    --exclude="./wandb" `
    --exclude="./.env" `
    --exclude="./.git" `
    --exclude="./__pycache__" `
    --exclude="*/__pycache__" `
    --exclude="*.pyc" `
    --exclude="*.pt" `
    -C . .

$sizeMB = [math]::Round((Get-Item $archive).Length / 1MB, 1)
Write-Host "  Archive : $archive (~${sizeMB} MB)"
Write-Host ""
Write-Host "=== Uploading (single connection) ===" -ForegroundColor Cyan
Write-Host "  To : ${REMOTE_USER}@${REMOTE_IP}:${rpath}/  (port $port)"

ssh -p $port "${REMOTE_USER}@${REMOTE_IP}" "mkdir -p $rpath"
scp -P $port $archive "${REMOTE_USER}@${REMOTE_IP}:${rpath}/code.tar.gz"

Write-Host ""
Write-Host "=== Extracting on remote ===" -ForegroundColor Cyan

ssh -p $port "${REMOTE_USER}@${REMOTE_IP}" @"
cd $rpath
tar -xzf code.tar.gz
rm code.tar.gz
echo 'Done!'
"@

Remove-Item $archive -Force

Write-Host ""
Write-Host "=== Uploading .env (separately, not in archive) ===" -ForegroundColor Cyan

# On remote, DATA_DIR must be "data" (relative), not a Windows path
$envContent = Get-Content ".env" -Raw
$envContent = $envContent -replace "(?m)^DATA_DIR=.*$", "DATA_DIR=data"
$tmpEnv = "$env:TEMP\hateful_memes_remote.env"
Set-Content -Path $tmpEnv -Value $envContent -NoNewline

scp -P $port "$tmpEnv" "${REMOTE_USER}@${REMOTE_IP}:${rpath}/.env"
Remove-Item $tmpEnv -Force
Write-Host "  .env -> remote OK (DATA_DIR set to 'data')" -ForegroundColor Green

Write-Host ""
Write-Host "=== All done! ===" -ForegroundColor Green
Write-Host "Next: on the server run:"
Write-Host "  bash scripts/download_dataset_remote.sh"
