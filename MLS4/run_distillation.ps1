# PowerShell script to run distillation with XLA completely disabled
# This ensures XLA is disabled BEFORE Python starts

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting Distillation with XLA JIT DISABLED" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Set environment variables to completely disable XLA
$env:TF_XLA_FLAGS = "--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0"
$env:TF_DISABLE_XLA_JIT = "1"
$env:TF_XLA_AUTO_JIT = "0"
$env:TF_ENABLE_LAYOUT_OPT = "0"
$env:TF_DETERMINISTIC_OPS = "0"
$env:TF_CUDNN_DETERMINISTIC = "0"
$env:TF_ENABLE_AUTOGRAPH = "0"
$env:TF_ENABLE_FUNCTION_INLINING = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:TF_USE_LEGACY_KERAS = "1"

Write-Host "Environment variables set:" -ForegroundColor Green
Write-Host "  TF_XLA_FLAGS = $env:TF_XLA_FLAGS" -ForegroundColor Yellow
Write-Host "  TF_DISABLE_XLA_JIT = $env:TF_DISABLE_XLA_JIT" -ForegroundColor Yellow
Write-Host "  TF_XLA_AUTO_JIT = $env:TF_XLA_AUTO_JIT" -ForegroundColor Yellow
Write-Host ""

# Run the main script with distillation only
python main.py --onlyd --distillation-batch-size 32 --seed 42

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Distillation completed" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
