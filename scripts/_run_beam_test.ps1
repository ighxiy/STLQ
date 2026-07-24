$ErrorActionPreference = 'Continue'
$env:STLQ_CUDA_BEAM_SYNC = '1'
Set-Location 'd:\code\vq\stlq\stlq_gpu\cmake-build-win'
Write-Host "=== Running test_beam_large_root_best_err_matches_cost.exe ==="
Write-Host "STLQ_CUDA_BEAM_SYNC = $($env:STLQ_CUDA_BEAM_SYNC)"
& '.\test_beam_large_root_best_err_matches_cost.exe' 2>&1
Write-Host "Exit code: $LASTEXITCODE"
