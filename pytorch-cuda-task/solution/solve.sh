#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo " Harbor GPU Benchmark Solution"
echo "======================================"

BENCHMARK="/solution/benchmark_cuda.py"

if [[ ! -f "$BENCHMARK" ]]; then
    echo "ERROR: Benchmark not found: $BENCHMARK"
    exit 1
fi

echo ""
echo "Checking Python and PyTorch..."

python - <<'PY'
import sys
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)

if not torch.cuda.is_available():
    print("ERROR: CUDA GPU is not available.")
    sys.exit(1)

print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
PY

echo ""
echo "Running GPU matrix multiplication benchmark..."

python "$BENCHMARK" \
    --size 2048 \
    --warmup 5 \
    --iterations 20

echo ""
echo "======================================"
echo " GPU benchmark completed successfully"
echo "======================================"

