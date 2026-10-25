import argparse
import json
import statistics
import time

import torch


def run_benchmark(size=2048, warmup=5, iterations=20):
    """Benchmark GPU matrix multiplication using median execution time."""

    if size <= 0 or warmup < 0 or iterations <= 0:
        raise ValueError(
            "size and iterations must be positive; warmup cannot be negative"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. GPU-only execution is required."
        )

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)

    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Matrix size: {size} x {size}")
    print(f"runs: {iterations}")
    
    # Allocate input tensors directly on the GPU.
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)

    # Warm up CUDA execution and GPU libraries.
    for _ in range(warmup):
        result = torch.matmul(a, b)

    # Ensure warm-up work has completed before timing.
    torch.cuda.synchronize(device)

    # Reset memory statistics after warm-up.
    torch.cuda.reset_peak_memory_stats(device)

    # Measure each run separately.
    timings = []

    for _ in range(iterations):
        torch.cuda.synchronize(device)
        start = time.perf_counter()

        result = torch.matmul(a, b)

        # Wait for GPU work to finish before stopping the timer.
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        timings.append(elapsed)

    # Report the median of the individual timed runs.
    median_seconds = statistics.median(timings)
    min_seconds = min(timings)
    max_seconds = max(timings)

    # Calculate approximate FP32 matrix multiplication throughput.
    operations = 2 * size**3
    median_tflops = operations / median_seconds / 1e12

    # Collect GPU memory statistics.
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)

    report = {
        "median_execution_seconds": median_seconds,
        "minimum_execution_seconds": min_seconds,
        "maximum_execution_seconds": max_seconds,
    }

    print("\n========== GPU BENCHMARK RESULTS ==========")
    print(json.dumps(report, indent=2))
    print("===========================================")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU-only PyTorch CUDA benchmark using median timing"
    )

    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)

    args = parser.parse_args()

    run_benchmark(
        size=args.size,
        warmup=args.warmup,
        iterations=args.iterations,
    )
