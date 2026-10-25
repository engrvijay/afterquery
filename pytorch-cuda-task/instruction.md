# Task: GPU-Only PyTorch CUDA Benchmark

## Objective

Execute a matrix multiplication benchmark using an NVIDIA GPU
and PyTorch CUDA.

## Requirements

1. Execute GPU operations exclusively.
2. Fail if CUDA is unavailable.
3. Perform GPU warm-up before timing.
4. Synchronize CUDA before and after each timed operation.
5. Perform repeated benchmark runs.
6. Report median, minimum, and maximum execution times.
7. Calculate estimated TFLOP/s.
8. Report GPU memory usage.
9. Verify numerical correctness.

## Acceptance Criteria

- The benchmark runs successfully on a CUDA-enabled GPU.
- The reported device is CUDA.
- Median execution time is positive.
- Numerical validation passes.
- GPU memory statistics are reported.

--RuntimeError: CUDA GPU unavailable. GPU-only execution is required.
