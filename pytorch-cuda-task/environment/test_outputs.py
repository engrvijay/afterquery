"""
Use this file to define pytest tests that verify the outputs of the task.

This file will be copied to /tests/test_outputs.py and run by the /tests/test.sh file
from the working directory.
"""
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

# Override BENCHMARK_PATH if Harbor mounts the solution elsewhere.
BENCHMARK = Path(os.environ.get("BENCHMARK_PATH", "/solution/benchmark_cuda.py"))


def run_benchmark(size=1024, warmup=2, iterations=5):
    assert BENCHMARK.is_file(), (
        f"Benchmark not found at {BENCHMARK}. Set BENCHMARK_PATH if needed."
    )
    return subprocess.run(
        [
            sys.executable, str(BENCHMARK),
            "--size", str(size),
            "--warmup", str(warmup),
            "--iterations", str(iterations),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def extract_json_report(output):
    """Extract the first JSON object from benchmark output."""
    if "GPU BENCHMARK RESULTS" in output:
        output = output.split("GPU BENCHMARK RESULTS", 1)[1]

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", output):
        candidate = output[match.start():].lstrip()
        try:
            report, _ = decoder.raw_decode(candidate)
            if isinstance(report, dict):
                return report
        except json.JSONDecodeError:
            continue

    pytest.fail("Could not find a valid JSON report in benchmark output.")


def test_cuda_gpu_is_available():
    """Fail if Harbor's verifier environment cannot access a CUDA GPU."""
    assert torch.cuda.is_available(), (
        "CUDA is unavailable. Configure Harbor to expose an NVIDIA GPU "
        "to the verifier environment."
    )
    assert torch.cuda.device_count() > 0
    gpu_name = torch.cuda.get_device_name(0)
    assert gpu_name.strip(), "CUDA device name is empty."
    print(f"Detected GPU: {gpu_name}")


def test_matrix_multiplication_executes_on_cuda():
    """Run real matrix multiplication using CUDA tensors."""
    assert torch.cuda.is_available(), "CUDA is unavailable."

    device = torch.device("cuda:0")
    a = torch.randn((512, 512), device=device)
    b = torch.randn((512, 512), device=device)
    result = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    assert a.device.type == "cuda"
    assert b.device.type == "cuda"
    assert result.device.type == "cuda"
    assert result.shape == (512, 512)
    assert torch.isfinite(result).all().item()


def test_benchmark_runs_successfully_and_reports_gpu():
    """Run the benchmark and check its exit status and GPU information."""
    completed = run_benchmark()
    output = completed.stdout + "\n" + completed.stderr

    assert completed.returncode == 0, (
        f"Benchmark exited with code {completed.returncode}.\n{output}"
    )
    assert "GPU:" in output, "Benchmark did not report a GPU name."
    assert "CUDA:" in output, "Benchmark did not report the CUDA version."
    assert "GPU BENCHMARK RESULTS" in output, "Benchmark results heading missing."


def test_benchmark_timing_metrics_are_valid():
    """Validate median, minimum, and maximum timings in the JSON report."""
    completed = run_benchmark()
    output = completed.stdout + "\n" + completed.stderr

    assert completed.returncode == 0, (
        f"Benchmark exited with code {completed.returncode}.\n{output}"
    )

    report = extract_json_report(output)
    required = (
        "median_execution_seconds",
        "minimum_execution_seconds",
        "maximum_execution_seconds",
    )

    for key in required:
        assert key in report, f"Missing timing field: {key}"
        value = report[key]
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{key} must be numeric; got {value!r}"
        )
        assert math.isfinite(value) and value > 0, (
            f"{key} must be a finite positive number; got {value!r}"
        )

    assert report["minimum_execution_seconds"] <= report["median_execution_seconds"]
    assert report["median_execution_seconds"] <= report["maximum_execution_seconds"]
