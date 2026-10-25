FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

COPY benchmark_cuda.py /workspace/solution/benchmark_cuda.py
COPY solve.sh /workspace/solution/solve.sh

COPY test.sh /workspace/tests/test.sh 
COPY test_outputs.py /workspace/tests/test_outputs.py 

COPY --chmod=0755 timer.sh  /workspace/timer.sh 

RUN curl -fsSL https://claude.ai/install.sh | bash

CMD ["python", "/workspace/benchmark_cuda.py", "--size", "2048", "--warmup", "5", "--iterations", "20"]
