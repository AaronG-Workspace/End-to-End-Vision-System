# Edge latency benchmark

Intel(R) Core(TM) Ultra 7 155U (14 logical cores), Windows-11-10.0.26200-SP0. ONNX Runtime 1.31.0 CPUExecutionProvider, graph optimization ORT_ENABLE_ALL, 4 intra-op / 1 inter-op threads; NN search in torch 2.14.1+cpu on 4 threads. Batch 1, one val image, 50 warmup runs per variant, 1000 timed runs per variant, interleaved, order alternated. 2026-10-25. Notes: plugged in, Zoom/Teams/browsers closed, VS Code with Claude Code running.

| variant | extractor p50 / p95 / p99 ms | NN search p50 / p95 / p99 ms | total p50 / p95 / p99 ms | size MB (ONNX + bank) |
|---|---|---|---|---|
| fp32 | 14.45 / 18.07 / 18.99 | 79.88 / 98.21 / 102.70 | 95.24 / 114.00 / 119.05 | 10.62 + 21.59 = 32.21 |
| int8 | 6.74 / 8.07 / 8.63 | 77.79 / 91.43 / 99.46 | 84.56 / 98.14 / 106.81 | 2.72 + 21.59 = 24.31 |
