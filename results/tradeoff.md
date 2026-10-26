# Accuracy-latency-size tradeoff (test)

Accuracy at the primary threshold (max_val_normal, frozen on val). Latency = ONNX Runtime extractor + nearest-neighbor search, batch 1, 4 threads, on Intel(R) Core(TM) Ultra 7 155U (results/bench.json). Size = ONNX file + bank file, MB = 2^20 bytes.

| variant | AUROC | AP | F1 | recall | p50 ms | p95 ms | size MB |
|---|---|---|---|---|---|---|---|
| fp32 | 1.0000 | 1.0000 | 0.969 | 1.000 (63/63) | 95.24 | 114.00 | 32.21 |
| int8 | 1.0000 | 1.0000 | 0.977 | 1.000 (63/63) | 84.56 | 98.14 | 24.31 |
