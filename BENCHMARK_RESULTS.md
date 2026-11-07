# LLM Inference Benchmark Results

## Hardware

- GPU: NVIDIA GeForce GTX 1650
- VRAM: 4 GB
- Model: Qwen2.5-1.5B-Instruct
- Inference Engine: llama.cpp

## 1. Model Size Comparison

| Model Format | Size |
|---|---:|
| F16 GGUF | 2944.68 MiB |
| Q4_K_M GGUF | 934.69 MiB |

Approximate storage reduction: 68.3%.

## 2. Inference Performance

| Model | Execution | Prompt Processing | Generation |
|---|---|---:|---:|
| Q4_K_M | GPU | 218.4 tokens/s | 92.8 tokens/s |
| Q4_K_M | CPU | 103.6 tokens/s | 14.4 tokens/s |
| F16 | GPU | 53.3 tokens/s | 40.1 tokens/s |

## 3. GPU Memory Usage

| Configuration | Observed GPU Memory |
|---|---:|
| Q4_K_M | 1995 MiB |
| F16 | 3203 MiB |

These are snapshots of total GPU memory usage while inference was running, not isolated measurements of model weights.

## 4. Observations

- Q4_K_M significantly reduced model storage requirements.
- GPU inference was faster than CPU inference in these experiments.
- Q4_K_M showed higher measured throughput than F16 in the individual runs.
- F16 required more GPU memory than Q4_K_M.

## 5. Benchmark Limitations

These results are preliminary measurements from individual runs.

For reliable comparisons, future experiments should control:

- Prompt length.
- Number of generated tokens.
- Context length.
- Batch size.
- GPU temperature and utilization.
- Background system processes.
- Number of repeated runs.

The results should not be treated as definitive performance comparisons until repeated under controlled conditions.