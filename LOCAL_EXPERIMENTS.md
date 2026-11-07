# Local LLM Quantization and Inference Experiments

## Overview

This document records hands-on experiments with Large Language Model (LLM) quantization, GGUF conversion, CUDA acceleration, GPU layer offloading, and local inference using llama.cpp.

The experiments use Qwen2.5-1.5B-Instruct on an NVIDIA GTX 1650 with 4 GB VRAM.

## Hardware and Environment

| Component | Configuration |
|---|---|
| Operating System | Windows 11 |
| GPU | NVIDIA GeForce GTX 1650 |
| GPU Memory | 4 GB |
| CUDA Toolkit | 12.5 |
| Python | 3.11 |
| Compiler | MSVC |
| Build System | CMake + Ninja |
| Inference Engine | llama.cpp |

## 1. Model Conversion

Original model: Qwen2.5-1.5B-Instruct

Source: Hugging Face

The original model was downloaded in Safetensors format.

It was converted into GGUF format using the llama.cpp conversion script.

Conversion command:

```bash
python convert_hf_to_gguf.py models/qwen2.5-1.5b-instruct --outfile models/converted/qwen2.5-1.5b-f16.gguf --outtype f16
```

Output: F16 GGUF model.

## 2. Model Quantization

The F16 GGUF model was quantized into Q4_K_M.

Command:

```bash
build-cuda/bin/llama-quantize.exe models/converted/qwen2.5-1.5b-f16.gguf models/converted/qwen2.5-1.5b-q4_k_m.gguf Q4_K_M
```

Q4_K_M is a K-quantization method that uses mixed precision for selected tensors.

### Model Size Comparison

| Format | Size |
|---|---:|
| F16 | 2944.68 MiB |
| Q4_K_M | 934.69 MiB |

The quantized model reduced storage requirements by approximately 68%.

## 3. Local Inference

Inference was performed using llama.cpp's `llama-cli`.

GPU inference:

```bash
build-cuda/bin/llama-cli.exe -m models/converted/qwen2.5-1.5b-q4_k_m.gguf -p "Explain how a transformer architecture works in simple terms." -n 200 -ngl 99
```

CPU inference:

```bash
build-cuda/bin/llama-cli.exe -m models/converted/qwen2.5-1.5b-q4_k_m.gguf -p "Explain how a transformer architecture works in simple terms." -n 200 -ngl 0
```

## 4. GPU Layer Offloading

GPU layer offloading allows selected model layers to execute on the GPU rather than the CPU.

The `-ngl` parameter controls the number of model layers requested for GPU offloading.

Examples:

- `-ngl 0`: CPU-only inference.
- `-ngl 10`: Request up to 10 layers on the GPU.
- `-ngl 99`: Request maximum available GPU offloading.

The actual number of layers offloaded depends on the model and available GPU memory.

During the experiment, llama.cpp reported that 29/29 layers were offloaded for the Qwen model.

## 5. Local HTTP Serving

Explored llama.cpp's `llama-server` for serving the model through an HTTP interface.

The server provides OpenAI-compatible API endpoints, allowing external applications to communicate with the locally running model.

## 6. Initial Performance Measurements

| Configuration | Prompt Processing | Generation |
|---|---:|---:|
| Q4_K_M - GPU | 218.4 tokens/s | 92.8 tokens/s |
| Q4_K_M - CPU | 103.6 tokens/s | 14.4 tokens/s |
| F16 - GPU | 53.3 tokens/s | 40.1 tokens/s |

These are initial individual-run measurements, not controlled benchmark results.

## 7. Key Learnings

- Difference between model conversion and quantization.
- GGUF model format and its role in llama.cpp.
- K-quantization and mixed-precision quantization.
- CUDA-enabled llama.cpp builds.
- GPU layer offloading.
- CPU versus GPU inference.
- Monitoring GPU memory using `nvidia-smi`.
- Running local LLMs through an HTTP API.

## 8. Future Experiments

- Compare Q4_K_M, Q5_K_M, Q6_K, and Q8_0.
- Measure output quality across quantization levels.
- Benchmark inference under controlled conditions.
- Explore context length and KV cache memory consumption.
- Test concurrent requests using llama-server.
- Dockerize the inference server.
- Evaluate serving latency and throughput.