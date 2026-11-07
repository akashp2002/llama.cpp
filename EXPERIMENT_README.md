# LLM Quantization and Local Inference Experiments

Hands-on exploration of LLM quantization, GGUF conversion, CUDA-accelerated inference, GPU layer offloading, and local model serving using llama.cpp.

This project documents experiments with running and optimizing Large Language Models locally on consumer GPU hardware.

## Objectives

- Convert Hugging Face models into GGUF format.
- Explore different model quantization techniques.
- Run quantized LLMs locally using llama.cpp.
- Understand CUDA acceleration and GPU layer offloading.
- Compare CPU and GPU inference performance.
- Explore local LLM serving through HTTP APIs.

## Technology Stack

- **Inference Engine:** llama.cpp
- **Model:** Qwen2.5-1.5B-Instruct
- **Model Format:** GGUF
- **Quantization:** Q4_K_M
- **GPU Acceleration:** NVIDIA CUDA
- **Model Conversion:** Hugging Face to GGUF
- **Model Serving:** llama-server

## Hardware and Environment

| Component | Configuration |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce GTX 1650 |
| VRAM | 4 GB |
| CUDA Toolkit | 12.5 |
| Python | 3.11 |
| Build System | CMake + Ninja |
| Compiler | MSVC |

## Experiments

### 1. Model Conversion

Downloaded Qwen2.5-1.5B-Instruct from Hugging Face and converted the original Safetensors checkpoint into GGUF format using llama.cpp.

Output: F16 GGUF model.

### 2. Model Quantization

Quantized the F16 model into Q4_K_M using llama.cpp's quantization utility.

| Model Format | Size |
|---|---:|
| F16 GGUF | 2944.68 MiB |
| Q4_K_M GGUF | 934.69 MiB |

The Q4_K_M model achieved approximately 68% storage reduction compared with F16.

### 3. GPU Layer Offloading

Explored GPU layer offloading using the `-ngl` parameter.

- `-ngl 0`: CPU-only inference.
- `-ngl 10`: Request up to 10 layers on GPU.
- `-ngl 99`: Request maximum available GPU offloading.

Verified GPU allocation using NVIDIA System Management Interface (`nvidia-smi`).

### 4. CPU vs GPU Inference

Compared inference performance using CPU-only execution and CUDA-enabled GPU execution.

| Configuration | Prompt Processing | Generation |
|---|---:|---:|
| Q4_K_M - GPU | 218.4 tokens/s | 92.8 tokens/s |
| Q4_K_M - CPU | 103.6 tokens/s | 14.4 tokens/s |
| F16 - GPU | 53.3 tokens/s | 40.1 tokens/s |

These are initial individual-run measurements, not controlled benchmark results.

### 5. Local Model Serving

Explored `llama-server` to expose the locally running model through HTTP endpoints.

The server provides an OpenAI-compatible API interface that allows external applications to communicate with the local model.

## Key Learnings

- Model conversion vs quantization.
- GGUF model format.
- K-quantization and mixed precision.
- CUDA-enabled llama.cpp builds.
- GPU layer offloading.
- CPU vs GPU inference.
- GPU memory monitoring.
- Local LLM inference and HTTP serving.

## Future Improvements

- Benchmark additional quantization formats such as Q5_K_M, Q6_K, and Q8_0.
- Evaluate output quality across quantization levels.
- Perform controlled performance benchmarks.
- Explore KV cache memory consumption.
- Test concurrent inference requests.
- Containerize the inference server using Docker.

## References

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [GGUF Documentation](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

---

*Personal experiments documenting local LLM optimization and inference using llama.cpp.*