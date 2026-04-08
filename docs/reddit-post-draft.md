# Reddit r/LocalLLaMA Post Draft

> Flair: Discussion
> 게시 타이밍: 화-목 오전 9-11 AM ET (한국시간 밤 10시-새벽 12시)

---

## Title Options (pick one)

**Option A** (surprising finding lead):
> I benchmarked Qwen3.5 (9B→122B) across 4 unified-memory platforms. The 35B MoE is faster than the 9B Dense on every single one.

**Option B** (hardware comparison lead):
> Apple Silicon vs DGX Spark vs Ryzen AI MAX 395 vs RTX 3090x2: 5,100 Qwen3.5 benchmark runs across 5 engines

**Option C** (practical angle):
> Ran 5,100 benchmarks comparing Qwen3.5 on 4 platforms with 5 backends. Here's which hardware actually matters for local inference.

---

## Post Body

I wanted to answer a simple question: if you have ~$2-5K to spend on local LLM inference, which hardware actually gives you the best tok/s?

So I ran Qwen3.5 (9B, 27B, 35B-A3B MoE, 122B-A10B MoE) on 4 platforms with 5 backends. ~5,100 total measurements, 4,200 valid after outlier filtering (CV<0.3). Every run uses cold prefill — no cache, no prompt reuse, per-run random nonce, server restart between prefill tracks.

### Hardware

| | M5 Max 128GB | RTX 3090×2 48GB | DGX Spark GB10 128GB | Ryzen AI MAX 395 96GB |
|--|:--:|:--:|:--:|:--:|
| Memory BW | 546 GB/s | ~936 GB/s | 273 GB/s | 256 GB/s |
| Architecture | Unified | Discrete VRAM | Unified | Unified (iGPU) |

### Generation Speed (llama.cpp, same GGUF, Q4_K_M, gen-512)

| Model | M5 Max | 3090×2 | DGX Spark | Ryzen AI |
|-------|------:|------:|------:|------:|
| **9B** Dense | 75.9 | **117.6** | 36.8 | 32.6 |
| **27B** Dense | 24.8 | **41.4** | 11.5 | 10.3 |
| **35B-A3B** MoE | 94.1 | **138.9** | 59.6 | 58.0 |
| **122B-A10B** MoE | 42.9 | OOM | 21.7 | 22.9 |

### The MoE surprise

35B-A3B MoE (only 3B active params) is **faster than 9B Dense on every platform**:

| Hardware | 9B Dense | 35B MoE | Speedup |
|---------|------:|------:|------:|
| M5 Max | 75.9 | **94.1** | +24% |
| 3090×2 | 117.6 | **138.9** | +18% |
| DGX Spark | 36.8 | **59.6** | +62% |
| Ryzen AI | 32.6 | **58.0** | +78% |

The speedup is most dramatic on the lower-bandwidth platforms (DGX Spark, Ryzen AI) — makes sense since MoE only loads ~3B weights per token vs 9B for the dense model.

### Engine comparison (gen-512, Q4_K_M)

On the 3090×2, vLLM GPTQ-Marlin hit **156.3 tok/s** on the 35B MoE — fastest single result across the entire benchmark. But for most models, llama.cpp wins:

| Platform | Best Engine | 9B | 27B | 35B MoE |
|----------|-----------|----:|----:|--------:|
| Mac | MLX | **102.4** | 28.8 | **138.3** |
| 3090×2 | llama.cpp/vLLM | 117.3 | 41.5 | **156.3** (vLLM) |
| DGX Spark | llama.cpp | 35.7 | 11.5 | 61.2 |
| Ryzen AI | llama.cpp | 36.2 | 12.3 | 58.4 |

### 122B on a $2K mini PC

The Ryzen AI MAX 395 (HP Z2 Mini G1a) runs 122B-A10B MoE at **22.9 tok/s** — usable for real work. DGX Spark is similar at 21.7. The 3090×2 can't even load it (48GB VRAM, 122B GPTQ is ~65GB).

M5 Max leads 122B at **42.9 tok/s** — 128GB unified memory + 546 GB/s bandwidth wins for the biggest models.

### Key takeaways

- **Memory bandwidth is king.** 3090×2 at 936 GB/s dominates everything that fits in VRAM. Once you exceed VRAM → it falls apart
- **MoE models flip the rankings.** 35B-A3B MoE is 18-78% faster than 9B Dense despite being "bigger" — the active parameter count matters more than total params
- **Unified memory platforms run everything.** Mac/DGX/Ryzen all run 122B. The 3090 can't
- **DGX Spark and Ryzen AI MAX 395 are surprisingly close.** Despite very different architectures (Blackwell vs Strix Halo RDNA 3.5), gen TPS is within 10-15% on most models
- **llama.cpp is the universal winner** for generation speed (except MLX on Mac, vLLM GPTQ on 3090)

### Methodology

- 5 runs per combo, median
- Warmup 1 (separate prompt) + measure 5
- `--no-cache-prompt`, `--slot-prompt-similarity 0`, per-run random nonce prefix
- Server restart between prefill tracks
- Thermal guard: 85°C → 60s cooldown
- Randomized backend/model/track order

Full write-up with prefill data, Q8_0 results, and methodology deep-dive: [blog link]

Open-source benchmark tool + raw CSV data: [repo link]

**Want me to run a specific benchmark on this hardware?** Open a [GitHub issue](https://github.com/baem1n/llm-bench/issues) with the model/quant/track you want tested and I'll run it. Just keep in mind this tool measures single-stream inference speed under controlled conditions — it's not a general-purpose quality or accuracy eval, just raw throughput on specific hardware.

---

What are you running Qwen3.5 on? Curious how the 35B MoE performs on single-GPU setups (4090, 5080) — if anyone has numbers I'd love to compare.
