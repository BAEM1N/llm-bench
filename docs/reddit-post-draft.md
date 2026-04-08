# Reddit r/LocalLLaMA Post Draft

> Flair: Discussion
> 게시 타이밍: 화-목 오전 9-11 AM ET (한국시간 밤 10시-새벽 12시)
> 차별점: 기존 리뷰(Tom's Hardware, NotebookCheck)는 1:1 비교 + quick run. 우리는 4개 플랫폼 × 5개 엔진 × 5,100회 통제 측정.

---

## Title

> I bought 3 unified-memory machines (M5 Max / DGX Spark / Ryzen AI MAX 395) to compare local inference. Added my 3090×2 as a discrete VRAM baseline. 5,100 runs later, here's what I found.

---

## Post Body

I recently picked up three unified-memory machines for different parts of my workflow — M5 Max as my primary dev machine, DGX Spark for fine-tuning LLM/embedding models before renting cloud GPUs, and the HP Z2 Mini (Ryzen AI MAX 395) as a dedicated inference server running llama.cpp for embeddings, reranking, and LLM serving. I already had a 3090×2 rig sitting around, so I figured: why not benchmark all of them under the same controlled conditions and share the results?

Most reviews I've seen compare these machines in pairs (DGX Spark vs Strix Halo, Mac vs everything else), and they're usually single-model quick runs without controlling for caching or prompt variation. I wanted proper apples-to-apples numbers across all four platforms with multiple engines.

So I ran Qwen3.5 (9B, 27B, 35B-A3B, 122B-A10B) across 5 backends (llama.cpp, MLX, Ollama, vLLM, Lemonade). ~5,100 total runs, ~4,200 valid after filtering (CV<0.3 per 5-run set). Cold prefill on every run — `--no-cache-prompt`, random nonce prefix, server restart between prefill tracks. No warm cache tricks.

### The machines

| | M5 Max | 3090×2 | DGX Spark | Ryzen AI 395 |
|--|--:|--:|--:|--:|
| Memory | 128GB unified | 48GB VRAM + 128GB DDR4 | 128GB unified | 96GB VRAM (unified) |
| Bandwidth | 546 GB/s | ~936 GB/s | 273 GB/s | 256 GB/s |

### Generation tok/s (llama.cpp, same GGUF, Q4_K_M)

| Model | M5 Max | 3090×2 | DGX Spark | Ryzen AI |
|-------|------:|------:|------:|------:|
| **9B** Dense | 75.9 | **117.6** | 36.8 | 32.6 |
| **27B** Dense | 24.8 | **41.4** | 11.5 | 10.3 |
| **35B-A3B** MoE | 94.1 | **138.9** | 59.6 | 58.0 |
| **122B** MoE | **42.9** | OOM | 21.7 | 22.9 |

The 3090 is fastest when the model fits in 48GB. Once it doesn't, game over — can't even load 122B.

### The MoE thing nobody talks about

35B-A3B only activates ~3B params per token. That means it's **faster than the 9B Dense** on every single platform:

| | 9B Dense | 35B MoE | Delta |
|--|------:|------:|------:|
| M5 Max | 75.9 | **94.1** | +24% |
| 3090×2 | 117.6 | **138.9** | +18% |
| DGX Spark | 36.8 | **59.6** | +62% |
| Ryzen AI | 32.6 | **58.0** | +78% |

The lower your bandwidth, the bigger the MoE advantage. DGX Spark and Ryzen AI see +62-78% because they're bandwidth-starved — loading 3B instead of 9B per token makes a massive difference there.

### Best engine per platform

| Platform | Winner | 9B | 35B MoE |
|----------|--------|----:|--------:|
| Mac | MLX | **102.4** | **138.3** |
| 3090×2 | vLLM (GPTQ) | 83.6 | **156.3** |
| DGX Spark | llama.cpp | **35.7** | **61.2** |
| Ryzen AI | llama.cpp | **36.2** | **58.4** |

vLLM with GPTQ-Marlin on the 3090 hit **156.3 tok/s** for 35B MoE — the single fastest result in the entire benchmark. But llama.cpp is the most consistent winner across all platforms.

### 122B on a mini PC

The $2K HP Z2 Mini (Ryzen AI MAX 395) runs 122B-A10B at **22.9 tok/s**. DGX Spark does 21.7. Both are genuinely usable. The 3090×2 can't even load it.

M5 Max leads at **42.9 tok/s** — 546 GB/s bandwidth wins for the biggest models.

### DGX Spark vs Ryzen AI: surprisingly close

Despite completely different architectures (Blackwell GB10 vs Strix Halo RDNA 3.5), gen TPS is within 10-15% on most models. Both have ~128GB unified memory, both are bandwidth-limited at ~256-273 GB/s. Tom's Hardware and NotebookCheck benchmarks showed similar results — but with single runs, not 5,100 controlled measurements.

### What I'd buy

- **Already have a beefy GPU (4090/3090)?** Keep it. Fastest for anything that fits in VRAM
- **Want to run 70B+ daily?** M5 Max if you can afford it, Ryzen AI MAX 395 if budget matters
- **DGX Spark?** Cool hardware, but the Ryzen AI delivers ~90% of the performance for less money

### Methodology

5 runs × median, warmup with separate prompt, `--no-cache-prompt`, `--slot-prompt-similarity 0`, random nonce per run, server restart between prefill tracks, 85°C thermal guard with 60s cooldown, randomized execution order.

Full results (prefill, Q8_0, engine comparisons): https://baem1n.dev/posts/llm-bench-03-results-tables/

Benchmark tool + all raw CSV data (5,100 runs): https://github.com/baem1n/llm-bench

I did this because I genuinely needed to know which machine to use for what, and I figured the data might help others making similar decisions. Hope it's useful.

**If there's a specific model, quantization, or metric you'd like tested on any of this hardware** — [open a GitHub issue](https://github.com/baem1n/llm-bench/issues/new?template=experiment_request.md) and I'll run it. Things like different context lengths, other model families, Q3/Q5/Q6 quants, whatever. This measures single-stream inference throughput (gen tok/s, prefill tok/s, TTFT) — not quality or accuracy.

If you have **128GB+ unified memory hardware** (M3 Ultra Mac Studio, or similar) and want to contribute results — especially prefill throughput at longer contexts (64K/128K) — PRs to the repo are very welcome.

---

Anyone else running Qwen3.5 locally? Curious about 4090/5080 single-GPU numbers for the 35B MoE, or how the M3 Ultra Mac Studio stacks up on prefill with that 800 GB/s bandwidth. Would love to expand the comparison.

---

## 게시 전 체크리스트

### 필수
- [ ] Mac Ollama 잔여 실험 완료 후 데이터 업데이트 (9B Q4, 27B Q8, 122B Q8 남음)
- [ ] 블로그 포스트에 Mac Ollama 결과 반영 (현재 `—` 표시된 셀들)
- [ ] README 데이터도 동기화
- [ ] 영문 블로그 포스트 작성 or 영문 요약 추가 (/en/ 경로)

### 권장
- [ ] 블로그 포스트에 answer block 추가 (테이블 앞에 1-2문장 요약 — AI 인용성 개선)
- [ ] 블로그 포스트에 질문형 헤딩 추가 ("Which hardware runs Qwen3.5 fastest?")
- [ ] 프라이버시 폴리시 페이지 추가
- [ ] 저자 프로필 사진 추가 (Person schema image)

### 게시 타이밍
- [ ] 화-목 오전 9-11 AM ET (한국시간 밤 10-12시)
- [ ] 게시 직후 30분간 댓글 모니터링 (첫 반응이 Reddit 알고리즘에 critical)

### 게시 후 팔로업
- [ ] 인기 댓글에 24시간 내 답변 (특히 "can you test X?" 요청)
- [ ] 요청된 추가 실험 → GitHub issue로 전환
- [ ] 1주 후 팔로업 포스트 고려 ("Follow-up: community-requested experiments")
- [ ] 동일 내용 Hacker News 제출 (Show HN 포맷, Reddit 반응 확인 후)
