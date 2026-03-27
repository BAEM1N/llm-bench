# Model Download Status

> 마지막 업데이트: 2026-03-28 00:30 KST
> 디바이스: Apple Silicon MacBook, 128GB unified memory

---

## GGUF

저장 위치: `~/.lmstudio/models/unsloth/`
다운로드 명령: `huggingface-cli download <repo> <file> --local-dir <dest>`

| 모델 | 양자화 | 크기 | 상태 | HuggingFace | 다운로드 명령 |
|------|--------|------|------|-------------|--------------|
| Qwen3.5-9B | Q4_K_M | 5.3G | ✅ | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | `hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/.lmstudio/models/unsloth/Qwen3.5-9B-GGUF` |
| Qwen3.5-9B | Q8_0 | 8.9G | ✅ | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | `hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q8_0.gguf --local-dir ~/.lmstudio/models/unsloth/Qwen3.5-9B-GGUF` |
| Qwen3.5-27B | Q4_K_M | 16G | ✅ | [unsloth/Qwen3.5-27B-GGUF](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | `hf download unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf --local-dir ~/.lmstudio/models/unsloth/Qwen3.5-27B-GGUF` |
| Qwen3.5-27B | Q8_0 | 27G | ✅ | [unsloth/Qwen3.5-27B-GGUF](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | `hf download unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q8_0.gguf --local-dir ~/.lmstudio/models/unsloth/Qwen3.5-27B-GGUF` |
| Qwen3.5-35B-A3B | Q4_K_M | 3.4G / 22G | ⏳ 진행중 | [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) | `curl -L -C - https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf -o ~/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf` |
| Qwen3.5-35B-A3B | Q8_0 | 34G | ✅ | [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) | `hf download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q8_0.gguf --local-dir ~/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF` |
| Qwen3.5-122B-A10B | Q4_K_M | ~72G (3-shard) | ✅ | [unsloth/Qwen3.5-122B-A10B-GGUF](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF) | `hf download unsloth/Qwen3.5-122B-A10B-GGUF --include "Qwen3.5-122B-A10B-Q4_K_M-*.gguf" --local-dir ~/.lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF` |

> **주의**: HF CLI가 stall될 경우 `curl -L -C -` 로 직접 resume. 파일명은 HF repo의 `resolve/main/<filename>` 경로 사용.

---

## MLX

저장 위치: `~/.cache/mlx/`
다운로드 명령: `hf download <repo> --local-dir <dest>`

| 모델 | 양자화 | 크기 | 상태 | HuggingFace | 다운로드 명령 |
|------|--------|------|------|-------------|--------------|
| Qwen3.5-9B | 4bit | 2.3G | ✅ | [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | `hf download mlx-community/Qwen3.5-9B-4bit --local-dir ~/.cache/mlx/Qwen3.5-9B-4bit` |
| Qwen3.5-9B | 8bit | 6.2G | ⏳ 진행중 | [mlx-community/Qwen3.5-9B-8bit](https://huggingface.co/mlx-community/Qwen3.5-9B-8bit) | `hf download mlx-community/Qwen3.5-9B-8bit --local-dir ~/.cache/mlx/Qwen3.5-9B-8bit` |
| Qwen3.5-27B | 4bit | 4.6G | ⏳ 진행중 | [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | `hf download mlx-community/Qwen3.5-27B-4bit --local-dir ~/.cache/mlx/Qwen3.5-27B-4bit` |
| Qwen3.5-27B | 8bit | 9.6G | ⏳ 진행중 | [mlx-community/Qwen3.5-27B-8bit](https://huggingface.co/mlx-community/Qwen3.5-27B-8bit) | `hf download mlx-community/Qwen3.5-27B-8bit --local-dir ~/.cache/mlx/Qwen3.5-27B-8bit` |
| Qwen3.5-35B-A3B | 4bit | 8.5G | ⏳ 진행중 | [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) | `hf download mlx-community/Qwen3.5-35B-A3B-4bit --local-dir ~/.cache/mlx/Qwen3.5-35B-A3B-4bit` |
| Qwen3.5-122B-A10B | 4bit | 18G | ✅ | [mlx-community/Qwen3.5-122B-A10B-4bit](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-4bit) | `hf download mlx-community/Qwen3.5-122B-A10B-4bit --local-dir ~/.cache/mlx/Qwen3.5-122B-A10B-4bit` |

> **비고**: MLX 35B-A3B는 8bit 버전 미제공 (mlx-community). 벤치마크에서 제외.

---

## Ollama

Ollama 라이브러리: [ollama.com/library/qwen3.5](https://ollama.com/library/qwen3.5)

| 태그 | 크기 | 상태 | 다운로드 명령 |
|------|------|------|--------------|
| qwen3.5:9b-q4_K_M | 6.6G | ✅ | `ollama pull qwen3.5:9b-q4_K_M` |
| qwen3.5:9b-q8_0 | 10G | ✅ | `ollama pull qwen3.5:9b-q8_0` |
| qwen3.5:27b-q4_K_M | 17G | ✅ | `ollama pull qwen3.5:27b-q4_K_M` |
| qwen3.5:27b-q8_0 | ~27G | ⏳ 진행중 | `ollama pull qwen3.5:27b-q8_0` |
| qwen3.5:35b-a3b-q4_K_M | 23G | ✅ | `ollama pull qwen3.5:35b-a3b-q4_K_M` |
| qwen3.5:35b-a3b-q8_0 | ~35G | ⏳ 대기 | `ollama pull qwen3.5:35b-a3b-q8_0` |
| qwen3.5:122b-a10b-q4_K_M | ~72G | ⏳ 대기 | `ollama pull qwen3.5:122b-a10b-q4_K_M` |

---

## LM Studio (GUI 로드 완료)

LM Studio API: `http://localhost:1234`
모델 카탈로그: [lmstudio.ai](https://lmstudio.ai) → Search `qwen3.5`

| LM Studio 모델 ID | 상태 |
|-------------------|------|
| `qwen3.5-9b@q4_k_m` | ✅ |
| `qwen3.5-9b@q8_0` | ✅ |
| `qwen3.5-27b@q4_k_m` | ✅ |
| `qwen3.5-27b@q8_0` | ✅ |
| `qwen/qwen3.5-35b-a3b` | ✅ |
| `qwen3.5-122b-a10b` | ✅ |

---

## 전체 현황 요약

| 모델 | GGUF Q4 | GGUF Q8 | MLX 4bit | MLX 8bit | Ollama Q4 | Ollama Q8 |
|------|---------|---------|---------|---------|----------|----------|
| 9B | ✅ | ✅ | ✅ | ⏳ | ✅ | ✅ |
| 27B | ✅ | ✅ | ⏳ | ⏳ | ✅ | ⏳ |
| 35B-A3B | ⏳ | ✅ | ⏳ | — | ✅ | ⏳ |
| 122B-A10B | ✅ | — | ✅ | — | ⏳ | — |

> — : 해당 포맷/양자화 미제공
