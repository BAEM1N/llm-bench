"""벤치마크용 프롬프트 생성 — 토큰 수 기반."""

# 약 1 토큰 ≈ 4 영문 글자 (tiktoken 기준 근사)
# 실제 토큰 수는 모델/토크나이저마다 다름 — ±10% 오차 가정.
# input_tokens는 명목값이며, 꼬리 지시문(~45 tok)이 고정 추가되므로
# 텍스트 본문은 target_tokens - 45 tokens 정도로 짧아진다.
# 백엔드 간 prefill_tps 비교는 반드시 prefill_tps_source="native" 컬럼으로 필터링할 것.
CHARS_PER_TOKEN = 3.8

# 베이스 텍스트 (반복하여 목표 길이 달성)
_BASE_TEXT = """\
The history of computing spans several decades and has transformed nearly every aspect of human civilization. \
From early mechanical calculators to modern quantum computers, the evolution of computation represents one of \
humanity's greatest intellectual achievements. Early pioneers like Charles Babbage and Ada Lovelace laid the \
conceptual groundwork for programmable machines in the 19th century. The 20th century saw rapid advancement \
with vacuum tube computers, transistors, integrated circuits, and microprocessors. Each generation of hardware \
brought exponential improvements in speed, efficiency, and miniaturization. Software development evolved in \
parallel, from machine code and assembly language to high-level programming languages, operating systems, \
and the vast ecosystem of applications we rely on today. The internet connected billions of devices and people, \
enabling unprecedented information exchange and collaboration. Artificial intelligence and machine learning have \
emerged as transformative technologies, enabling computers to perform tasks that once required human intelligence. \
"""

# 생성 트랙 꼬리 지시문 — 모든 트랙에서 동일한 번호 매기기 형식 사용
# 일관된 태스크 구조로 트랙 간 공정 비교 보장.
# 짧은 트랙은 일부 질문만 답하고 끝남 — 이것이 바로 측정 목표.
_TAIL_NUMBERED = """\
Write a comprehensive technical FAQ covering the history, architecture, \
performance, and applications of computing systems. \
Number each question and provide detailed multi-sentence answers with examples. \
Do not stop early — continue answering until the response is complete. Question 1:"""

_TAIL_NUMBERED_TOKENS = 45  # 꼬리 근사 토큰 수


def _pick_tail(max_tokens: int) -> tuple:
    """꼬리 지시문과 그 근사 토큰 수 반환.

    모든 트랙에 동일한 numbered 형식을 사용하여 태스크 구조를 통일.
    max_tokens 인자는 하위 호환성을 위해 유지하나 현재는 사용하지 않음.
    """
    return _TAIL_NUMBERED, _TAIL_NUMBERED_TOKENS


def build_prefill_prompt(target_tokens: int) -> str:
    """목표 토큰 수에 맞는 프리필 프롬프트 생성. 마지막에 짧은 질문 추가."""
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    repeated = (_BASE_TEXT * ((target_chars // len(_BASE_TEXT)) + 2))[:target_chars]
    return repeated + "\n\nBased on the above text, what is the main topic discussed? Answer in one word:"


def build_generation_prompt(target_tokens: int, max_tokens: int = 512) -> str:
    """생성 트랙 프롬프트: target_tokens 근사 입력, 긴 출력 유도.

    max_tokens에 따라 꼬리 지시문 강도를 조정한다.
    4096+ 트랙은 번호 매기기 형식으로 모델의 조기 EOS를 방지.
    """
    tail, tail_tok = _pick_tail(max_tokens)
    tail_chars = int(tail_tok * CHARS_PER_TOKEN)
    text_chars = max(0, int(target_tokens * CHARS_PER_TOKEN) - tail_chars - 2)
    repeated = (_BASE_TEXT * ((text_chars // len(_BASE_TEXT)) + 2))[:text_chars]
    return repeated + "\n\n" + tail
