"""벤치마크용 프롬프트 생성 — 토큰 수 기반."""

# 약 1 토큰 ≈ 4 영문 글자 (tiktoken 기준 근사)
# 실제 토큰 수는 모델/토크나이저마다 다름 — 여기선 근사값 사용
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

# 생성 트랙 꼬리 지시문 — max_tokens에 따라 선택
# 짧은 트랙(≤2048): 자유 서술로 충분
# 긴 트랙(>2048): 번호 매기기 형식으로 모델이 EOS를 내기 어렵게 유도
_TAIL_SHORT = "Continue writing the next detailed technical section at length:"

_TAIL_4096 = """\
Write a comprehensive technical FAQ with detailed answers covering the history, \
architecture, performance, and future of computing systems. \
Number each question and provide thorough multi-sentence answers with examples. \
Do not summarize or stop until all questions are fully answered. Question 1:"""

_TAIL_8192 = """\
Write a comprehensive technical encyclopedia with detailed entries. \
For each entry write a complete explanation covering definition, historical context, \
technical details, practical applications, and current research directions. \
Number every entry and do not stop or summarize until the encyclopedia is complete. Entry 1:"""


def _pick_tail(max_tokens: int) -> tuple:
    """꼬리 지시문과 그 근사 토큰 수 반환."""
    if max_tokens <= 2048:
        return _TAIL_SHORT, 15
    if max_tokens <= 4096:
        return _TAIL_4096, 45
    return _TAIL_8192, 45


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
