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

# 생성 트랙 출력 유도용 꼬리 지시문 (~15 tok)
_GENERATION_TAIL = "Continue writing the next detailed technical section at length:"


def build_prefill_prompt(target_tokens: int) -> str:
    """목표 토큰 수에 맞는 프리필 프롬프트 생성. 마지막에 짧은 질문 추가."""
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    repeated = (_BASE_TEXT * ((target_chars // len(_BASE_TEXT)) + 2))[:target_chars]
    return repeated + "\n\nBased on the above text, what is the main topic discussed? Answer in one word:"


def build_generation_prompt(target_tokens: int) -> str:
    """생성 트랙 프롬프트: target_tokens 근사 입력, 긴 출력 유도.

    꼬리 지시문(~15 tok)을 제외한 나머지를 _BASE_TEXT로 채운다.
    이렇게 하면 config의 input_tokens가 실제 입력 길이를 제어한다.
    """
    tail_chars = int(15 * CHARS_PER_TOKEN)
    text_chars = max(0, int(target_tokens * CHARS_PER_TOKEN) - tail_chars - 2)
    repeated = (_BASE_TEXT * ((text_chars // len(_BASE_TEXT)) + 2))[:text_chars]
    return repeated + "\n\n" + _GENERATION_TAIL
