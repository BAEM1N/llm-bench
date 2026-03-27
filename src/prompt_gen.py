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

# 생성 트랙용 프롬프트 (짧은 입력, 긴 출력 유도)
GENERATION_PROMPTS = {
    "gen-512":  "Write a detailed technical explanation of how transformer neural networks work, covering attention mechanisms, positional encoding, and training procedures. Be thorough and precise.",
    "gen-2048": "Write a comprehensive technical guide on building a distributed systems architecture, covering consistency models, consensus algorithms, fault tolerance, load balancing, and monitoring. Include concrete examples and tradeoffs.",
    "gen-4096": "Write an exhaustive textbook chapter on operating system design, covering process management, memory management, file systems, I/O subsystems, scheduling algorithms, synchronization primitives, and security. Include pseudocode examples and detailed explanations.",
    "gen-8192": "Write a complete, detailed reference manual on database internals, covering storage engines, B-tree and LSM-tree structures, transaction processing, MVCC, query optimization, indexing strategies, replication, sharding, and recovery mechanisms. Be exhaustive and technically precise.",
}


def build_prefill_prompt(target_tokens: int) -> str:
    """목표 토큰 수에 맞는 프리필 프롬프트 생성. 마지막에 짧은 질문 추가."""
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    # 베이스 텍스트 반복
    repeated = (_BASE_TEXT * ((target_chars // len(_BASE_TEXT)) + 2))[:target_chars]
    return repeated + "\n\nBased on the above text, what is the main topic discussed? Answer in one word:"


def build_generation_prompt(track_id: str) -> str:
    """생성 트랙 프롬프트 반환."""
    return GENERATION_PROMPTS.get(track_id, GENERATION_PROMPTS["gen-2048"])
