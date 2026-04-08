"""벤치마크용 프롬프트 생성 — 실제 tokenizer 기반 정확한 토큰 수.

v3: 정확한 토큰 길이 + cold prefill 보장
- tokenizer로 정확히 target_tokens 길이의 프롬프트 생성
- 매 호출마다 랜덤 nonce prefix → cache hit 불가
- --no-cache-prompt + --slot-prompt-similarity 0 과 함께 사용
"""
import random
import string
from typing import Optional

# tokenizer 캐시 (모델별로 한 번만 로드)
_tokenizer_cache: dict = {}

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

# 생성 트랙 꼬리 지시문
_TAIL_NUMBERED = """\
Write a comprehensive technical FAQ covering the history, architecture, \
performance, and applications of computing systems. \
Number each question and provide detailed multi-sentence answers with examples. \
Do not stop early — continue answering until the response is complete. Question 1:"""


def _nonce(length: int = 32) -> str:
    """랜덤 nonce — prefix cache / reuse 방지."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _get_tokenizer(model_id: Optional[str] = None):
    """Qwen3.5 tokenizer 로드 (캐시됨). 없으면 근사 모드."""
    if model_id and model_id in _tokenizer_cache:
        return _tokenizer_cache[model_id]

    try:
        from transformers import AutoTokenizer
        # Qwen3.5 계열 공통 tokenizer
        tok_id = model_id or "Qwen/Qwen3.5-9B"
        tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
        if model_id:
            _tokenizer_cache[model_id] = tok
        return tok
    except Exception:
        return None


def _count_tokens(text: str, tokenizer=None) -> int:
    """토큰 수 계산. tokenizer 없으면 근사."""
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(text) / 3.8)


def _build_text_exact(target_tokens: int, suffix: str, tokenizer=None) -> str:
    """정확히 target_tokens 길이의 프롬프트 생성.

    1) nonce prefix + base text 반복으로 초과 생성
    2) tokenizer로 정확히 target_tokens에 맞게 절단
    3) tokenizer 없으면 문자 수 근사 폴백
    """
    nonce = _nonce()
    prefix = f"[run:{nonce}] "

    suffix_tokens = _count_tokens(suffix, tokenizer) if suffix else 0
    body_target = target_tokens - suffix_tokens

    if tokenizer:
        # tokenizer 기반: 정확한 토큰 수
        prefix_tokens = _count_tokens(prefix, tokenizer)
        text_target = body_target - prefix_tokens

        # 충분히 긴 텍스트 생성
        chars_estimate = int(text_target * 4.5)  # 여유있게
        raw = (_BASE_TEXT * ((chars_estimate // len(_BASE_TEXT)) + 2))[:chars_estimate]

        # encode → 정확히 text_target 토큰만 취 → decode
        tokens = tokenizer.encode(raw, add_special_tokens=False)[:text_target]
        body = tokenizer.decode(tokens, skip_special_tokens=True)

        result = prefix + body
        if suffix:
            result += "\n\n" + suffix

        # 최종 검증
        actual = _count_tokens(result, tokenizer)
        if abs(actual - target_tokens) > 2:
            # 미세 조정 (1-2 토큰 오차까지)
            tokens = tokenizer.encode(result, add_special_tokens=False)[:target_tokens]
            result = tokenizer.decode(tokens, skip_special_tokens=True)

        return result
    else:
        # 근사 폴백
        chars_per_token = 3.8
        prefix_chars = len(prefix)
        suffix_chars = len(suffix) + 2 if suffix else 0
        text_chars = max(0, int(body_target * chars_per_token) - prefix_chars)
        raw = (_BASE_TEXT * ((text_chars // len(_BASE_TEXT)) + 2))[:text_chars]
        result = prefix + raw
        if suffix:
            result += "\n\n" + suffix
        return result


def init_tokenizer(model_id: Optional[str] = None) -> None:
    """모델 로드 전에 tokenizer를 미리 초기화. runner에서 호출."""
    _get_tokenizer(model_id)


def build_prefill_prompt(target_tokens: int, model_id: Optional[str] = None) -> str:
    """정확히 target_tokens 길이의 프리필 프롬프트.

    매 호출마다 다른 nonce prefix → cold prefill 보장.
    """
    tokenizer = _get_tokenizer(model_id)
    suffix = "Based on the above text, what is the main topic discussed? Answer in one word:"
    return _build_text_exact(target_tokens, suffix, tokenizer)


def build_generation_prompt(target_tokens: int, max_tokens: int = 512,
                            model_id: Optional[str] = None) -> str:
    """정확히 target_tokens 길이의 생성 프롬프트.

    매 호출마다 다른 nonce prefix → cold prefill 보장.
    """
    tokenizer = _get_tokenizer(model_id)
    return _build_text_exact(target_tokens, _TAIL_NUMBERED, tokenizer)
