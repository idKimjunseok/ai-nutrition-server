from __future__ import annotations

"""AI 프로바이더(Gemini/Claude) 공통 호출 유틸리티."""

import asyncio
from typing import Any, List, Union

LLM_CALL_TIMEOUT_SECONDS = 15.0


def message_content_to_text(content: Union[str, List[Any]]) -> str:
    """AIMessage.content가 문자열 또는 멀티파트 블록 리스트일 때 모두 문자열로 합칩니다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)


async def ainvoke_with_timeout(llm: Any, message: Any, timeout: float = LLM_CALL_TIMEOUT_SECONDS) -> Any:
    """
    LLM 호출에 하드 타임아웃을 건다.
    langchain/google-api-core 내부 재시도가 매우 길게(수십~100초+) 걸릴 수 있고,
    ainvoke가 내부적으로 동기 호출을 포함해 이벤트 루프를 블로킹할 수 있어
    클라이언트의 timeout/max_retries 설정만으로는 신뢰할 수 없다.
    별도 스레드(asyncio.to_thread)에서 동기 invoke를 실행해 이벤트 루프를 막지 않고
    asyncio.wait_for로 상한선을 강제한다.
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(llm.invoke, [message]), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{timeout:.0f}초 내에 응답을 받지 못했습니다 (timeout)") from exc
