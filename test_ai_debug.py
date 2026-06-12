"""
각 AI(Gemini / Claude)의 원본 응답과 파싱 결과를 진단하는 스크립트.
사용법:  python test_ai_debug.py <이미지_경로>
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import load_settings
from app.services.nutrition_service import (
    _analyze_with_claude,
    _analyze_with_gemini,
    _parse_json_from_llm,
    _normalize_food_dicts,
    _message_content_to_text,
    _strip_json_fence,
    _lc_image_url_block,
    _anthropic_image_block,
)
from app.utils.image_utils import prepare_image_for_llm


SEP = "=" * 70


async def test_raw_gemini(settings, image_bytes: bytes, mime: str) -> str:
    """Gemini 원본 텍스트 응답만 반환."""
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    message = HumanMessage(content=[
        {"type": "text", "text": settings.prompt},
        _lc_image_url_block(mime, image_bytes),
    ])
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model_id,
        api_key=settings.google_api_key,
        temperature=0.2,
    )
    response = await llm.ainvoke([message])
    return _message_content_to_text(response.content)


async def test_raw_claude(settings, image_bytes: bytes, mime: str) -> str:
    """Claude 원본 텍스트 응답만 반환."""
    from langchain_core.messages import HumanMessage
    from langchain_anthropic import ChatAnthropic

    message = HumanMessage(content=[
        {"type": "text", "text": settings.prompt},
        _anthropic_image_block(mime, image_bytes),
    ])
    llm = ChatAnthropic(
        model=settings.claude_model_id,
        api_key=settings.anthropic_api_key,
        temperature=0.2,
    )
    response = await llm.ainvoke([message])
    return _message_content_to_text(response.content)


def diagnose_parse(label: str, raw_text: str) -> None:
    print(f"\n[{label}] 원본 응답 텍스트:")
    print(raw_text[:2000])

    print(f"\n[{label}] JSON 펜스 제거 후:")
    stripped = _strip_json_fence(raw_text)
    print(stripped[:2000])

    print(f"\n[{label}] JSON 파싱 결과:")
    try:
        parsed = _parse_json_from_llm(raw_text)
        print(json.dumps(parsed, ensure_ascii=False, indent=2)[:2000])
    except ValueError as e:
        print(f"  ❌ 파싱 실패: {e}")
        return

    print(f"\n[{label}] 음식 dict 정규화:")
    try:
        food_dicts = _normalize_food_dicts(parsed)
        print(json.dumps(food_dicts, ensure_ascii=False, indent=2)[:2000])
        print(f"  → 항목 수: {len(food_dicts)}")
        for i, fd in enumerate(food_dicts):
            print(f"  항목[{i}] 키 목록: {list(fd.keys())}")
    except ValueError as e:
        print(f"  ❌ 정규화 실패: {e}")


async def main(image_path: str) -> None:
    settings = load_settings()
    raw = Path(image_path).read_bytes()
    image_bytes, mime = prepare_image_for_llm(raw, None, image_path)

    print(SEP)
    print(f"이미지: {image_path}  ({len(raw):,} bytes, mime={mime})")
    print(f"Gemini 모델: {settings.gemini_model_id}")
    print(f"Claude 모델: {settings.claude_model_id}")
    print(f"프롬프트: {settings.prompt}")
    print(SEP)

    # ── Gemini 테스트 ────────────────────────────────────────────────────────
    print("\n★ GEMINI 테스트")
    try:
        gemini_text = await test_raw_gemini(settings, image_bytes, mime)
        diagnose_parse("GEMINI", gemini_text)
    except Exception as e:
        print(f"  ❌ Gemini 호출 자체 실패: {type(e).__name__}: {e}")

    print("\n" + SEP)

    # ── Claude 테스트 ────────────────────────────────────────────────────────
    print("\n★ CLAUDE 테스트")
    try:
        claude_text = await test_raw_claude(settings, image_bytes, mime)
        diagnose_parse("CLAUDE", claude_text)
    except Exception as e:
        print(f"  ❌ Claude 호출 자체 실패: {type(e).__name__}: {e}")

    print("\n" + SEP)

    # ── 최종 합의 결과 테스트 ────────────────────────────────────────────────
    print("\n★ 최종 합의(consensus) 결과 테스트")
    from app.services.nutrition_service import analyze_image_parallel_consensus
    try:
        result = await analyze_image_parallel_consensus(
            settings, image_bytes, mime, len(raw)
        )
        print(json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"  ❌ 합의 결과 실패: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_ai_debug.py <이미지_경로>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
