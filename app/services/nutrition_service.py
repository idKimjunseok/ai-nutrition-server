from __future__ import annotations

"""오케스트레이션 전담: Gemini/Claude 병렬 호출 후 결과를 합쳐 반환한다.
AI 호출 자체는 app.services.providers, 응답 파싱은 app.services.parsing,
다국어 메시지는 app.services.i18n 에 위임한다."""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException

from app.core.config import Settings
from app.models.schemas import CalorieResult, DailyCalorieRecommendation, DailyCalorieRequest, DailyCalorieResult
from app.services.i18n import CLAUDE_FAILED_MESSAGE, CONSENSUS_MESSAGE, GEMINI_FAILED_MESSAGE, no_food_message, normalize_lang
from app.services.providers import claude as claude_provider
from app.services.providers import gemini as gemini_provider


def _consensus_merge(gemini: CalorieResult, claude: CalorieResult, lang: str = "ko") -> CalorieResult:
    """
    Gemini + Claude 결과를 AI별로 분리하여 반환합니다.
    앱에서 geminiItems / claudeItems 를 각각 표시하고 유저가 원하는 AI 결과를 선택합니다.
    """
    if not gemini.gemini_items and not claude.claude_items:
        message = no_food_message(gemini.image_received, lang)
    else:
        message = CONSENSUS_MESSAGE[normalize_lang(lang)]

    return CalorieResult(
        request_id=str(uuid.uuid4()),
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        image_received=gemini.image_received,  # 이미지 분석이면 True, 텍스트면 False
        image_size_bytes=claude.image_size_bytes or gemini.image_size_bytes,
        message=message,
        gemini_items=gemini.gemini_items,
        claude_items=claude.claude_items,
        gemini_total_calories_kcal=gemini.gemini_total_calories_kcal,
        claude_total_calories_kcal=claude.claude_total_calories_kcal,
        disclaimer=claude.disclaimer,
    )


def _finalize_parallel_results(
    gemini_res: Any,
    claude_res: Any,
    image_received: bool,
    image_size_bytes: Optional[int],
    lang: str = "ko",
) -> CalorieResult:
    """
    Gemini / Claude 병렬 호출 결과를 받아 fallback 처리 후 최종 CalorieResult를 반환합니다.
    image_received: 이미지 분석이면 True, 텍스트 분석이면 False.
    """
    if isinstance(gemini_res, Exception) and isinstance(claude_res, Exception):
        raise HTTPException(
            status_code=502,
            detail=f"Gemini와 Claude 모두 호출에 실패했습니다. Gemini: {gemini_res} / Claude: {claude_res}",
        )

    if isinstance(claude_res, Exception):
        if isinstance(gemini_res, CalorieResult):
            if not gemini_res.gemini_items:
                message = no_food_message(image_received, lang)
            else:
                message = CLAUDE_FAILED_MESSAGE[normalize_lang(lang)]
            return CalorieResult(
                request_id=str(uuid.uuid4()),
                analyzed_at=datetime.now(timezone.utc).isoformat(),
                image_received=image_received,
                image_size_bytes=image_size_bytes,
                message=message,
                gemini_items=gemini_res.gemini_items,
                claude_items=[],
                gemini_total_calories_kcal=gemini_res.gemini_total_calories_kcal,
                claude_total_calories_kcal=None,
                disclaimer=gemini_res.disclaimer,
            )
        raise HTTPException(status_code=502, detail=f"Claude 호출 실패, Gemini 결과도 비정상입니다: {claude_res}")

    if isinstance(gemini_res, Exception):
        if isinstance(claude_res, CalorieResult):
            if not claude_res.claude_items:
                message = no_food_message(image_received, lang)
            else:
                message = GEMINI_FAILED_MESSAGE[normalize_lang(lang)]
            return CalorieResult(
                request_id=str(uuid.uuid4()),
                analyzed_at=datetime.now(timezone.utc).isoformat(),
                image_received=image_received,
                image_size_bytes=image_size_bytes,
                message=message,
                gemini_items=[],
                claude_items=claude_res.claude_items,
                gemini_total_calories_kcal=None,
                claude_total_calories_kcal=claude_res.claude_total_calories_kcal,
                disclaimer=claude_res.disclaimer,
            )
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패, Claude 결과도 비정상입니다: {gemini_res}")

    return _consensus_merge(gemini_res, claude_res, lang)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public API — 이미지 분석
# ---------------------------------------------------------------------------
async def analyze_image_parallel_consensus(
    settings: Settings,
    image_bytes: bytes,
    mime_for_llm: str,
    original_upload_size_bytes: int,
    lang: str = "ko",
) -> CalorieResult:
    """Gemini + Claude 병렬 이미지 분석 후 AI별 결과를 반환합니다."""
    results = await asyncio.gather(
        gemini_provider.analyze_image(settings, image_bytes, mime_for_llm, original_upload_size_bytes, lang),
        claude_provider.analyze_image(settings, image_bytes, mime_for_llm, original_upload_size_bytes, lang),
        return_exceptions=True,
    )
    return _finalize_parallel_results(
        results[0], results[1],
        image_received=True,
        image_size_bytes=original_upload_size_bytes,
        lang=lang,
    )


# ---------------------------------------------------------------------------
# Public API — 텍스트 분석
# ---------------------------------------------------------------------------
async def analyze_text_parallel_consensus(settings: Settings, food_text: str, lang: str = "ko") -> CalorieResult:
    """Gemini + Claude 병렬 텍스트 분석 후 AI별 결과를 반환합니다."""
    results = await asyncio.gather(
        gemini_provider.analyze_text(settings, food_text, lang),
        claude_provider.analyze_text(settings, food_text, lang),
        return_exceptions=True,
    )
    return _finalize_parallel_results(
        results[0], results[1],
        image_received=False,
        image_size_bytes=None,
        lang=lang,
    )


# ---------------------------------------------------------------------------
# Public API — 하루 권장 칼로리
# ---------------------------------------------------------------------------
async def analyze_daily_calories_parallel_consensus(
    settings: Settings,
    request: DailyCalorieRequest,
) -> DailyCalorieResult:
    """Gemini + Claude 병렬 분석 후 하루 권장 칼로리 평균값만 반환합니다."""
    results = await asyncio.gather(
        gemini_provider.analyze_daily_calories(settings, request),
        claude_provider.analyze_daily_calories(settings, request),
        return_exceptions=True,
    )
    gemini_res, claude_res = results[0], results[1]

    if isinstance(gemini_res, Exception) and isinstance(claude_res, Exception):
        raise HTTPException(
            status_code=502,
            detail=f"Gemini와 Claude 모두 호출에 실패했습니다. Gemini: {gemini_res} / Claude: {claude_res}",
        )

    calorie_values = [
        result.daily_calories_kcal
        for result in (gemini_res, claude_res)
        if isinstance(result, DailyCalorieRecommendation)
    ]
    average_daily_calories = sum(calorie_values) / len(calorie_values)

    return DailyCalorieResult(daily_calories_kcal=round(average_daily_calories, 1))
