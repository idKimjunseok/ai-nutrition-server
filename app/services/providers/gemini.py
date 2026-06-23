from __future__ import annotations

"""Google Gemini 호출 전담 모듈 — 클라이언트 생성, 요청 조립, 응답 텍스트 추출까지만 담당한다.
결과 파싱은 app.services.parsing, 메시지/프롬프트는 i18n·prompts 모듈에 위임한다."""

import base64
import json
import logging
from typing import Any, Dict

from fastapi import HTTPException

from app.core.config import Settings
from app.models.schemas import CalorieResult, DailyCalorieRecommendation, DailyCalorieRequest
from app.services.parsing import llm_text_to_calorie_result, llm_text_to_daily_calorie_recommendation
from app.services.prompts.daily_calorie_prompt import build_daily_calorie_prompt
from app.services.prompts.image_prompt import select_image_prompt
from app.services.prompts.text_prompt import build_text_prompt
from app.services.providers.common import ainvoke_with_timeout, message_content_to_text

logger = logging.getLogger(__name__)

try:
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:  # pragma: no cover
    HumanMessage = None  # type: ignore[misc, assignment]
    ChatGoogleGenerativeAI = None  # type: ignore[misc, assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _ensure_available(settings: Settings) -> None:
    if _IMPORT_ERROR is not None or ChatGoogleGenerativeAI is None or HumanMessage is None:
        raise HTTPException(status_code=500, detail=f"langchain-google-genai를 불러올 수 없습니다: {_IMPORT_ERROR}")
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY가 설정되지 않았습니다.")


def _build_client(settings: Settings) -> "ChatGoogleGenerativeAI":
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model_id,
        api_key=settings.google_api_key,
        temperature=0.2,
        timeout=15,
        max_retries=0,
    )


def _image_url_block(mime_type: str, image_bytes: bytes) -> Dict[str, Any]:
    """
    langchain_google_genai는 `type: image` 블록을 거부하는 버전이 있어,
    OpenAI 호환 블록인 image_url + data URI 를 사용합니다.
    """
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    data_uri = f"data:{mime_type};base64,{b64}"
    return {"type": "image_url", "image_url": {"url": data_uri}}


async def analyze_image(settings: Settings, image_bytes: bytes, mime_for_llm: str, upload_size: int, lang: str = "ko") -> CalorieResult:
    _ensure_available(settings)

    prompt_text = select_image_prompt(settings, lang)
    message = HumanMessage(content=[{"type": "text", "text": prompt_text}, _image_url_block(mime_for_llm, image_bytes)])
    try:
        response = await ainvoke_with_timeout(_build_client(settings), message)
    except Exception as exc:
        logger.exception("Gemini 호출 실패")
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {exc}") from exc

    text = message_content_to_text(response.content)
    try:
        return llm_text_to_calorie_result(upload_size, text, source="gemini", lang=lang)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Gemini 응답 JSON 파싱 실패: %s", text[:500])
        raise HTTPException(status_code=502, detail=f"Gemini 응답을 CalorieResult로 변환 실패: {exc}") from exc


async def analyze_text(settings: Settings, food_text: str, lang: str = "ko") -> CalorieResult:
    _ensure_available(settings)

    message = HumanMessage(content=[{"type": "text", "text": build_text_prompt(food_text, lang)}])
    try:
        response = await ainvoke_with_timeout(_build_client(settings), message)
    except Exception as exc:
        logger.exception("Gemini 텍스트 분석 호출 실패")
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {exc}") from exc

    text = message_content_to_text(response.content)
    try:
        return llm_text_to_calorie_result(None, text, source="gemini", lang=lang)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Gemini 텍스트 응답 JSON 파싱 실패: %s", text[:500])
        raise HTTPException(status_code=502, detail=f"Gemini 응답 변환 실패: {exc}") from exc


async def analyze_daily_calories(settings: Settings, request: DailyCalorieRequest) -> DailyCalorieRecommendation:
    _ensure_available(settings)

    message = HumanMessage(content=[{"type": "text", "text": build_daily_calorie_prompt(request)}])
    try:
        response = await ainvoke_with_timeout(_build_client(settings), message)
    except Exception as exc:
        logger.exception("Gemini 하루 권장 칼로리 분석 호출 실패")
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {exc}") from exc

    text = message_content_to_text(response.content)
    try:
        return llm_text_to_daily_calorie_recommendation(text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Gemini 하루 권장 칼로리 응답 JSON 파싱 실패: %s", text[:500])
        raise HTTPException(status_code=502, detail=f"Gemini 응답 변환 실패: {exc}") from exc
