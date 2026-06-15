from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from fastapi import HTTPException

from app.core.config import Settings
from app.models.schemas import (
    CalorieResult,
    DailyCalorieRecommendation,
    DailyCalorieRequest,
    DailyCalorieResult,
    DetectedFoodItem,
    Macros,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangChain imports (optional at import time)
# ---------------------------------------------------------------------------
try:
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:  # pragma: no cover
    HumanMessage = None  # type: ignore[misc, assignment]
    ChatGoogleGenerativeAI = None  # type: ignore[misc, assignment]
    _LANGCHAIN_GOOGLE_IMPORT_ERROR = e
else:
    _LANGCHAIN_GOOGLE_IMPORT_ERROR = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError as e:  # pragma: no cover
    ChatAnthropic = None  # type: ignore[misc, assignment]
    _LANGCHAIN_ANTHROPIC_IMPORT_ERROR = e
else:
    _LANGCHAIN_ANTHROPIC_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Parsing helpers (LLM text -> JSON -> CalorieResult)
# ---------------------------------------------------------------------------
def _message_content_to_text(content: Union[str, List[Any]]) -> str:
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


def _strip_json_fence(text: str) -> str:
    """모델이 ```json ... ``` 로 감싼 경우 내용만 추출합니다."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text


def _parse_json_from_llm(text: str) -> Any:
    """응답 문자열에서 JSON(객체 또는 배열)을 파싱합니다."""
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(open_ch), cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("LLM 응답에서 JSON을 파싱할 수 없습니다.")


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = re.sub(r"[^\d.\-]", "", val.replace(",", ""))
        if s in ("", "-", "."):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _get_str(d: Dict[str, Any], *keys: str) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return str(d[k]).strip()
    lower_map = {str(kk).lower(): kk for kk in d}
    for k in keys:
        lk = k.lower()
        if lk in lower_map:
            v = d[lower_map[lk]]
            if v is not None:
                return str(v).strip()
    return None


def _get_float(d: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            f = _to_float(d[k])
            if f is not None:
                return f
    lower_map = {str(kk).lower(): kk for kk in d}
    for k in keys:
        lk = k.lower()
        if lk in lower_map:
            f = _to_float(d[lower_map[lk]])
            if f is not None:
                return f
    return None


def _normalize_food_dicts(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if not isinstance(parsed, dict):
        raise ValueError("JSON 루트는 객체 또는 배열이어야 합니다.")
    if not parsed:
        return []
    for key in ("foods", "items", "results", "dishes"):
        inner = parsed.get(key)
        if isinstance(inner, list) and all(isinstance(x, dict) for x in inner):
            return inner
    return [parsed]


def _dict_to_detected_item(d: Dict[str, Any], default_confidence: float = 0.85) -> DetectedFoodItem:
    food_name = _get_str(d, "food_name", "name", "food", "음식", "음식이름", "identified_food") or "식별 불가"
    calories = _get_float(d, "calories_kcal", "calories", "kcal", "calorie", "칼로리", "estimated_calories")
    if calories is None:
        calories = 0.0

    carb = _get_float(
        d,
        "carbohydrates_g",
        "carbohydrate_g",
        "carbs_g",
        "carb_g",
        "탄수화물_g",
        "carbohydrates",
        "탄수화물",
    )
    protein = _get_float(d, "protein_g", "protein", "단백질_g", "단백질")
    fat = _get_float(d, "fat_g", "fat", "지방_g", "지방")

    conf = _get_float(d, "confidence", "신뢰도", "score")
    confidence = default_confidence if conf is None else max(0.0, min(1.0, conf))

    portion = _get_str(d, "portion_description", "portion", "serving", "분량", "인분") or "예상 1인분 기준 (LLM 추정)"

    return DetectedFoodItem(
        food_name=food_name,
        confidence=confidence,
        portion_description=portion,
        calories_kcal=float(calories),
        macros=Macros(carbohydrate_g=carb, protein_g=protein, fat_g=fat),
    )


def _llm_text_to_calorie_result(image_size_bytes: Optional[int], llm_text: str, source: str) -> CalorieResult:
    """
    LLM 텍스트 응답을 CalorieResult로 변환합니다.
    source: "gemini" 또는 "claude" — 어느 AI의 결과인지 구분하여 각 AI 전용 필드에 저장합니다.
    """
    parsed = _parse_json_from_llm(llm_text)
    food_dicts = _normalize_food_dicts(parsed)
    items = [_dict_to_detected_item(fd) for fd in food_dicts]
    total = sum(i.calories_kcal for i in items) if items else None
    is_image = image_size_bytes is not None
    return CalorieResult(
        request_id=str(uuid.uuid4()),
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        image_received=is_image,
        image_size_bytes=image_size_bytes,
        gemini_items=items if source == "gemini" else [],
        claude_items=items if source == "claude" else [],
        gemini_total_calories_kcal=total if source == "gemini" else None,
        claude_total_calories_kcal=total if source == "claude" else None,
        disclaimer=(
            "본 결과는 생성형 AI의 추정치이며 의학적·영양학적 진단 또는 개인별 식단 지침을 "
            "대체하지 않습니다."
        ),
    )


# ---------------------------------------------------------------------------
# Consensus helpers
# ---------------------------------------------------------------------------
def _no_food_message(image_received: bool) -> str:
    """Gemini/Claude 둘 다 음식을 찾지 못했을 때 사용자에게 보여줄 메시지."""
    if image_received:
        return "이미지에서 음식을 찾을 수 없습니다. 음식이 잘 보이는 사진으로 다시 촬영해주세요."
    return "입력한 텍스트에서 음식 정보를 찾을 수 없습니다. 음식 이름을 다시 확인해주세요."


def _consensus_merge(gemini: CalorieResult, claude: CalorieResult) -> CalorieResult:
    """
    Gemini + Claude 결과를 AI별로 분리하여 반환합니다.
    앱에서 geminiItems / claudeItems 를 각각 표시하고 유저가 원하는 AI 결과를 선택합니다.
    """
    if not gemini.gemini_items and not claude.claude_items:
        message = _no_food_message(gemini.image_received)
    else:
        message = "Gemini와 Claude 다중 AI 교차 검증이 완료되었습니다."

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


# ---------------------------------------------------------------------------
# Multimodal message blocks
# ---------------------------------------------------------------------------
def _lc_image_url_block(mime_type: str, image_bytes: bytes) -> Dict[str, Any]:
    """
    langchain_google_genai는 `type: image` 블록을 거부하는 버전이 있어,
    OpenAI 호환 블록인 image_url + data URI 를 사용합니다.
    """
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    data_uri = f"data:{mime_type};base64,{b64}"
    return {"type": "image_url", "image_url": {"url": data_uri}}


def _anthropic_image_block(mime_type: str, image_bytes: bytes) -> Dict[str, Any]:
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}}


# ---------------------------------------------------------------------------
# AI 호출 공통 타임아웃
# ---------------------------------------------------------------------------
LLM_CALL_TIMEOUT_SECONDS = 15.0


async def _ainvoke_with_timeout(llm: Any, message: "HumanMessage", timeout: float = LLM_CALL_TIMEOUT_SECONDS) -> Any:
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


# ---------------------------------------------------------------------------
# Public service API
# ---------------------------------------------------------------------------
async def _analyze_with_gemini(settings: Settings, image_bytes: bytes, mime_for_llm: str, upload_size: int) -> CalorieResult:
    if _LANGCHAIN_GOOGLE_IMPORT_ERROR is not None or ChatGoogleGenerativeAI is None or HumanMessage is None:
        raise HTTPException(status_code=500, detail=f"langchain-google-genai를 불러올 수 없습니다: {_LANGCHAIN_GOOGLE_IMPORT_ERROR}")
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY가 설정되지 않았습니다.")

    message = HumanMessage(content=[{"type": "text", "text": settings.prompt}, _lc_image_url_block(mime_for_llm, image_bytes)])
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model_id,
        api_key=settings.google_api_key,
        temperature=0.2,
        timeout=15,
        max_retries=0,
    )
    try:
        response = await _ainvoke_with_timeout(llm, message)
    except Exception as exc:
        logger.exception("Gemini 호출 실패")
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {exc}") from exc

    text = _message_content_to_text(response.content)
    try:
        return _llm_text_to_calorie_result(upload_size, text, source="gemini")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Gemini 응답 JSON 파싱 실패: %s", text[:500])
        raise HTTPException(status_code=502, detail=f"Gemini 응답을 CalorieResult로 변환 실패: {exc}") from exc


async def _analyze_with_claude(settings: Settings, image_bytes: bytes, mime_for_llm: str, upload_size: int) -> CalorieResult:
    if _LANGCHAIN_ANTHROPIC_IMPORT_ERROR is not None or ChatAnthropic is None or HumanMessage is None:
        raise HTTPException(status_code=500, detail=f"langchain-anthropic을 불러올 수 없습니다: {_LANGCHAIN_ANTHROPIC_IMPORT_ERROR}")
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    message = HumanMessage(content=[{"type": "text", "text": settings.prompt}, _anthropic_image_block(mime_for_llm, image_bytes)])
    llm = ChatAnthropic(
        model=settings.claude_model_id,
        api_key=settings.anthropic_api_key,
        temperature=0.2,
        timeout=15,
        max_retries=0,
    )
    try:
        response = await _ainvoke_with_timeout(llm, message)
    except Exception as exc:
        logger.exception("Claude 호출 실패")
        raise HTTPException(status_code=502, detail=f"Claude 호출 실패: {exc}") from exc

    text = _message_content_to_text(response.content)
    try:
        return _llm_text_to_calorie_result(upload_size, text, source="claude")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Claude 응답 JSON 파싱 실패: %s", text[:500])
        raise HTTPException(status_code=502, detail=f"Claude 응답을 CalorieResult로 변환 실패: {exc}") from exc


# ---------------------------------------------------------------------------
# 공통 병렬 결과 처리 (이미지·텍스트 공유)
# ---------------------------------------------------------------------------
def _finalize_parallel_results(
    gemini_res: Any,
    claude_res: Any,
    image_received: bool,
    image_size_bytes: Optional[int],
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
                message = _no_food_message(image_received)
            else:
                message = "Claude 호출이 실패하여 Gemini 단독 결과로 반환합니다."
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
                message = _no_food_message(image_received)
            else:
                message = "Gemini 호출이 실패하여 Claude 단독 결과로 반환합니다."
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

    return _consensus_merge(gemini_res, claude_res)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public API — 이미지 분석
# ---------------------------------------------------------------------------
async def analyze_image_parallel_consensus(
    settings: Settings,
    image_bytes: bytes,
    mime_for_llm: str,
    original_upload_size_bytes: int,
) -> CalorieResult:
    """Gemini + Claude 병렬 이미지 분석 후 AI별 결과를 반환합니다."""
    results = await asyncio.gather(
        _analyze_with_gemini(settings, image_bytes, mime_for_llm, original_upload_size_bytes),
        _analyze_with_claude(settings, image_bytes, mime_for_llm, original_upload_size_bytes),
        return_exceptions=True,
    )
    return _finalize_parallel_results(
        results[0], results[1],
        image_received=True,
        image_size_bytes=original_upload_size_bytes,
    )


# ---------------------------------------------------------------------------
# 텍스트 분석 — LLM 호출
# ---------------------------------------------------------------------------
def _build_text_prompt(food_text: str) -> str:
    """텍스트 음식명 분석용 프롬프트를 생성합니다."""
    return (
        "다음 음식 목록의 각 항목에 대해 예상 1인분 기준 칼로리와 영양소를 분석해서 "
        "아래 JSON 배열 형식으로만 응답해. 다른 설명 텍스트는 절대 포함하지 마. "
        "음식이 쉼표나 공백으로 구분되어 있으면 각각 별도 항목으로 처리해. "
        "입력에 음식으로 인식할 수 있는 내용이 전혀 없으면 빈 배열 []로만 응답해. "
        "모든 숫자 값은 단위 없이 숫자만 입력해.\n\n"
        f"음식: {food_text}\n\n"
        "[\n"
        "  {\n"
        '    "food_name": "음식 이름",\n'
        '    "calories_kcal": 칼로리_숫자,\n'
        '    "portion_description": "예상 1인분 기준",\n'
        '    "carbs_g": 탄수화물_그램_숫자,\n'
        '    "protein_g": 단백질_그램_숫자,\n'
        '    "fat_g": 지방_그램_숫자,\n'
        '    "confidence": 0.0~1.0_신뢰도_숫자\n'
        "  }\n"
        "]"
    )


async def _analyze_text_with_gemini(settings: Settings, food_text: str) -> CalorieResult:
    if _LANGCHAIN_GOOGLE_IMPORT_ERROR is not None or ChatGoogleGenerativeAI is None or HumanMessage is None:
        raise HTTPException(status_code=500, detail=f"langchain-google-genai를 불러올 수 없습니다: {_LANGCHAIN_GOOGLE_IMPORT_ERROR}")
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY가 설정되지 않았습니다.")

    prompt = _build_text_prompt(food_text)
    message = HumanMessage(content=[{"type": "text", "text": prompt}])
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model_id,
        api_key=settings.google_api_key,
        temperature=0.2,
        timeout=15,
        max_retries=0,
    )
    try:
        response = await _ainvoke_with_timeout(llm, message)
    except Exception as exc:
        logger.exception("Gemini 텍스트 분석 호출 실패")
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {exc}") from exc

    text = _message_content_to_text(response.content)
    try:
        return _llm_text_to_calorie_result(None, text, source="gemini")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Gemini 텍스트 응답 JSON 파싱 실패: %s", text[:500])
        raise HTTPException(status_code=502, detail=f"Gemini 응답 변환 실패: {exc}") from exc


async def _analyze_text_with_claude(settings: Settings, food_text: str) -> CalorieResult:
    if _LANGCHAIN_ANTHROPIC_IMPORT_ERROR is not None or ChatAnthropic is None or HumanMessage is None:
        raise HTTPException(status_code=500, detail=f"langchain-anthropic을 불러올 수 없습니다: {_LANGCHAIN_ANTHROPIC_IMPORT_ERROR}")
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    prompt = _build_text_prompt(food_text)
    message = HumanMessage(content=[{"type": "text", "text": prompt}])
    llm = ChatAnthropic(
        model=settings.claude_model_id,
        api_key=settings.anthropic_api_key,
        temperature=0.2,
        timeout=15,
        max_retries=0,
    )
    try:
        response = await _ainvoke_with_timeout(llm, message)
    except Exception as exc:
        logger.exception("Claude 텍스트 분석 호출 실패")
        raise HTTPException(status_code=502, detail=f"Claude 호출 실패: {exc}") from exc

    text = _message_content_to_text(response.content)
    try:
        return _llm_text_to_calorie_result(None, text, source="claude")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Claude 텍스트 응답 JSON 파싱 실패: %s", text[:500])
        raise HTTPException(status_code=502, detail=f"Claude 응답 변환 실패: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API — 텍스트 분석
# ---------------------------------------------------------------------------
async def analyze_text_parallel_consensus(settings: Settings, food_text: str) -> CalorieResult:
    """Gemini + Claude 병렬 텍스트 분석 후 AI별 결과를 반환합니다."""
    results = await asyncio.gather(
        _analyze_text_with_gemini(settings, food_text),
        _analyze_text_with_claude(settings, food_text),
        return_exceptions=True,
    )
    return _finalize_parallel_results(
        results[0], results[1],
        image_received=False,
        image_size_bytes=None,
    )


# ---------------------------------------------------------------------------
# 하루 권장 칼로리 분석
# ---------------------------------------------------------------------------
def _build_daily_calorie_prompt(request: DailyCalorieRequest) -> str:
    """나이/성별/키/몸무게 기반 하루 권장 칼로리 분석 프롬프트를 생성합니다."""
    weight_text = f"{request.weight_kg}kg" if request.weight_kg is not None else "미입력"
    height_text = f"{request.height_cm}cm" if request.height_cm is not None else "미입력"
    return (
        "다음 사용자 정보로 하루 권장 칼로리(유지 칼로리)를 분석해. "
        "활동량 정보가 없으므로 일반적인 낮은~보통 활동량 성인을 기준으로 보수적으로 추정해. "
        "키와 몸무게가 모두 있으면 Mifflin-St Jeor 방식의 BMR을 참고하고, 없으면 나이와 성별 기반의 일반 추정치로 계산해. "
        "의학적 진단처럼 단정하지 말고 추정치임을 note에 반영해. "
        "아래 JSON 객체 형식으로만 응답해. 다른 설명 텍스트는 절대 포함하지 마. "
        "모든 숫자 값은 단위 없이 숫자만 입력해.\n\n"
        f"나이: {request.age}\n"
        f"성별: {request.gender}\n"
        f"몸무게: {weight_text}\n"
        f"키: {height_text}\n\n"
        "{\n"
        '  "daily_calories_kcal": 하루_권장_칼로리_숫자,\n'
        '  "bmr_kcal": 기초대사량_숫자_또는_null,\n'
        '  "calculation_basis": "계산 기준 설명",\n'
        '  "confidence": 0.0~1.0_신뢰도_숫자,\n'
        '  "note": "키/몸무게 누락 여부와 추정치 관련 설명"\n'
        "}"
    )


def _dict_to_daily_calorie_recommendation(d: Dict[str, Any]) -> DailyCalorieRecommendation:
    daily = _get_float(
        d,
        "daily_calories_kcal",
        "daily_calories",
        "recommended_calories_kcal",
        "recommended_calories",
        "maintenance_calories_kcal",
        "calories_kcal",
        "권장칼로리",
        "하루권장칼로리",
    )
    if daily is None:
        raise ValueError("daily_calories_kcal 값이 없습니다.")

    bmr = _get_float(d, "bmr_kcal", "bmr", "basal_metabolic_rate_kcal", "기초대사량")
    basis = _get_str(d, "calculation_basis", "basis", "method", "계산기준") or "나이·성별 기반 하루 권장 칼로리 추정"
    conf = _get_float(d, "confidence", "신뢰도", "score")
    confidence = 0.75 if conf is None else max(0.0, min(1.0, conf))
    note = _get_str(d, "note", "notes", "설명", "주의사항")

    return DailyCalorieRecommendation(
        daily_calories_kcal=float(daily),
        bmr_kcal=bmr,
        calculation_basis=basis,
        confidence=confidence,
        note=note,
    )


def _llm_text_to_daily_calorie_recommendation(llm_text: str) -> DailyCalorieRecommendation:
    parsed = _parse_json_from_llm(llm_text)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed and isinstance(parsed[0], dict) else None
    if not isinstance(parsed, dict):
        raise ValueError("JSON 루트는 객체여야 합니다.")
    return _dict_to_daily_calorie_recommendation(parsed)


async def _analyze_daily_calories_with_gemini(
    settings: Settings,
    request: DailyCalorieRequest,
) -> DailyCalorieRecommendation:
    if _LANGCHAIN_GOOGLE_IMPORT_ERROR is not None or ChatGoogleGenerativeAI is None or HumanMessage is None:
        raise HTTPException(status_code=500, detail=f"langchain-google-genai를 불러올 수 없습니다: {_LANGCHAIN_GOOGLE_IMPORT_ERROR}")
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY가 설정되지 않았습니다.")

    message = HumanMessage(content=[{"type": "text", "text": _build_daily_calorie_prompt(request)}])
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model_id,
        api_key=settings.google_api_key,
        temperature=0.2,
        timeout=15,
        max_retries=0,
    )
    try:
        response = await _ainvoke_with_timeout(llm, message)
    except Exception as exc:
        logger.exception("Gemini 하루 권장 칼로리 분석 호출 실패")
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {exc}") from exc

    text = _message_content_to_text(response.content)
    try:
        return _llm_text_to_daily_calorie_recommendation(text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Gemini 하루 권장 칼로리 응답 JSON 파싱 실패: %s", text[:500])
        raise HTTPException(status_code=502, detail=f"Gemini 응답 변환 실패: {exc}") from exc


async def _analyze_daily_calories_with_claude(
    settings: Settings,
    request: DailyCalorieRequest,
) -> DailyCalorieRecommendation:
    if _LANGCHAIN_ANTHROPIC_IMPORT_ERROR is not None or ChatAnthropic is None or HumanMessage is None:
        raise HTTPException(status_code=500, detail=f"langchain-anthropic을 불러올 수 없습니다: {_LANGCHAIN_ANTHROPIC_IMPORT_ERROR}")
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    message = HumanMessage(content=[{"type": "text", "text": _build_daily_calorie_prompt(request)}])
    llm = ChatAnthropic(
        model=settings.claude_model_id,
        api_key=settings.anthropic_api_key,
        temperature=0.2,
        timeout=15,
        max_retries=0,
    )
    try:
        response = await _ainvoke_with_timeout(llm, message)
    except Exception as exc:
        logger.exception("Claude 하루 권장 칼로리 분석 호출 실패")
        raise HTTPException(status_code=502, detail=f"Claude 호출 실패: {exc}") from exc

    text = _message_content_to_text(response.content)
    try:
        return _llm_text_to_daily_calorie_recommendation(text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Claude 하루 권장 칼로리 응답 JSON 파싱 실패: %s", text[:500])
        raise HTTPException(status_code=502, detail=f"Claude 응답 변환 실패: {exc}") from exc


async def analyze_daily_calories_parallel_consensus(
    settings: Settings,
    request: DailyCalorieRequest,
) -> DailyCalorieResult:
    """Gemini + Claude 병렬 분석 후 하루 권장 칼로리 평균값만 반환합니다."""
    results = await asyncio.gather(
        _analyze_daily_calories_with_gemini(settings, request),
        _analyze_daily_calories_with_claude(settings, request),
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
