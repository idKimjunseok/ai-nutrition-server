from __future__ import annotations

"""LLM 텍스트 응답 -> JSON -> Pydantic 모델 변환."""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.models.schemas import (
    CalorieResult,
    DailyCalorieRecommendation,
    DetectedFoodItem,
    Macros,
)
from app.services.i18n import DISCLAIMER, normalize_lang


def _strip_json_fence(text: str) -> str:
    """모델이 ```json ... ``` 로 감싼 경우 내용만 추출합니다."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text


def parse_json_from_llm(text: str) -> Any:
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


def llm_text_to_calorie_result(image_size_bytes: Optional[int], llm_text: str, source: str, lang: str = "ko") -> CalorieResult:
    """
    LLM 텍스트 응답을 CalorieResult로 변환합니다.
    source: "gemini" 또는 "claude" — 어느 AI의 결과인지 구분하여 각 AI 전용 필드에 저장합니다.
    """
    parsed = parse_json_from_llm(llm_text)
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
        disclaimer=DISCLAIMER[normalize_lang(lang)],
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


def llm_text_to_daily_calorie_recommendation(llm_text: str) -> DailyCalorieRecommendation:
    parsed = parse_json_from_llm(llm_text)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed and isinstance(parsed[0], dict) else None
    if not isinstance(parsed, dict):
        raise ValueError("JSON 루트는 객체여야 합니다.")
    return _dict_to_daily_calorie_recommendation(parsed)
