from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile

from app.core.config import load_settings
from app.core.rate_limit import limiter
from app.core.security import verify_api_key
from app.models.schemas import CalorieResult, DailyCalorieRequest, DailyCalorieResult
from app.services.nutrition_service import (
    analyze_daily_calories_parallel_consensus,
    analyze_image_parallel_consensus,
    analyze_text_parallel_consensus,
    lang_from_accept_language,
)
from app.utils.image_utils import prepare_image_for_llm, upload_looks_acceptable

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/v1/analyze-food", response_model=CalorieResult, response_model_by_alias=True, dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def analyze_food(
    request: Request,
    image: UploadFile = File(..., description="음식 사진 바이너리 (JPEG, PNG, HEIC)"),
    accept_language: Optional[str] = Header(None, description="응답 언어 결정 (예: en-US,en;q=0.9,ko;q=0.8). 기기 locale을 전달하세요."),
):
    """이미지를 업로드하면 Gemini + Claude가 음식과 칼로리를 분석합니다."""
    settings = load_settings()
    lang = lang_from_accept_language(accept_language)

    try:
        raw = await image.read()
    finally:
        await image.close()

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    if not upload_looks_acceptable(image.content_type, image.filename, raw):
        raise HTTPException(
            status_code=400,
            detail=(
                "지원하지 않는 요청입니다. image/* MIME을 사용하거나, "
                "HEIC/JPEG/PNG에 대해 올바른 확장자·바이너리를 보내주세요."
            ),
        )

    try:
        prepared, mime = prepare_image_for_llm(raw, image.content_type, image.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await analyze_image_parallel_consensus(settings, prepared, mime, len(raw), lang)


@router.get("/v1/analyze-food-text", response_model=CalorieResult, response_model_by_alias=True, dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def analyze_food_text(
    request: Request,
    text: str = Query(..., description="분석할 음식명. 쉼표·공백 구분 가능 (예: 우동,새우튀김)"),
    accept_language: Optional[str] = Header(None, description="응답 언어 결정 (예: en-US,en;q=0.9,ko;q=0.8). 기기 locale을 전달하세요."),
):
    """
    음식명 텍스트를 쿼리 파라미터로 입력하면 Gemini + Claude가 칼로리를 분석합니다.

    - 쉼표 구분: `?text=우동,새우튀김`
    - 공백 구분: `?text=우동 새우튀김`
    - 혼합:      `?text=우동,새우튀김 김밥`
    """
    settings = load_settings()
    lang = lang_from_accept_language(accept_language)

    food_text = text.strip()
    if not food_text:
        raise HTTPException(status_code=400, detail="음식명을 입력해주세요.")

    return await analyze_text_parallel_consensus(settings, food_text, lang)


@router.get("/v1/recommend-daily-calories", response_model=DailyCalorieResult, response_model_by_alias=True, dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def recommend_daily_calories(
    request: Request,
    age: int = Query(..., ge=1, le=120, description="나이"),
    gender: str = Query(..., description="성별 (male/female 또는 남성/여성)"),
    weight_kg: Optional[float] = Query(None, gt=0, le=500, alias="weightKg", description="몸무게 kg (선택)"),
    height_cm: Optional[float] = Query(None, gt=0, le=300, alias="heightCm", description="키 cm (선택)"),
):
    """
    나이·성별·선택 입력(몸무게/키)을 쿼리 파라미터로 받아 하루 권장 칼로리를 분석합니다.

    - 필수: `age`, `gender`
    - 선택: `weightKg`, `heightCm`
    - 예시: `?age=30&gender=male&weightKg=70&heightCm=175`
    """
    settings = load_settings()
    request = DailyCalorieRequest(age=age, gender=gender, weight_kg=weight_kg, height_cm=height_cm)
    return await analyze_daily_calories_parallel_consensus(settings, request)
