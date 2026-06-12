from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class DailyCalorieRequest(BaseModel):
    """하루 권장 칼로리 분석 요청 모델."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    age: int = Field(..., ge=1, le=120, description="나이")
    gender: str = Field(..., description="성별 (예: male/female 또는 남성/여성)")
    weight_kg: Optional[float] = Field(None, gt=0, le=500, description="몸무게 kg, 선택사항")
    height_cm: Optional[float] = Field(None, gt=0, le=300, description="키 cm, 선택사항")


class DailyCalorieRecommendation(BaseModel):
    """AI별 하루 권장 칼로리 분석 결과."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    daily_calories_kcal: float = Field(..., description="하루 권장 칼로리 kcal")
    bmr_kcal: Optional[float] = Field(None, description="기초대사량 kcal. 키/몸무게 부족 시 null 가능")
    calculation_basis: str = Field(..., description="계산 기준 설명")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0~1 사이 신뢰도")
    note: Optional[str] = Field(None, description="키/몸무게 누락 등 보충 설명")


class DailyCalorieResult(BaseModel):
    """하루 권장 칼로리 API 응답 모델."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    daily_calories_kcal: float = Field(..., description="Gemini와 Claude 권장 칼로리 평균값")


class Macros(BaseModel):
    """탄수화물·단백질·지방 3대 영양소."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    carbohydrate_g: Optional[float] = Field(None, description="탄수화물 (g)")
    protein_g: Optional[float] = Field(None, description="단백질 (g)")
    fat_g: Optional[float] = Field(None, description="지방 (g)")


class DetectedFoodItem(BaseModel):
    """이미지에서 식별된 음식 한 항목."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    food_name: str = Field(..., description="음식 이름")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0~1 사이 신뢰도 (모델이 제공하지 않으면 서버에서 기본값 부여)",
    )
    portion_description: str = Field(
        ...,
        description="분량 설명 (예: 1인분 기준)",
    )
    calories_kcal: float = Field(..., description="해당 항목 기준 추정 칼로리 (kcal)")
    macros: Macros = Field(
        default_factory=Macros,
        description="탄수화물·단백질·지방",
    )


class CalorieResult(BaseModel):
    """클라이언트에 돌려줄 최상위 JSON 구조."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    request_id: str = Field(..., description="요청 추적용 UUID")
    analyzed_at: str = Field(..., description="서버 기준 분석 시각 (ISO-8601 UTC)")
    image_received: bool = Field(..., description="이미지 바이너리 수신 여부")
    image_size_bytes: Optional[int] = Field(
        None,
        description="수신한 파일 크기(바이트)",
    )
    message: str = Field(
        default="Gemini와 Claude 다중 AI 교차 검증 및 합의가 완료되었습니다.",
        description="사람이 읽기 쉬운 상태 메시지",
    )
    gemini_items: List[DetectedFoodItem] = Field(
        default_factory=list,
        description="Gemini가 식별한 음식 목록",
    )
    claude_items: List[DetectedFoodItem] = Field(
        default_factory=list,
        description="Claude가 식별한 음식 목록",
    )
    gemini_total_calories_kcal: Optional[float] = Field(
        None,
        description="Gemini가 독립적으로 계산한 총 칼로리 (kcal). Gemini 호출 실패 시 null",
    )
    claude_total_calories_kcal: Optional[float] = Field(
        None,
        description="Claude가 독립적으로 계산한 총 칼로리 (kcal). Claude 호출 실패 시 null",
    )
    disclaimer: str = Field(
        ...,
        description="의료/영양 조언이 아님을 명시",
    )
