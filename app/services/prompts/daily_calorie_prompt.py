from __future__ import annotations

"""하루 권장 칼로리 분석용 프롬프트."""

from app.models.schemas import DailyCalorieRequest


def build_daily_calorie_prompt(request: DailyCalorieRequest) -> str:
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
