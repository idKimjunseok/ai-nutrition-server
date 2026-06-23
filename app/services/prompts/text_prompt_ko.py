from __future__ import annotations

"""텍스트 음식명 분석용 한국어 프롬프트."""


def build(food_text: str) -> str:
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
