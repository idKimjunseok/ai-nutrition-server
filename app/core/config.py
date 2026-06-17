from __future__ import annotations

import os
from dataclasses import dataclass


def _load_dotenv_if_available() -> None:
    """
    로컬 개발 편의를 위해 `.env`를 자동 로드합니다.
    - `.env` 파일은 `.gitignore`에 포함되어 Git에 올라가지 않게 구성합니다.
    - 운영 환경에서는 보통 컨테이너/런타임 환경 변수로 주입하는 것을 권장합니다.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        # override=True: 셸에 빈 값으로 이미 세팅된 변수도 .env 값으로 덮어씁니다.
        load_dotenv(override=True)
    except Exception:
        # python-dotenv가 없어도 환경 변수로만 동작하도록 조용히 패스합니다.
        pass


@dataclass(frozen=True)
class Settings:
    """환경 변수 기반 설정."""

    google_api_key: str
    anthropic_api_key: str
    gemini_model_id: str
    claude_model_id: str
    prompt: str
    prompt_en: str
    api_key: str


def load_settings() -> Settings:
    """환경 변수에서 설정을 읽어 Settings로 반환합니다."""
    _load_dotenv_if_available()

    google_api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    anthropic_api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()

    gemini_model_id = (os.getenv("GEMINI_MODEL_ID") or "gemini-2.5-flash").strip()
    # Claude 3.5 계열 ID는 API에서 제거되어 404가 납니다. 현재 GA 기본은 Sonnet 4.6 (공식 문서 기준).
    # 다른 예: claude-haiku-4-5, claude-opus-4-7, claude-sonnet-4-5-20250929
    claude_model_id = (os.getenv("CLAUDE_MODEL_ID") or "claude-sonnet-4-6").strip()

    prompt = (
        os.getenv("NUTRITION_PROMPT")
        or (
            "이미지에 있는 음식을 분석해서 아래 JSON 배열 형식으로만 응답해. "
            "다른 설명 텍스트는 절대 포함하지 마. "
            "음식이 여러 개면 배열 요소를 추가해. "
            "이미지에 음식이 없거나, 음식인지 식별할 수 없거나, 이미지가 너무 흐리거나 "
            "음식과 무관한 사진이면 빈 배열 []로만 응답해. "
            "모든 숫자 값은 단위 없이 숫자만 입력해.\n\n"
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
    ).strip()

    prompt_en = (
        os.getenv("NUTRITION_PROMPT_EN")
        or (
            "Analyze the food in the image and respond ONLY in the following JSON array format. "
            "Do not include any explanatory text. "
            "If there are multiple foods, add an element for each one. "
            "If there is no food in the image, the food cannot be identified, the image is too "
            "blurry, or the photo is unrelated to food, respond with an empty array []. "
            "Write all food names and portion descriptions in English. "
            "All numeric values must be plain numbers without units.\n\n"
            "[\n"
            "  {\n"
            '    "food_name": "Food name in English",\n'
            '    "calories_kcal": calorie_number,\n'
            '    "portion_description": "Estimated serving size",\n'
            '    "carbs_g": carbohydrate_grams_number,\n'
            '    "protein_g": protein_grams_number,\n'
            '    "fat_g": fat_grams_number,\n'
            '    "confidence": confidence_number_between_0_and_1\n'
            "  }\n"
            "]"
        )
    ).strip()

    api_key = (os.getenv("API_KEY") or "").strip()

    return Settings(
        google_api_key=google_api_key,
        anthropic_api_key=anthropic_api_key,
        gemini_model_id=gemini_model_id,
        claude_model_id=claude_model_id,
        prompt=prompt,
        prompt_en=prompt_en,
        api_key=api_key,
    )

