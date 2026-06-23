from __future__ import annotations

"""텍스트 음식명 분석용 프롬프트 — 언어별 빌더로 위임."""

from app.services.i18n import normalize_lang
from app.services.prompts import text_prompt_en, text_prompt_ko


def build_text_prompt(food_text: str, lang: str = "ko") -> str:
    if normalize_lang(lang) == "en":
        return text_prompt_en.build(food_text)
    return text_prompt_ko.build(food_text)
