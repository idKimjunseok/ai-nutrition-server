from __future__ import annotations

"""이미지 분석용 프롬프트 선택 (한국어/영어는 Settings에 내장된 프롬프트를 사용)."""

from app.core.config import Settings
from app.services.i18n import normalize_lang


def select_image_prompt(settings: Settings, lang: str = "ko") -> str:
    return settings.prompt if normalize_lang(lang) == "ko" else settings.prompt_en
