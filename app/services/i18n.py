from __future__ import annotations

"""다국어(한국어/영어) 언어 판별 및 고정 메시지 모음."""

from typing import Optional


def normalize_lang(lang: Optional[str]) -> str:
    """'en'으로 시작하면 영어, 그 외에는 한국어로 취급합니다."""
    return "en" if (lang or "").strip().lower().startswith("en") else "ko"


def lang_from_accept_language(accept_language: Optional[str]) -> str:
    """
    `Accept-Language` 헤더(예: "en-US,en;q=0.9,ko;q=0.8")에서
    우선순위가 가장 높은 언어를 골라 "en" 또는 "ko"로 정규화합니다.
    헤더가 없거나 파싱할 수 없으면 "ko"를 기본값으로 사용합니다.
    """
    if not accept_language:
        return "ko"

    best_tag, best_q = "ko", -1.0
    for part in accept_language.split(","):
        part = part.strip()
        if not part:
            continue
        tag, _, q_part = part.partition(";")
        tag = tag.strip()
        if not tag:
            continue
        q = 1.0
        q_part = q_part.strip()
        if q_part.startswith("q="):
            try:
                q = float(q_part[2:])
            except ValueError:
                q = 1.0
        if q > best_q:
            best_q = q
            best_tag = tag

    return normalize_lang(best_tag.split("-")[0])


DISCLAIMER = {
    "ko": "본 결과는 생성형 AI의 추정치이며 의학적·영양학적 진단 또는 개인별 식단 지침을 대체하지 않습니다.",
    "en": "These results are AI-generated estimates and are not a substitute for medical or nutritional advice.",
}

CONSENSUS_MESSAGE = {
    "ko": "Gemini와 Claude 다중 AI 교차 검증이 완료되었습니다.",
    "en": "Cross-verification by Gemini and Claude is complete.",
}

CLAUDE_FAILED_MESSAGE = {
    "ko": "Claude 호출이 실패하여 Gemini 단독 결과로 반환합니다.",
    "en": "The Claude call failed; returning Gemini-only results.",
}

GEMINI_FAILED_MESSAGE = {
    "ko": "Gemini 호출이 실패하여 Claude 단독 결과로 반환합니다.",
    "en": "The Gemini call failed; returning Claude-only results.",
}

NO_FOOD_IMAGE_MESSAGE = {
    "ko": "이미지에서 음식을 찾을 수 없습니다. 음식이 잘 보이는 사진으로 다시 촬영해주세요.",
    "en": "No food was detected in the image. Please retake the photo with the food clearly visible.",
}

NO_FOOD_TEXT_MESSAGE = {
    "ko": "입력한 텍스트에서 음식 정보를 찾을 수 없습니다. 음식 이름을 다시 확인해주세요.",
    "en": "No food information was found in the text. Please check the food name and try again.",
}


def no_food_message(image_received: bool, lang: str = "ko") -> str:
    """Gemini/Claude 둘 다 음식을 찾지 못했을 때 사용자에게 보여줄 메시지."""
    lang = normalize_lang(lang)
    return NO_FOOD_IMAGE_MESSAGE[lang] if image_received else NO_FOOD_TEXT_MESSAGE[lang]
