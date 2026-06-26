from __future__ import annotations

"""사용자 피드백을 Resend API를 통해 이메일로 전달."""

import logging
from html import escape

import httpx
from fastapi import HTTPException

from app.core.config import Settings

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"
REQUEST_TIMEOUT_SECONDS = 10.0


async def send_feedback_email(settings: Settings, message: str) -> None:
    if not settings.resend_api_key:
        raise HTTPException(status_code=503, detail="RESEND_API_KEY가 설정되지 않았습니다.")
    if not settings.feedback_email_to:
        raise HTTPException(status_code=503, detail="FEEDBACK_EMAIL_TO가 설정되지 않았습니다.")

    payload = {
        "from": settings.feedback_email_from,
        "to": [settings.feedback_email_to],
        "subject": "[AI Nutrition] 새 피드백이 도착했습니다",
        "html": f"<p>{escape(message).replace(chr(10), '<br>')}</p>",
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                RESEND_API_URL,
                headers={"Authorization": f"Bearer {settings.resend_api_key}"},
                json=payload,
            )
    except httpx.HTTPError as exc:
        logger.exception("피드백 메일 발송 요청 실패")
        raise HTTPException(status_code=502, detail=f"피드백 메일 발송 요청 실패: {exc}") from exc

    if response.status_code >= 400:
        logger.error("피드백 메일 발송 실패: %s %s", response.status_code, response.text[:500])
        raise HTTPException(status_code=502, detail=f"피드백 메일 발송 실패: {response.status_code}")
