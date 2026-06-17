from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from app.core.config import load_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    settings = load_settings()
    if not settings.api_key:
        return  # API_KEY 미설정 시 인증 스킵 (로컬 개발 편의)
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
