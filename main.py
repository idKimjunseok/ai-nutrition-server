from __future__ import annotations

"""
FastAPI 앱 엔트리포인트.

`main.py`는 **앱 초기화 + 라우터 등록만** 담당합니다.
실제 스키마/이미지 전처리/AI 호출/합의 로직은 `app/` 패키지 아래 모듈에 있습니다.

실행:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.endpoints import router
from app.core.rate_limit import limiter

app = FastAPI(
    title="AI Nutrition (Consensus)",
    description="이미지 업로드 → Gemini/Claude 병렬 분석 → 합의 결과(JSON) 반환",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)