from __future__ import annotations

import io
from typing import Optional, Tuple

from fastapi import HTTPException


def looks_like_jpeg(b: bytes) -> bool:
    return len(b) >= 3 and b[0:3] == b"\xff\xd8\xff"


def looks_like_png(b: bytes) -> bool:
    return len(b) >= 8 and b[0:8] == b"\x89PNG\r\n\x1a\n"


def looks_like_heic(b: bytes) -> bool:
    """
    HEIF/HEIC 컨테이너(ISOBMFF) 시그니처: offset 4에 'ftyp', 8~12에 major brand.
    iPhone 사진 등에서 흔한 brand: heic, heix, mif1, msf1 등.
    """
    if len(b) < 12:
        return False
    if b[4:8] != b"ftyp":
        return False
    brand = b[8:12]
    return brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1")


def heic_filename_or_mime(content_type: Optional[str], filename: Optional[str]) -> bool:
    """클라이언트가 HEIC/HEIF로 표시했는지(바이트 시그니처와 별도)."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in ("image/heic", "image/heif", "image/heif-sequence"):
        return True
    fn = (filename or "").lower()
    return fn.endswith((".heic", ".heif", ".hif"))


def convert_heif_bytes_to_jpeg(raw: bytes) -> Tuple[bytes, str]:
    """
    HEIC/HEIF 바이트를 JPEG으로 변환합니다.
    Gemini/Claude 쪽은 HEIC raw 바이트를 직접 쓰기보다 JPEG로 통일하는 편이 안정적입니다.
    """
    try:
        import pillow_heif  # type: ignore[import-untyped]
        from PIL import Image
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "HEIC/HEIF 이미지를 읽으려면 pillow, pillow-heif 가 필요합니다: "
                "pip install pillow pillow-heif"
            ),
        ) from e

    pillow_heif.register_heif_opener()
    try:
        image = Image.open(io.BytesIO(raw))
        image = image.convert("RGB")
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=92)
        return out.getvalue(), "image/jpeg"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"HEIC/HEIF 이미지를 디코딩하지 못했습니다: {e}") from e


def upload_looks_acceptable(content_type: Optional[str], filename: Optional[str], raw: bytes) -> bool:
    """
    image/* 는 허용.
    application/octet-stream 등은 JPEG/PNG/HEIC 시그니처 또는 확장자로 보완 판단.
    """
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct.startswith("image/"):
        return True
    if ct in ("application/octet-stream", ""):
        fn = (filename or "").lower()
        if fn.endswith((".jpg", ".jpeg", ".png", ".heic", ".heif", ".hif")):
            return True
        if looks_like_jpeg(raw) or looks_like_png(raw) or looks_like_heic(raw):
            return True
    return False


def prepare_image_for_llm(
    raw: bytes,
    content_type: Optional[str],
    filename: Optional[str],
) -> Tuple[bytes, str]:
    """
    LLM에 넣을 바이트와 MIME을 결정합니다.
    순서: JPEG/PNG 매직 우선 → HEIC(매직 또는 파일명/MIME) → 기타 image/* 그대로.
    """
    if looks_like_jpeg(raw):
        return raw, "image/jpeg"
    if looks_like_png(raw):
        return raw, "image/png"

    if looks_like_heic(raw) or heic_filename_or_mime(content_type, filename):
        return convert_heif_bytes_to_jpeg(raw)

    ct = (content_type or "").split(";")[0].strip().lower()
    if ct.startswith("image/"):
        return raw, ct

    raise ValueError("지원하지 않는 이미지 형식입니다. JPEG, PNG, HEIC(HEIF)를 사용하세요.")

