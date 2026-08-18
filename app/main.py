from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.audio import CONTENT_TYPES, encode_audio
from app.config import Settings, get_settings
from app.schemas import HealthResponse, SpeechRequest, VoiceInfo
from app.tts_engine import VOICE_CATALOG, TTSEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qwen3-tts-api")


def create_engine(settings: Settings) -> TTSEngine:
    return TTSEngine(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = create_engine(settings)
    app.state.settings = settings
    app.state.engine = engine
    try:
        engine.load()
    except Exception:
        logger.exception("Failed to preload model; will retry on first request")
    yield


app = FastAPI(
    title="Qwen3 TTS Local API",
    description="OpenAI-compatible local TTS API for Qwen3-TTS (CustomVoice). Bound to 0.0.0.0 for LAN access.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine(request: Request) -> TTSEngine:
    return request.app.state.engine


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def verify_api_key(
    settings: Annotated[Settings, Depends(get_app_settings)],
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    expected = settings.api_key.strip()
    if not expected:
        return

    provided: str | None = None
    if x_api_key:
        provided = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health", response_model=HealthResponse)
def health(
    engine: Annotated[TTSEngine, Depends(get_engine)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> HealthResponse:
    return HealthResponse(
        status="ok" if engine._model is not None else "loading",
        model=engine.model_ref,
        device=settings.device,
        speakers=engine.list_speakers(),
    )


@app.get("/v1/models")
def list_models(
    engine: Annotated[TTSEngine, Depends(get_engine)],
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> dict:
    mid = engine.model_ref
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "owned_by": "qwen3-tts"},
            {"id": "tts-1-hd", "object": "model", "owned_by": "qwen3-tts"},
            {"id": mid, "object": "model", "owned_by": "qwen3-tts"},
        ],
    }


@app.get("/v1/voices", response_model=list[VoiceInfo])
def list_voices(_: Annotated[None, Depends(verify_api_key)] = None) -> list[VoiceInfo]:
    return VOICE_CATALOG


@app.post("/v1/audio/speech")
def create_speech(
    body: SpeechRequest,
    engine: Annotated[TTSEngine, Depends(get_engine)],
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> Response:
    try:
        result = engine.synthesize(
            text=body.input,
            voice=body.voice,
            language=body.language,
            instructions=body.instructions,
        )
        payload = encode_audio(result.audio, result.sample_rate, body.response_format, speed=body.speed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

    return Response(
        content=payload,
        media_type=CONTENT_TYPES[body.response_format],
        headers={
            "X-Voice-Speaker": result.speaker,
            "Content-Disposition": f'attachment; filename="speech.{body.response_format}"',
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
