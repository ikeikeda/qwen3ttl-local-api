from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1", description="Accepted for OpenAI compatibility")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize")
    voice: str = Field(default="Vivian", description="Preset speaker or OpenAI voice alias")
    response_format: ResponseFormat = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Accepted for compatibility; applied as playback stretch when != 1.0")
    language: str = Field(default="Auto", description="Auto, Japanese, English, Chinese, ...")
    instructions: str | None = Field(default=None, description="Style/emotion instruct for CustomVoice")


class VoiceInfo(BaseModel):
    id: str
    name: str
    description: str
    language_hint: str
    openai_aliases: list[str] = []


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    speakers: list[str]
