from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.schemas import VoiceInfo

logger = logging.getLogger(__name__)


# Built-in CustomVoice speakers (multiple patterns for the user to pick from)
VOICE_CATALOG: list[VoiceInfo] = [
    VoiceInfo(
        id="Vivian",
        name="Vivian",
        description="明るい女性声（中国語寄り）",
        language_hint="Chinese",
        openai_aliases=["alloy", "shimmer"],
    ),
    VoiceInfo(
        id="Serena",
        name="Serena",
        description="落ち着いた女性声",
        language_hint="Chinese",
        openai_aliases=["ash", "marin"],
    ),
    VoiceInfo(
        id="Uncle_Fu",
        name="Uncle_Fu",
        description="落ち着いた男性声（年配寄り）",
        language_hint="Chinese",
        openai_aliases=["ballad"],
    ),
    VoiceInfo(
        id="Dylan",
        name="Dylan",
        description="男性声（北京語寄り）",
        language_hint="Chinese",
        openai_aliases=["coral"],
    ),
    VoiceInfo(
        id="Eric",
        name="Eric",
        description="男性声（四川寄り）",
        language_hint="Chinese",
        openai_aliases=["echo"],
    ),
    VoiceInfo(
        id="Ryan",
        name="Ryan",
        description="自然な男性声（英語寄り）",
        language_hint="English",
        openai_aliases=["fable", "verse"],
    ),
    VoiceInfo(
        id="Aiden",
        name="Aiden",
        description="若い男性声（英語寄り）",
        language_hint="English",
        openai_aliases=["onyx", "cedar"],
    ),
    VoiceInfo(
        id="Ono_Anna",
        name="Ono_Anna",
        description="女性声（日本語寄り）",
        language_hint="Japanese",
        openai_aliases=["nova"],
    ),
    VoiceInfo(
        id="Sohee",
        name="Sohee",
        description="女性声（韓国語寄り）",
        language_hint="Korean",
        openai_aliases=["sage"],
    ),
]


def _alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for voice in VOICE_CATALOG:
        mapping[voice.id.lower()] = voice.id
        for alias in voice.openai_aliases:
            mapping[alias.lower()] = voice.id
    return mapping


ALIAS_TO_SPEAKER = _alias_map()


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    speaker: str


class TTSEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._lock = threading.Semaphore(settings.max_concurrent)
        self._load_lock = threading.Lock()

    @property
    def model_ref(self) -> str:
        return self.settings.model_path or self.settings.model_id

    def resolve_speaker(self, voice: str) -> str:
        key = voice.strip()
        speaker = ALIAS_TO_SPEAKER.get(key.lower())
        if speaker is None:
            known = ", ".join(v.id for v in VOICE_CATALOG)
            raise ValueError(f"Unknown voice '{voice}'. Available: {known}")
        return speaker

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from qwen_tts import Qwen3TTSModel

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map[self.settings.dtype]
            kwargs: dict = {
                "device_map": self.settings.device,
                "dtype": dtype,
            }
            if self.settings.attn_implementation:
                kwargs["attn_implementation"] = self.settings.attn_implementation

            logger.info("Loading Qwen3-TTS from %s on %s (%s)", self.model_ref, self.settings.device, self.settings.dtype)
            self._model = Qwen3TTSModel.from_pretrained(self.model_ref, **kwargs)
            logger.info("Model loaded")

    def list_speakers(self) -> list[str]:
        if self._model is None:
            return [v.id for v in VOICE_CATALOG]
        try:
            return list(self._model.get_supported_speakers())
        except Exception:
            return [v.id for v in VOICE_CATALOG]

    def synthesize(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instructions: str | None = None,
    ) -> SynthesisResult:
        if self._model is None:
            self.load()

        speaker = self.resolve_speaker(voice)
        acquired = self._lock.acquire(timeout=300)
        if not acquired:
            raise TimeoutError("TTS queue timed out waiting for a free slot")

        try:
            assert self._model is not None
            kwargs: dict = {
                "text": text,
                "language": language or "Auto",
                "speaker": speaker,
            }
            if instructions:
                kwargs["instruct"] = instructions

            wavs, sr = self._model.generate_custom_voice(**kwargs)
            audio = np.asarray(wavs[0], dtype=np.float32)
            return SynthesisResult(audio=audio, sample_rate=int(sr), speaker=speaker)
        finally:
            self._lock.release()
