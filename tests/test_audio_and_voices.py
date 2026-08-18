"""Lightweight tests that do not require GPU / model weights."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio import encode_audio
from app.tts_engine import ALIAS_TO_SPEAKER, TTSEngine
from app.config import Settings


def test_openai_aliases_resolve_to_speakers() -> None:
    assert ALIAS_TO_SPEAKER["alloy"] == "Vivian"
    assert ALIAS_TO_SPEAKER["nova"] == "Ono_Anna"
    assert ALIAS_TO_SPEAKER["fable"] == "Ryan"
    assert ALIAS_TO_SPEAKER["ono_anna"] == "Ono_Anna"


def test_resolve_speaker_unknown() -> None:
    engine = TTSEngine(Settings())
    with pytest.raises(ValueError, match="Unknown voice"):
        engine.resolve_speaker("not-a-real-voice")


def test_encode_wav_and_pcm() -> None:
    sr = 24000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    wav = encode_audio(audio, sr, "wav")
    pcm = encode_audio(audio, sr, "pcm")
    assert wav[:4] == b"RIFF"
    assert len(pcm) == len(audio) * 2


def test_encode_mp3_via_ffmpeg() -> None:
    sr = 24000
    t = np.linspace(0, 0.15, int(sr * 0.15), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    mp3 = encode_audio(audio, sr, "mp3", speed=1.0)
    assert len(mp3) > 100
