from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    return np.clip(arr, -1.0, 1.0)


def encode_audio(audio: np.ndarray, sample_rate: int, fmt: str, speed: float = 1.0) -> bytes:
    """Encode float waveform to the requested format. Uses ffmpeg for compressed formats."""
    wav = _to_mono_float32(audio)
    fmt = fmt.lower()

    # Approximate OpenAI speed via ffmpeg atempo (0.5–2.0 per filter; chain if needed)
    tempo = float(speed) if speed and speed > 0 else 1.0

    if fmt == "pcm":
        pcm = (wav * 32767.0).astype(np.int16)
        return pcm.tobytes()

    if fmt in {"wav", "flac"} and abs(tempo - 1.0) < 1e-3:
        buf = io.BytesIO()
        sf.write(buf, wav, sample_rate, format=fmt.upper())
        return buf.getvalue()

    with tempfile.TemporaryDirectory(prefix="qwen3-tts-") as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / f"out.{fmt if fmt != 'pcm' else 'wav'}"
        sf.write(str(src), wav, sample_rate, format="WAV")

        args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
        if abs(tempo - 1.0) >= 1e-3:
            # atempo accepts 0.5–2.0; chain filters for wider range
            filters: list[str] = []
            remaining = tempo
            while remaining > 2.0:
                filters.append("atempo=2.0")
                remaining /= 2.0
            while remaining < 0.5:
                filters.append("atempo=0.5")
                remaining /= 0.5
            filters.append(f"atempo={remaining:.6f}")
            args += ["-filter:a", ",".join(filters)]

        if fmt == "mp3":
            args += ["-codec:a", "libmp3lame", "-q:a", "2"]
        elif fmt == "opus":
            args += ["-codec:a", "libopus", "-b:a", "64k"]
        elif fmt == "aac":
            args += ["-codec:a", "aac", "-b:a", "128k"]
        elif fmt == "flac":
            args += ["-codec:a", "flac"]
        elif fmt == "wav":
            args += ["-codec:a", "pcm_s16le"]
        else:
            raise ValueError(f"Unsupported response_format: {fmt}")

        args.append(str(dst))
        subprocess.run(args, check=True)
        return dst.read_bytes()
