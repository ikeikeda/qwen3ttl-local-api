#!/usr/bin/env bash
# Quick smoke test against a running server.
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
VOICE="${2:-Ono_Anna}"
OUT="${3:-speech.mp3}"

echo "Health: $(curl -sS "$BASE_URL/health")"
echo "Voices:"
curl -sS "$BASE_URL/v1/voices" | python3 -m json.tool | head -n 40

curl -sS -X POST "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"tts-1\",\"input\":\"こんにちは。ローカルの Qwen3 TTS です。\",\"voice\":\"$VOICE\",\"language\":\"Japanese\",\"response_format\":\"mp3\",\"instructions\":\"落ち着いたトーンで話してください。\"}" \
  --output "$OUT"

echo "Wrote $OUT ($(wc -c < "$OUT") bytes)"
