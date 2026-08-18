# Qwen3 TTS Local API

12GB GPU 向けの **Qwen3-TTS（CustomVoice）ローカル TTS API**。  
OpenAI 互換の `POST /v1/audio/speech` を `0.0.0.0` で待ち受け、LAN 内の他端末から利用できます。

## できること

- プリセット音声 **9 パターン**（Vivian / Serena / Uncle_Fu / Dylan / Eric / Ryan / Aiden / Ono_Anna / Sohee）
- OpenAI ボイス名エイリアス（`alloy` → Vivian など）
- `language` / `instructions`（話し方・感情の指示）
- 出力形式: `mp3` / `wav` / `flac` / `opus` / `aac` / `pcm`
- 任意の API キー（LAN 共有時の簡易保護）

音声クローンや VoiceDesign は対象外です（VRAM 12GB で TTS のみ安定運用する方針）。

## 前提

| 項目 | 推奨 |
|------|------|
| GPU | NVIDIA / VRAM **12GB** |
| モデル | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`（約 8GB） |
| OS | Linux + NVIDIA Driver + CUDA |
| その他 | Docker（推奨）または Python 3.12 + ffmpeg |

VRAM が足りない場合は `MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` に切り替えてください。

## クイックスタート（Docker）

```bash
cp .env.example .env
docker compose up --build -d
```

初回は Hugging Face からモデルを取得するため時間がかかります。ログでロード完了を確認:

```bash
docker compose logs -f qwen3-tts
curl http://127.0.0.1:8000/health
```

### LAN からアクセス

サーバー機の LAN IP を確認（例: `192.168.1.20`）し、他端末から:

```bash
curl -X POST http://192.168.1.20:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "こんにちは。LAN 経由のテストです。",
    "voice": "Ono_Anna",
    "language": "Japanese",
    "response_format": "mp3",
    "instructions": "明るく、はっきりと話してください。"
  }' \
  --output speech.mp3
```

ホストのファイアウォールで **TCP 8000** を LAN に許可してください。

## ローカル Python 起動

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

cp .env.example .env
python -m app.main
```

モデル先行ダウンロード（任意）:

```bash
python scripts/download_model.py --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

## API

| Method | Path | 説明 |
|--------|------|------|
| GET | `/health` | 状態・ロード済みスピーカー |
| GET | `/v1/voices` | 利用可能な声の一覧 |
| GET | `/v1/models` | OpenAI 互換モデル一覧 |
| POST | `/v1/audio/speech` | 音声合成 |

### `POST /v1/audio/speech` 主なフィールド

| フィールド | 例 | 説明 |
|------------|----|------|
| `input` | `"こんにちは"` | 読み上げテキスト |
| `voice` | `"Ono_Anna"` / `"alloy"` | スピーカー or OpenAI エイリアス |
| `language` | `"Japanese"` / `"Auto"` | 言語 |
| `instructions` | `"落ち着いたトーンで"` | 話し方の指示 |
| `response_format` | `"mp3"` | 出力形式 |
| `speed` | `1.0` | ffmpeg による再生速度調整 |

### OpenAI SDK 例

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.20:8000/v1", api_key="not-needed")
audio = client.audio.speech.create(
    model="tts-1",
    voice="nova",  # → Ono_Anna
    input="LAN から呼び出しています。",
)
audio.write_to_file("out.mp3")
```

`.env` で `API_KEY` を設定した場合は、その値を `api_key` または `X-API-Key` に渡してください。

## ボイス一覧（複数パターン）

| voice | 傾向 | OpenAI エイリアス例 |
|-------|------|---------------------|
| Vivian | 明るい女性（中国語寄り） | alloy, shimmer |
| Serena | 落ち着いた女性 | ash, marin |
| Uncle_Fu | 年配寄りの男性 | ballad |
| Dylan | 男性（北京語寄り） | coral |
| Eric | 男性（四川寄り） | echo |
| Ryan | 英語寄りの男性 | fable, verse |
| Aiden | 若い英語寄りの男性 | onyx, cedar |
| Ono_Anna | 日本語寄りの女性 | nova |
| Sohee | 韓国語寄りの女性 | sage |

同じテキストでも `voice` / `language` / `instructions` の組み合わせで複数パターンを作れます。

## 環境変数

`.env.example` を参照。主なもの:

- `HOST` / `PORT` — デフォルト `0.0.0.0:8000`
- `MODEL_ID` — Hugging Face モデル ID
- `MODEL_PATH` — ローカルディレクトリ（設定時は `MODEL_ID` より優先）
- `DEVICE` — `cuda:0` / `cpu`
- `DTYPE` — `bfloat16`（推奨）/ `float16` / `float32`
- `API_KEY` — 空なら認証なし
- `MAX_CONCURRENT` — 同時生成数（12GB では `1` 推奨）

## スモークテスト

```bash
chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh http://127.0.0.1:8000 Ono_Anna speech.mp3
```

## ライセンス

本リポジトリのコードは Apache-2.0 相当で利用想定。モデル自体は [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) のライセンスに従ってください。
