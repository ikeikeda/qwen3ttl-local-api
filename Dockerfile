# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsox-dev \
      sox \
      git \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv が Python 3.12 を取得し、依存を /opt/venv に入れる
COPY pyproject.toml uv.lock README.md ./
COPY app ./app
RUN uv python install 3.12 \
 && uv sync --frozen --no-dev --no-editable

COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
