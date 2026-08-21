#!/usr/bin/env bash
# Bootstrap local env with uv (pins Python 3.12, creates .venv, installs deps).
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv が見つかりません。インストールします..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

echo "Python $(cat .python-version) を確保して依存関係を同期します..."
uv python install
uv sync --group dev

cat <<'EOF'

準備完了。起動例:

  cp -n .env.example .env
  uv run python -m app.main

テスト:

  uv run pytest
EOF
