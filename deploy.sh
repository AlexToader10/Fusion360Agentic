#!/usr/bin/env bash
# Builds the images and starts the backend and the frontend with Docker Compose.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "Missing .env: copy .env.example to .env and set OPENROUTER_API_KEY" >&2
    exit 1
fi

docker compose up --build --detach --wait
docker compose ps
