#!/usr/bin/env bash
# =============================================================================
# load_env.sh — Загрузить переменные из .env и экспортировать их.
# Используется всеми остальными скриптами через: source scripts/load_env.sh
# =============================================================================

ENV_FILE="$(dirname "$0")/../.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Copy .env.example → .env and fill in your values."
    exit 1
fi

# Читаем .env: пропускаем комментарии и пустые строки
while IFS='=' read -r key value; do
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    key="${key//[[:space:]]/}"     # убрать пробелы
    value="${value%%#*}"           # убрать inline-комментарий
    value="${value//[[:space:]]/}" # убрать пробелы
    [[ -z "$value" ]] && continue
    export "$key=$value"
done < "$ENV_FILE"
