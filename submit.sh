#!/usr/bin/env bash
set -euo pipefail

COMPETITION="molecular-graph-captioning"

FILE="$1"
MESSAGE="$2"

if [ $# -ne 2 ]; then
    echo "Usage: ./submit.sh <file> <message>"
    exit 1
fi

[ -f "$FILE" ] || { echo "File not found: $FILE"; exit 1; }

uvx kaggle competitions submit \
    -c "$COMPETITION" \
    -f "$FILE" \
    -m "$MESSAGE"
