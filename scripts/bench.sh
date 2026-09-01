#!/usr/bin/env bash
# Measures real generation throughput for each model on THIS machine.
set -euo pipefail
PROMPT='Rewrite this Python function to validate its input, and return only the complete function:

def parse_query(q):
    return q.strip().split()'

for MODEL in "$@"; do
  echo "=== $MODEL ==="
  curl -s http://localhost:11434/api/generate -d "$(jq -n \
      --arg m "$MODEL" --arg p "$PROMPT" \
      '{model:$m, prompt:$p, stream:false, options:{num_ctx:8192, temperature:0.1}}')" \
  | jq -r '
      "eval tokens      : \(.eval_count)",
      "eval duration    : \(.eval_duration/1e9 | .*100|round/100) s",
      "THROUGHPUT       : \(.eval_count / (.eval_duration/1e9) | .*10|round/10) tok/s",
      "load duration    : \(.load_duration/1e9 | .*100|round/100) s"'
  echo
done
