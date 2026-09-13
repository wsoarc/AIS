#!/usr/bin/env bash
set -euo pipefail
quiz_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$quiz_root"
if [[ -d "$quiz_root/.quiz-deps" ]]; then
    export PYTHONPATH="$quiz_root/.quiz-deps${PYTHONPATH:+:$PYTHONPATH}"
fi
exec python3 -m quiz_app "$@"
