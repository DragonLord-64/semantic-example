#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${VIRTUAL_ENV:-$example_dir/../.venv}"
if [[ ! -x "$venv_dir/bin/python" ]]; then
    python3 -m venv "$venv_dir"
fi
"$venv_dir/bin/python" -m pip install -r "$example_dir/requirements.txt"
"$venv_dir/bin/python" "$example_dir/setup_model.py"
printf '\nIn VS Code, select this Python interpreter: %s/bin/python\n' "$venv_dir"
printf 'Then open semantic_search/search.py and press Run Python File.\n'
