#!/usr/bin/env bash
cd "$(dirname "$0")"
# SSH（VS Code）から起動しても、Jetson につないだモニタにウィンドウを出す
export DISPLAY="${DISPLAY:-:0}"
.venv/bin/python app.py "$@"
