#!/bin/bash
# watch.sh - Auto-recompile LaTeX when .tex files change
# Requires: inotify-tools (apt install inotify-tools)
# Usage: ./watch.sh

cd "$(dirname "$0")"

echo "Watching for changes in .tex files..."
echo "Press Ctrl+C to stop"

while true; do
    inotifywait -e modify,create,delete -r . --include '.*\.tex$' 2>/dev/null
    echo ""
    echo "Change detected, recompiling..."
    ./build.sh
    echo "Waiting for next change..."
done
