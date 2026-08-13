#!/bin/bash
# launch_tracker.sh
# Opens the C. elegans Tracker GUI. Run setup.sh first (one time only).
#
# Usage:  bash launch_tracker.sh
# Or: right-click this file in the file manager -> Properties -> Permissions
# -> check "Allow executing file as program", then double-click it directly.

cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
    echo "python3 was not found. Please run setup.sh first."
    exit 1
fi

python3 tracker_gui.py
