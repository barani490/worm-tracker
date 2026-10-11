#!/bin/bash
# setup.sh
# Run this ONCE to install everything the tracker needs on a Raspberry Pi
# (or any Linux machine). After this, use launch_tracker.sh to open the GUI.
#
# How to run it:
#   1. Open a terminal in this folder (right-click in the file manager -> "Open Terminal Here")
#   2. Type:  bash setup.sh
#
# (Yes, this one step needs a terminal -- there isn't a reliable
# double-click-only installer on Linux the way there is on Windows.
# But it's the ONLY command you'll ever need to type.)

set -e
cd "$(dirname "$0")"

echo "=== C. elegans Tracker setup ==="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "python3 was not found. Raspberry Pi OS normally comes with it pre-installed."
    echo "Try: sudo apt update && sudo apt install python3 python3-pip"
    exit 1
fi

echo "Installing required system packages (this may ask for your password)..."
sudo apt update
sudo apt install -y python3-pip python3-tk python3-pygame

echo ""
echo "Installing Python packages..."
# --break-system-packages is needed on newer Raspberry Pi OS (Bookworm+),
# which blocks plain pip installs by default. Harmless on older versions.
pip3 install --break-system-packages -r requirements.txt || pip3 install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo "You can now run:  bash launch_tracker.sh"
echo "(or double-click launch_tracker.sh in the file manager, if your Pi is set up to run scripts on double-click)"
