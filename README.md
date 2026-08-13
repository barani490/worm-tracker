# C. Elegans Worm Tracker
Behavioral tracking system for C. elegans using Raspberry Pi 4 + HQ Camera (or any other video source — webcams and pre-recorded files also work).
Built(ing) as a part of SURF Research at the Ryu Lab, University of Toronto (2026)

## GOALS
- At the end of SURF, having a fully functioning setup that helps with thermotaxis research, where it records live occurrence of a C. elegans thermotaxis experiment, tracks worms, and seamlessly outputs relevant information and data for analysis

## REPO CONTENTS
This repo has three versions of the tool, depending on how you want to use it:
- **`cmd-line/`** — just `celegans_tracker.py` + docs. Run it from the command line. No GUI.
- **`gui-only/`** — the point-and-click GUI (`tracker_gui.py`) plus its launcher files. Requires `celegans_tracker.py` from the cmd-line version alongside it.
- **`full/`** — everything together: the tracker, the GUI, one-click Windows setup/launch (no command line needed at all), and all docs. **Start here if you're not sure which one you want.**

## WHAT IT DOES
- Analyzes video (live from a Raspberry Pi camera, or any pre-recorded video file) and tracks a single worm's position frame-by-frame using background subtraction + Kalman-filtered centroid tracking
- Detection is self-calibrating per frame (Otsu thresholding), so it works across different contrast/lighting setups without manual tuning
- Logs position, timestamp, area, and cumulative distance to CSV every frame
- Outputs a summary trajectory plot (matplotlib)
- Point-and-click GUI available (`tracker_gui.py`) — fills in the same options as the command line, so no terminal use is required
- Has been used for a real pilot experiment: testing whether a worm's locomotor speed decays back toward baseline after being handled (mechanosensory habituation), across 15+ tracked trials — see the analysis in [link/notes if you want to reference it]
- **Single-worm only, by design.** An earlier version attempted multi-worm identity tracking (Hungarian algorithm assignment across frames) but this was found unreliable — identity swaps when worms cross paths, ghost tracks from noise — and was deliberately dropped rather than ship something that looks like it works but silently produces bad per-worm data. If/when multi-worm tracking becomes a real goal, it needs to be rebuilt properly (full multi-object association + track lifecycle management), not patched onto the current single-worm core.
- Camera setup + calibration to a live experiment environment: work in progress

## HARDWARE (used in testing)
- Raspberry Pi 4 (8GB)
- Raspberry Pi HQ Camera + 12mm C-mount lens
- Raspberry Pi Camera Module 3

## DEPENDENCIES
Core (any machine, `--mode analyze`):
- opencv-python
- numpy
- matplotlib (optional — only needed for the summary plot)

GUI version only:
- pygame

Raspberry Pi only (`--mode record` / `record-analyze` / `live`):
- picamera2

See each folder's own `requirements.txt` / README for exact install steps.
