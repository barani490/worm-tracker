# worm-tracker

Behavioral tracking system for *C. elegans*, built during SURF research at the Ryu Lab, University of Toronto (2026). Tracks a single worm's position frame-by-frame in video, from a Raspberry Pi + HQ camera rig, a webcam, or any pre-recorded file, and outputs a CSV of position, speed, and detection quality over time.

The core detection/tracking algorithm makes no C.-elegans-specific or rig-specific assumptions: it works by modeling the static background and looking at absolute pixel-wise difference, so bright-field, dark-field, and inverted-contrast setups are all handled without code changes.

## Which folder do I want?

This repo has three versions of the same tracker, packaged for different setups:

| Folder | Use this if... |
|---|---|
| [`celegans_tracker_cmdline/`](celegans_tracker_cmdline) | You're comfortable with a terminal and just want the tracker (`celegans_tracker.py`), no GUI. Works on any OS. |
| [`celegans_tracker_rpi/`](celegans_tracker_rpi) | You're on a Raspberry Pi (or other Linux machine) and want the point-and-click GUI. Includes `setup.sh` / `launch_tracker.sh` for one-time setup and launching. |
| [`celegans_tracker_windows/`](celegans_tracker_windows) | You're on Windows and want the point-and-click GUI with a double-click installer (`Setup.vbs`) and launcher (`Launch Tracker.vbs`) — no terminal needed at all. |

All three run the same underlying `celegans_tracker.py` analysis engine. they differ only in launcher/packaging, not in tracking logic. Each folder has its own README with full setup instructions.

`old_files/` holds earlier development versions (multi-worm attempt, standalone detect/record/visualize scripts) — kept for history, not maintained or recommended for use.

## What it does

- Tracks a single worm's centroid frame-by-frame using background subtraction + Kalman-filtered position estimation
- Self-calibrates per frame (Otsu thresholding), so no manual tuning across different contrast/lighting setups
- Outputs a CSV every frame: timestamp, x/y position, area, detected (bool), cumulative distance
- Optional summary trajectory plot (matplotlib) and annotated debug video
- Optional live/record modes for Raspberry Pi camera capture (`--mode record`, `record-analyze`, `live`)

## Single-worm only, by design

An earlier attempt at multi-worm identity tracking (Hungarian algorithm assignment across frames ( see `old_files/`) )was found unreliable: identity swaps when worms cross paths, ghost tracks from noise. It was deliberately dropped rather than ship something that looks like it works but silently produces bad per-worm data. Multi-worm tracking, if needed later, should be rebuilt properly (full multi-object association + track lifecycle management), not patched onto the current single-worm core.

## What's tested vs. what isn't

The analysis pipeline (`--mode analyze`) has been run end-to-end against 7 real recordings and checked with an independent verification method — see each folder's `CHANGELOG.md` for details. 6 of 7 recordings track the worm in 100% of frames; the 7th has an honestly-flagged gap (worm plausibly out of frame/resting for part of the recording).

The Raspberry Pi camera code (`record` / `record-analyze` / `live` modes, anything using `picamera2`) was written against the documented API but has not been run on real Pi hardware. Run `--camera-selftest` before a real experiment to catch API mismatches early.

See the relevant folder's README for the full breakdown before trusting output for real analysis.

## Hardware used in testing

- Raspberry Pi 4 (8GB)
- Raspberry Pi HQ Camera + 12mm C-mount lens
- Raspberry Pi Camera Module 3

## For future students picking this up

Start with the "What's tested vs. what isn't" section in whichever folder's README applies to you, and re-run the verification steps in `CHANGELOG.md` yourself rather than taking prior notes at face value.
