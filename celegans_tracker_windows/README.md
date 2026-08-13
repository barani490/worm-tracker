# C. elegans Tracker

A tool for tracking a single *C. elegans* worm's movement in a video and measuring its speed over time. Point it at a video, and it outputs a CSV of the worm's position/speed every frame, plus an optional summary plot.

Works with any video source — a Raspberry Pi camera rig, a webcam, or a pre-recorded file. Comes with a simple point-and-click GUI, so you don't need to touch the command line to use it.

## Quick start (Windows — no typing required)

**1. Download and unzip** this whole folder somewhere on your computer. Keep every file together in the same folder.

**2. Install Python**, if you don't already have it: [python.org/downloads](https://www.python.org/downloads/). During install, check the box that says "Add Python to PATH." (This is the only step that isn't just double-clicking something in this folder.)

**3. Double-click `Setup.vbs`.** This installs everything the tracker needs automatically. You'll see a couple of small pop-up windows telling you it's working, then a "Setup complete!" message. Do this once.

**4. Double-click `Launch Tracker`** (the file named `Launch Tracker.vbs`) any time you want to open the program. No console window, no typing — it just opens the app.

If either step shows an error message instead of succeeding, see **Troubleshooting** below.

*Having trouble and want to see more detail about what's going wrong?* There's also `run_gui.bat`, which does the same thing as `Launch Tracker.vbs` but leaves a visible window open showing exactly what's happening — useful for figuring out an error, less "clean" for everyday use.

### Mac / Linux
There's no click-only installer for these yet — open a terminal in this folder and run:
```bash
pip install -r requirements.txt
python3 tracker_gui.py
```

A window should open with a form. If nothing opens, or you see an error message, see **Troubleshooting** below.

## Using the GUI

Fill in the fields, then click **Run**. Live output from the tracker streams into the black console panel at the bottom, exactly what you'd see if you'd typed the equivalent command yourself.

| Field | What it does | Required? |
|---|---|---|
| **Mode** | `analyze` — process an existing video file. This is the well-tested, main mode; use this unless you specifically need one of the Raspberry-Pi-camera modes (`record`, `record-analyze`, `live`) — see [Modes](#modes) below. | — |
| **Input video** | The video file you want to track. Click **Browse** to pick it. | Yes, for `analyze` mode |
| **Output CSV** | Where to save the frame-by-frame tracking data. Leave blank to use the tracker's default filename. | No |
| **Output video (optional)** | If set, saves a copy of the video with the tracked position drawn on top, for visually checking the tracking worked. | No |
| **Output plot PNG (optional)** | If set, saves (and the GUI previews) a summary plot of the worm's trajectory after the run finishes. | No |
| **Max frames (optional)** | Stop after this many frames — useful for a quick test run on a long video. | No |
| **Frame skip (optional)** | Process every Nth frame instead of every frame — speeds things up at the cost of temporal resolution. | No |
| **mm per px (optional)** | If you've calibrated your camera (know how many real-world millimeters one pixel represents), enter it here to get real speed units instead of just pixel-based ones. | No |
| **Show live preview window** | Check this to watch the tracking happen in a live video window while it runs. | No |
| **Extra flags (advanced)** | For anything not covered above. Type any additional command-line flags exactly as you'd type them in a terminal (e.g. `--clahe --min-area 100`). See `python3 celegans_tracker.py --help` for the full list. | No |

When it's done, the console will say `Finished (exit code 0)` — `0` means success. Anything else means something went wrong; scroll up in the console panel to see the actual error.

## What you get out

For `--mode analyze`, the main output is a **CSV file** with one row per frame: frame number, timestamp, the worm's tracked x/y position, its detected area, whether it was successfully detected that frame, and cumulative distance traveled. This is the raw data — from here you'd compute speed, normalize it, filter noise, etc. (ask whoever handed you this project for the analysis scripts/spreadsheet that do that, if you don't have them).

## Modes

| Mode | Description |
|---|---|
| `analyze` | (default, recommended) Analyze an existing video file. This is the well-tested, primary path. |
| `record` | Record from a Raspberry Pi camera to a file and stop. No analysis. *[Pi camera, hardware-untested]* |
| `record-analyze` | Record from the Pi camera, then run the same analysis as `analyze` on the saved file. *[Pi camera capture is hardware-untested; the analysis half it hands off to is the tested code path]* |
| `live` | Analyze frames live from the Pi camera as they arrive, with an optional simultaneous recording to disk. *[Pi camera, hardware-untested]* |

## Troubleshooting

**Double-clicking `Setup.vbs` or `Launch Tracker.vbs` does nothing at all.**
Windows may be blocking `.vbs` files from running for security reasons on some locked-down computers (common on school/lab machines). Right-click the file → **Properties** → check if there's an "Unblock" checkbox near the bottom → check it → OK, then try again. If that's not available, fall back to `run_gui.bat` instead, which doesn't have this restriction.

**"Python wasn't found on this computer" pop-up, even though you installed Python.**
Python wasn't added to your system PATH during install. Reinstall Python from python.org and make sure to check "Add Python to PATH" this time.

**Setup finishes with an error message.**
Follow the pop-up's instructions to run the command manually in Command Prompt — that'll show you the actual error text, which is usually a missing internet connection or an outdated pip.

**The GUI opens but the Run button seems to do nothing.**
Check the console panel at the bottom of the GUI window — the command may have failed instantly (e.g. no input video selected for `analyze` mode). Scroll up in the console text for the actual error line.

**Tracking runs but the worm isn't detected well / detection percentage is low.**
This is a tracker-tuning issue, not a GUI/setup issue — see "What's tested vs. what isn't" below, and try the advanced flags (`--min-area`, `--clahe`, etc.) via the Extra Flags field.

## What's tested vs. what isn't (please read before relying on it)

This project has two halves with very different levels of validation:

* The **analysis pipeline** (`--mode analyze`, and everything the GUI's default mode uses) has been run end-to-end against 7 real recordings from the original rig and checked with an independent, non-shared-code verification method (see `CHANGELOG.md` for exactly how, and what was found and fixed). As of the last update: 6 of 7 recordings detect the worm in 100% of frames with sampled detected frames landing on real worm tissue 100% of the time; the 7th detects it in 65.6% of frames (95% accuracy on what it does detect), with an honestly-flagged gap where the worm is plausibly out of frame or resting for a large stretch — a real limit of this kind of background-model approach, not hidden or patched over. Re-run the verification yourself before trusting this for real analysis.

* The **Raspberry Pi camera code** (`record`, `record-analyze`, `live` modes) was written against the documented Picamera2 API but could not be executed or tested in the environment it was developed in — there was no physical Raspberry Pi or camera available. It's written carefully, but validate it on real Pi hardware (`--camera-selftest` is a fast, low-stakes way to catch API mismatches) before trusting it for a real experiment.

* The **GUI** is a thin wrapper: it only builds and runs the same command you'd type yourself, and doesn't touch any tracking logic. Bugs here would be about the interface (a field not saving correctly, a button not working), not about tracking accuracy.

## Why the tracking approach generalizes across rigs / contrast settings

Rather than looking for "dark worm on bright background" or vice versa, the tracker builds a model of the *static* scene (background: dish, stains, vignetting, dust) and looks at the absolute pixel-wise difference between each new frame and that background. This means bright-field, dark-field, or inverted-contrast microscope settings are all handled without any code changes. Static stains/scratches/dust on the dish are, by definition, part of the background, so they don't get picked up as false detections. Detection sensitivity is recomputed on every frame (Otsu's method), so the tracker self-calibrates to whatever contrast level a given recording has.

This is a **single-worm tracker** by design — it will not correctly track multiple worms in the same frame.

## Project files

- `Setup.vbs` — one-time setup, installs everything needed (double-click first)
- `Launch Tracker.vbs` — opens the GUI, no console window (double-click any time after Setup)
- `run_gui.bat` — alternate launcher that shows visible output, useful for troubleshooting
- `tracker_gui.py` — the point-and-click GUI itself
- `celegans_tracker.py` — the underlying tracker/analysis engine (what the GUI runs for you; can also be used directly from the command line — run `python3 celegans_tracker.py --help`)
- `CHANGELOG.md` — full debugging history: bugs found, root causes, fixes, and verification steps
- `requirements.txt` — Python package dependencies
- `LICENSE` — MIT

## For future students picking this up

Start with **"What's tested vs. what isn't"** above before trusting any output for real analysis. Re-run the verification steps in `CHANGELOG.md` yourself rather than taking prior notes at face value — that's genuinely the fastest way to know whether anything's drifted since this was last touched.
