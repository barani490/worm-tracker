# worm-tracker

Single-worm behavioral tracker for *C. elegans*, built during SURF research at the Ryu Lab, University of Toronto (2026). It tracks one worm's position frame by frame, live from a Raspberry Pi camera or offline from any video file, and writes a CSV of position, area, circularity, detection status and distance traveled.

The detection method makes no rig-specific assumptions: it models the static background and looks at the absolute pixel difference from it, so bright-field, dark-field and inverted-contrast setups work without code changes.

## Quick start

**Windows (point-and-click GUI, no terminal)**
1. Click **Code → Download ZIP** on this page and unzip it. Keep all files together.
2. Install Python from [python.org](https://www.python.org/downloads/) and tick **"Add Python to PATH"**.
3. Double-click `Setup.vbs` (once), then `Launch Tracker.vbs` any time.
4. If the `.vbs` files do nothing (common on locked-down lab PCs): right-click → Properties → Unblock, or use `run_gui.bat` instead.

**Raspberry Pi / Linux (GUI)**
```bash
bash setup.sh          # once
bash launch_tracker.sh
```

**Command line (any OS)**
```bash
pip install -r requirements-core.txt
python3 celegans_tracker.py --mode analyze --input video.mp4 --output-csv out.csv --output-plot out.png
python3 celegans_tracker.py --help
```

## Modes

| Mode | What it does |
|---|---|
| `analyze` (default) | Track a worm in an existing video file. |
| `record` | Record from the Pi camera to a file. No analysis. |
| `record-analyze` | Record from the Pi camera, then analyze the saved file. |
| `live` | Track in real time from the Pi camera; optionally record the full video at the same time (`--record-output`). |
| `still` | Capture full-resolution stills at a fixed interval (`--still-interval`) and track across them. |

Useful flags: `--show` (live preview window), `--profile` (per-stage timing), `--clahe` (contrast enhancement), `--mm-per-px` (real-world units), `--camera-selftest` (check the Pi camera before an experiment).

## Output CSV

`frame, time_s, cx_px, cy_px, area_px, circularity, detected, cumulative_distance_px` (plus `cx_mm, cy_mm, cumulative_distance_mm` with `--mm-per-px`).

- Filter on `detected == 1` before computing speed. During short dropouts the position columns can be blank or hold the last known position.
- **`circularity`** = 4π·Area/Perimeter², the measure used by [Chen et al. (2021)](https://doi.org/10.1186/s13024-021-00497-6) to phenotype coiling. In that paper, normal crawling is about 0.2, an omega turn about 0.5, and coiling 0.6 or higher. Those cutoffs were not re-derived on this rig; check them on your own footage.
- **Live mode timing:** `time_s` is wall-clock time since recording started (or since tracking started, if you aren't recording). The first row appears only after the background-model warmup, about 9–10 s in testing, so live CSVs do not start at 0.

## What has been validated (and what hasn't)

Checked on real hardware (Raspberry Pi 4 and Pi 5 with the HQ Camera):

- **Timing.** Live CSV time matched a phone timer at 30 s and 120 s, and in a live + record run. This replaced an earlier bug where time was computed from an assumed frame rate and drifted.
- **Live vs. offline agreement.** A ~5 minute single-worm session was tracked live, then the saved video was re-analyzed offline. The two CSVs end within 0.7 s of each other (298.07 s vs 297.35 s). This check also exposed, and led to the fix of, a ~8.6 s clock offset between the recording and the live CSV.
- **Circularity.** On a ~84 s single-worm clip, video frames pulled at each circularity spike showed matching body bends: baseline crawling ≈ 0.25, moderate bends ≈ 0.4–0.5, and one near-closed loop at 0.76.
- **Behavior checks (manual, run during development):** worm leaving and re-entering the frame, worm resting about 2 min, fast movement, omega turns, and removing the worm entirely without the tracker locking onto dust.
- **Speed.** Same 2,508-frame clip, same code, `--mode analyze`: Pi 4 ≈ 7.4 fps, Pi 5 ≈ 18.4 fps, with identical tracking output (max position difference 0.0 px). Live on a Pi 5 + HQ Camera ran at ~14 fps; on a Pi 4 with the preview window open, live ran at roughly 2–4 fps. Analyze speed depends on the footage: a large or ambiguous foreground costs more per frame (`--profile` shows where the time goes).
- **Still mode** was exercised on a Pi 5 at full 12.3 MP and at 3.08 MP. The end-of-run CSV write is also covered by `tests/test_basics.py`.

Known limits:

- Single worm only. A multi-worm attempt (in `old_files/`) was dropped because identities swapped when worms crossed.
- "Detected" means a valid-area blob was found. That is coverage, not verified accuracy; dust or stains can also pass the filter. Check annotated output visually.
- No automated hardware tests. Automated checks (`tests/`) cover only circularity and the CSV writer. The manual protocol is in [TESTING.md](TESTING.md).
- `cumulative_distance_px` counts all pixel movement, including moves you make by sliding or repositioning the plate.
- Earlier development was validated on 7 recorded clips (details in [CHANGELOG.md](CHANGELOG.md)); those clips are not included in this repo.

## What it can do

**Tracking that holds up in messy real footage**
- Per-pixel, outlier-rejecting background model. In `analyze` mode it is built from frames spread across the whole file, so a worm that sits still isn't absorbed into the background. A foreground-protection step keeps a resting worm from being erased, and the patch the worm just left is actively "healed" so ghosts don't get tracked.
- Hysteresis (Canny-style) thresholding with a per-frame noise floor (MAD) plus Otsu. Faint head and tail tips stay in the contour, and thresholds self-calibrate to each recording's contrast, so there is no manual tuning across setups.
- Fresh-lock gating: a new lock needs worm-like elongation and enough local contrast, measured directly on the raw frame. This rejects dust, stains and static artifacts.
- Warmup detection skips the camera's auto-exposure / white-balance settling, so the startup brightness flicker doesn't poison the background.
- A constant-velocity Kalman filter smooths the intensity-weighted centroid, predicts where to look next, and coasts through brief dropouts. Implausible per-frame jumps are rejected.
- During a live run, a warning prints if detection collapses to about 0% (empty dish, worm out of focus), so you can check the setup right then instead of after the run.

**Capture and recording on a Raspberry Pi**
- `live` mode tracks from a separate low-resolution stream while the hardware H.264 encoder records the full-quality video, so tracking load doesn't touch recording quality. `--target-fps` leaves CPU headroom for other work.
- `still` mode captures annotated full-resolution stills (up to 12.3 MP) at a fixed interval, for long, low-magnification experiments.
- Clear failure handling: the resolution is checked against the Pi 4's encoder limit before recording starts, camera acquisition retries when the camera is still busy, `--camera-selftest` probes the camera, and video opening falls back to a GStreamer pipeline on Pi OpenCV builds that need it.

**Measurements and output**
- Per-frame CSV: position, area, circularity, detected flag and cumulative distance, with optional mm columns.
- Circularity per frame, for flagging omega turns and coiling.
- Trajectory and position-vs-time plots, plus an annotated preview video or overlay for checking the tracking by eye.

**Tooling**
- One shared tracker (`SingleWormTracker`) for live and offline, so the same logic runs in both. That is what makes the live-vs-offline comparison below meaningful.
- `--profile` and `--profile-csv` give per-stage timings. Other options: ROI crop, auto-mask for the microscope vignette, frame skip, downscale, CLAHE, and a webcam or camera index as input.
- A GUI for every mode, with a live console and an inline plot preview, plus Windows and Pi launchers.

## Hardware used

Raspberry Pi 4 (primary rig), Raspberry Pi HQ Camera (Sony IMX477) with a 12 mm C-mount lens, Raspberry Pi Camera Module 3. A Raspberry Pi 5 was also tested (benchmarks above). The Pi 4's hardware H.264 encoder tops out at 1920×1080, so the default recording preset is `1080p`.

## Repo layout

```
celegans_tracker.py     the tracker (command line)
tracker_gui.py          optional GUI (needs pygame; uses tkinter for file dialogs)
Setup.vbs, Launch Tracker.vbs, run_gui.bat       Windows launchers
setup.sh, launch_tracker.sh, Launch Tracker.desktop   Pi / Linux launchers
tests/test_basics.py    small automated checks
TESTING.md              manual hardware test protocol
CHANGELOG.md            debugging history
old_files/              early prototypes (multi-worm attempt etc.), not maintained
```

## Acknowledgments

Built with substantial help from AI (Claude): debugging, architecture, and implementing specific features, including the circularity metric and the fixes to the timing bugs. Comments in the code note where a section leaned heavily on AI or on StackOverflow OpenCV discussions (notably the background model). The circularity metric and its reference values come from:

> Chen KS, Menezes K, Rodgers JB, et al. Small molecule inhibitors of α-synuclein oligomers identified by targeting early dopamine-mediated motor impairment in *C. elegans*. *Mol Neurodegener.* 2021;16:77. https://doi.org/10.1186/s13024-021-00497-6

Thanks to Professor Ryu and the Ryu Lab.

## License

MIT, see [LICENSE](LICENSE).

## For future students

Re-run the checks in [TESTING.md](TESTING.md) on your own rig before trusting any output, rather than taking the numbers above at face value. Your camera, lens and lighting will differ from the ones used here.
