# celegans_tracker

Center-of-mass motion tracker for *C. elegans* recorded through a stereomicroscope with a Raspberry Pi HQ camera. Also works with any other video source (webcams, other microscope rigs, pre-recorded files) — the detection algorithm makes no C.-elegans-specific or rig-specific assumptions.

## Setup

```bash
pip install -r requirements.txt
```

`matplotlib` is only needed for the auto-generated summary plot; `picamera2` is only needed on a Raspberry Pi for live/recording modes. See `requirements.txt` for details.

## Usage

```bash
python3 celegans_tracker.py --help
```

## What's tested vs. what isn't (please read before relying on it)

This project has two halves with very different levels of validation:

* The ANALYSIS PIPELINE (BackgroundModel, CentroidKalman,
  segmentation, SingleWormTracker, and `--mode analyze`) has been run
  end-to-end against 7 real recordings from this rig and checked with
  an independent, non-shared-code verification method (see CHANGELOG.md
  for exactly how, and what was found and
  fixed). As of this commit: 6 of 7 recordings detect the worm in
  100% of frames with sampled detected frames landing on real worm
  tissue 100% of the time; the 7th (live_rec.mp4) detects 65.6% of
  frames (95% on-worm accuracy on what it does detect) with an
  honestly-flagged, genuine gap where the worm is plausibly out of
  frame or resting for a large fraction of the recording -- a real
  limit of any single global background model, not hidden or patched
  over. Re-run the verification yourself before trusting this for
  real analysis; don't take this comment's word for it either.

* The RASPBERRY PI CAMERA CODE (`--mode record`, `--mode
  record-analyze`, `--mode live`, and everything using `picamera2`)
  was written against the documented Picamera2 API but could NOT be
  executed or tested in the environment this was developed in --
  there is no Raspberry Pi, no camera hardware, and no `picamera2`
  package available there. It is written carefully and defensively,
  but you should validate it on your actual Pi 4 + HQ camera before
  trusting it for a real experiment. Run with `--camera-selftest`
  first (see below) -- it's a fast, low-stakes way to catch any API
  mismatches before a real recording session.

This isn't hedging for its own sake: I'd rather tell you exactly
which parts to double-check than have you discover it the hard way
mid-experiment.

## Why this approach generalizes across rigs / contrast settings

Rather than looking for "dark worm on bright background" or vice
versa, the tracker builds a model of the *static* scene (background:
dish, stains, vignetting, dust) and looks at the *absolute* pixel-wise
difference between each new frame and that background:

    diff = | current_frame - background_model |

A worm that is brighter than the background produces a positive
difference; a worm that is darker produces a negative difference --
but abs() makes both cases produce a positive "signal". This means
bright-field, dark-field, or inverted-contrast stereomicroscope
settings are all handled without any code changes (verified against
a contrast-inverted copy of the sample recording).

Static stains/scratches/dust on the dish are, by definition, part of
the background, so they naturally disappear in the diff image and do
not get picked up as false detections. The background itself is
re-estimated slowly over time (exponential moving average) so the
tracker tolerates slow illumination drift -- but the region currently
believed to contain the worm is protected from being absorbed into
the background, so a resting worm is not gradually "erased".

Detection sensitivity is recomputed on *every frame* with Otsu's
method, so the tracker self-calibrates to whatever contrast level a
given recording happens to have.

The tracked point is a genuine intensity-weighted center of mass:
within the blob believed to be the worm, each pixel's diff value is
used as its "mass" (via cv2.moments on the weighted intensity image,
not just the binary silhouette).

This is a SINGLE-worm tracker by design (see the note in
SingleWormTracker's docstring for why, and what to do if you need
multiple simultaneous worms).

## Modes

| Mode | Description |
|---|---|
| `--mode analyze` | (default) Analyze an existing video file. This is the well-tested, primary path. |
| `--mode record` | Record from the Pi camera to a file and stop. No analysis. *[Pi camera, hardware-untested]* |
| `--mode record-analyze` | Record from the Pi camera, then run the same analysis as `analyze` on the saved file. *[Pi camera capture is hardware-untested; the analysis half it hands off to is the tested code path]* |
| `--mode live` | Analyze frames live from the Pi camera as they arrive, with an optional simultaneous recording to disk. Pass no `--record-output` to never write the full video at all — only the CSV/plot of tracked positions are produced. *[Pi camera, hardware-untested]* |

Run `python3 celegans_tracker.py --help` for the full flag reference.

Written for a Raspberry Pi 4 + HQ camera + stereomicroscope C. elegans tracking rig.

## Project files

- `celegans_tracker.py` — the tracker / analysis pipeline (this repo's main script)
- `CHANGELOG.md` — full debugging history: bugs found, root causes, fixes, and verification steps
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT

## For future students picking this up

Start with the "WHAT'S TESTED VS. WHAT ISN'T" section above before trusting any output for real analysis — it tells you exactly which parts of the pipeline are validated and which aren't. Re-run the verification steps in CHANGELOG.md yourself rather than taking prior notes at face value.
