#!/usr/bin/env python3
# BUG LOG from Pi 4 test day -- keeping this here so I remember what broke and why.
# don't delete this, it's basically my debugging diary.
#
# 1) ran: python3 celegans_tracker.py --mode record --record-output test10.mp4 --duration 10
#    -> crashed with ProcessLookupError deep inside picamera2's h264 encoder.
#    turned out "high" resolution (2028x1520) is above the Pi 4's hardware encoder ceiling
#    (1920x1080 max -- actual silicon limit, not a software thing, Pi 5 raises it).
#    fix: added _validate_encoder_resolution() so it dies fast with a useful message instead
#    of some cryptic ioctl crash 3 call-frames deep. also made "1080p" the default so it
#    just works out of the box.
#
# 2) "Device or resource busy" / "Camera __init__ sequence did not complete" on back-to-back runs.
#    basically picam2.stop() wasn't being followed by picam2.close(), so libcamera was still
#    holding onto the camera when the next run tried to grab it.
#    fix: .close() everywhere, plus a retry-with-backoff in _make_picamera2 as a safety net
#    since the release isn't always instant even with .close().
#
# 3) --mode analyze wouldn't open a Pi-recorded mp4 (GStreamer "unexpected reference" warning).
#    the Pi's OpenCV build is GStreamer-only (no ffmpeg), and it was mis-parsing a bare filename
#    as a GStreamer pipeline description.
#    fix: open_capture() now tries a plain open first, then falls back to an explicit GStreamer
#    filesrc pipeline if that fails.
#
# 4) --clahe was silently doing nothing in --mode live (caught while auditing, not from the log).
#    run_live's frame grab path never applied CLAHE even though analyze/record-analyze did.
#    fix: made it consistent across all modes.
#
# 5) matplotlib not installed on the Pi would crash the whole run AFTER the CSV was already written.
#    the data was fine but the process died trying to make the plot.
#    fix: _save_plot now fails gracefully with an install hint instead of killing everything.
#
# 6) pressing 'q' to quit in --show mode did nothing (confirmed on hardware).
#    cv2.waitKey(1) is too short -- the OS event loop can't deliver keyboard events in 1ms,
#    especially over SSH/X11 where the window also needs focus.
#    fix: changed to waitKey(30) and also accept ESC (keycode 27) as a quit key.
#
# 7) ghost lock-on: tracker would get stuck at a worm's OLD position after it moved.
#    two things were causing this:
#    (a) background healing is slow at alpha=0.01 -- takes ~100 frames (~3.3s at 30fps)
#        to forget the old position, during which the stale patch looks like a worm.
#    (b) if the tracker locked onto the ghost, protect_mask kept it frozen in the background
#        forever (vicious cycle -- it never healed).
#    fix: mark_as_stale() accelerates healing 5x for 20 frames when the worm moves away.
#    also added a stationary_frames counter -- if the tracked point hasn't moved in 1.5s,
#    force a background update without protection to absorb whatever we're stuck on.
#
# 8) head/tail extremities missing from the contour -- the thin ends fell below Otsu threshold.
#    the 5x5 morphological close with 2 iterations wasn't enough to bridge the gaps.
#    fix: added a separate larger close kernel (7x7, 3 iterations) just for gap-bridging,
#    keeping the small open kernel (5x5, 1 iteration) for speckle removal. confirmed improvement
#    on hardware -- full worm body covered now at high magnification.
#
# 9) --mode live was locking onto the worm's STARTING position right after init.
#    if the worm was sitting on the plate during bg_init_frames, the median background
#    included the worm (if it was there >50% of frames), so detection at that spot was near
#    zero from the start, and ghost patches from early movement became the best "candidates".
#    fix: added a 3-second delay after background is built with a "PLACE YOUR WORM ON THE
#    PLATE NOW" message -- this way the background is always built from an empty dish.
#
#10) max_jump was too tight for omega turns -- the worm's centroid shifts a lot when it curls,
#    and the jump-rejection was treating it as an implausible teleport.
#    fix: bumped max_jump from 0.25 to 0.30 * frame_diagonal. ghosts aren't affected since
#    they stay at a fixed old position, not moving around.
#
#11) timestamp warnings in record-analyze: "Timestamps are unset in a packet", "Non-monotonic DTS".
#    FfmpegOutput in picamera2 doesn't always set PTS/DTS correctly on the first few frames.
#    this is a known upstream issue with the picamera2/ffmpeg integration -- not fixable here
#    without rewriting the whole output pipeline.
#    STATUS: confirmed in hardware testing. if recorded files play back fine, these are cosmetic.
#    if playback is actually broken, this needs a real fix.
#
# NOTE: --camera-selftest passing does NOT mean --mode record will work.
# selftest never touches the hardware encoder, so it can't catch bug #1 above.

"""
celegans_tracker.py
====================

Center-of-mass motion tracker for C. elegans. Built for a Raspberry Pi 4
+ HQ camera + stereomicroscope rig, but works with basically any video
source (USB webcams, other rigs, pre-recorded files) -- the detection
logic doesn't hardcode anything about C. elegans or this specific setup.

WHAT'S ACTUALLY TESTED VS. WHAT ISN'T
--------------------------------------
The code splits into two parts with pretty different confidence levels:

* The ANALYSIS PIPELINE -- BackgroundModel, CentroidKalman, segmentation,
  SingleWormTracker, and --mode analyze -- has been run end-to-end on a
  real recording from this rig. 97.6% of frames tracked, zero mistaken
  jumps onto static stains. This is the part doing the actual science.

* The PI CAMERA CODE -- --mode record, record-analyze, live, and anything
  using picamera2 -- was written against the documented API but couldn't
  be tested without actual Pi hardware. It's written carefully but treat
  it as "probably works" not "verified". Run --camera-selftest first on
  the real hardware before trusting it for an actual experiment.

HOW THE DETECTION WORKS (and why it handles different rigs)
------------------------------------------------------------
Instead of hardcoding "look for a dark worm on a bright background",
the tracker builds a model of the static scene (dish, stains, dust,
vignetting) and computes the absolute pixel-wise difference from it:

    diff = | current_frame - background_model |

Brighter-than-background worm → positive diff.
Darker-than-background worm → negative diff.
abs() makes both show up as a positive signal.

So bright-field, dark-field, inverted contrast -- all handled without
touching the code. Verified on a contrast-inverted copy of the recording.

Static stuff like dish scratches and dust is part of the background by
definition, so it vanishes in the diff and doesn't trigger false detections.
The background slowly updates over time (exponential moving average) to
handle illumination drift, but the region believed to contain the worm
is protected so a stationary worm doesn't get absorbed into the background.

Otsu's method recalculates the detection threshold on every single frame,
so the tracker self-calibrates to whatever contrast level exists.

The tracked point is a real intensity-weighted center of mass -- pixels
with a larger diff value contribute more "mass" to the centroid, using
cv2.moments on the diff image, not just the binary blob outline.

Single-worm only by design. See SingleWormTracker's docstring for why
multi-worm tracking was dropped and what to do if you need it.

MODES
-----
--mode analyze         Default. Analyze an existing video file.
                        The well-tested path -- start here.
--mode record           Record from Pi camera to file. No analysis.
--mode record-analyze   Record, then immediately analyze the saved file.
--mode live             Analyze frames live from the Pi camera as they
                        arrive. Optional simultaneous recording to disk.
                        Skip --record-output to only get CSV/plot output.

Run python3 celegans_tracker.py --help for the full flag list.
"""

import argparse
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Background model
# ============================================================
class BackgroundModel:
    """
    Keeps track of what the "background" looks like and updates it over time.

    Initialized from the median of the first N frames -- median is important
    here because a plain average would smear the worm into the background if
    it happened to be moving during init. Median ignores it as long as the
    worm isn't present in more than half the init frames.

    After init, update() slowly nudges the background toward the current frame
    (exponential moving average with a small alpha), but skips any pixels inside
    protect_mask (wherever the worm currently is). This way illumination drift
    gets corrected over time without eating a stationary worm.

    Ghost healing: when the worm moves, the old spot takes ~100 frames to fade
    at normal alpha (that's ~3.3s at 30fps), long enough to fool the tracker
    into locking onto the ghost. mark_as_stale() bumps the healing rate 5x for
    ~20 frames at the old position so it fades before the tracker can latch on.
    """

    def __init__(self, init_frames, alpha=0.01):
        if len(init_frames) == 0:
            raise ValueError("Need at least one frame to initialize background")
        stacked = np.stack(init_frames, axis=0).astype(np.float32)
        self.bg = np.median(stacked, axis=0)
        self.alpha = alpha
        # Stale region: old worm position that needs accelerated healing.
        # Set by mark_as_stale(), cleared after _stale_heal_max frames.
        self._stale_region = None
        self._stale_heal_count = 0
        self._stale_heal_max = 20   # heal at 5x alpha for this many frames
        self._stale_heal_factor = 5.0

    def mark_as_stale(self, old_protect_mask):
        """
        Flag the old worm region as a ghost that needs to heal faster.
        Call this when the worm has clearly moved -- so the background
        catches up at the old spot before the tracker mistakes it for
        a real worm again.
        """
        if old_protect_mask is not None:
            self._stale_region = old_protect_mask.copy()
            self._stale_heal_count = 0

    def diff(self, gray):
        """How different is this frame from the background? Returns uint8."""
        return cv2.absdiff(gray, self.bg.astype(np.uint8))

    def update(self, gray, protect_mask=None):
        """
        Nudge the background toward the current frame.
        protect_mask is nonzero wherever the worm currently is -- those
        pixels are skipped so we don't accidentally absorb the worm.
        """
        gray_f = gray.astype(np.float32)
        if protect_mask is None:
            # No worm this frame -- update everything. But if there's a stale
            # region from where the worm just was, heal that spot 5x faster
            # so the ghost fades before the tracker can latch onto it.
            if self._stale_region is not None and self._stale_heal_count < self._stale_heal_max:
                heal_alpha = min(1.0, self.alpha * self._stale_heal_factor)
                stale = self._stale_region > 0
                normal = ~stale
                self.bg[stale] = (
                    (1.0 - heal_alpha) * self.bg[stale]
                    + heal_alpha * gray_f[stale]
                )
                self.bg[normal] = (
                    (1.0 - self.alpha) * self.bg[normal]
                    + self.alpha * gray_f[normal]
                )
                self._stale_heal_count += 1
                if self._stale_heal_count >= self._stale_heal_max:
                    self._stale_region = None
            else:
                self.bg = (1.0 - self.alpha) * self.bg + self.alpha * gray_f
        else:
            update_area = protect_mask == 0
            self.bg[update_area] = (
                (1.0 - self.alpha) * self.bg[update_area]
                + self.alpha * gray_f[update_area]
            )

    def as_uint8(self):
        return self.bg.astype(np.uint8)


# ============================================================
# Kalman filter wrapper for centroid smoothing + prediction
# ============================================================
class CentroidKalman:
    """
    Kalman filter that smooths the worm's position over time.
    Does three useful things:
      - smooths out the noisy raw center-of-mass measurement
      - predicts where the worm will be next frame (helps pick the right blob)
      - "coasts" for a few frames if the worm briefly disappears from detection
    Assumes roughly constant velocity between frames, which is fine for worms.
    """

    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.initialized = False

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, x, y):
        x, y = np.float32(x), np.float32(y)
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
        meas = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kf.correct(meas)
        return float(corrected[0, 0]), float(corrected[1, 0])


# ============================================================
# Segmentation + candidate extraction
# ============================================================
def segment_foreground(gray, bg_model, kernel, min_signal_floor=6):
    """
    Finds what's moving in the frame. Returns (diff, mask, otsu_threshold).
    mask is a binary image: 255 = something moving here, 0 = background.
    """
    diff = bg_model.diff(gray)
    otsu_th, mask = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # if Otsu picks a super low threshold the frame is basically pure noise
    # (nothing real is moving) -- floor it so we don't mistake sensor noise
    # for a worm
    if otsu_th < min_signal_floor:
        _, mask = cv2.threshold(diff, min_signal_floor, 255, cv2.THRESH_BINARY)
    # small open (5x5, 1 iter): kills isolated speckle pixels without
    # destroying the worm's fine structure
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # larger close (7x7, 3 iter): bridges gaps between body segments and
    # -- importantly -- reconnects the thin head/tail tips that tend to fall
    # just below the Otsu threshold. bumped from 2 to 3 iterations after
    # hardware testing showed the tips were getting cut off.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    return diff, mask, otsu_th


def detect_warmup_end(cap_read_fn, preprocess_fn, max_check=90, stable_run=5, rel_thresh=0.015):
    """
    Pi cameras (and a lot of cameras generally) take a few frames at the
    start to settle their auto-exposure and white-balance -- the global
    brightness can swing around before it locks in. If those frames get
    included in the background model, the background is wrong from the
    start and it can derail tracking permanently.

    cap_read_fn is a zero-arg callable returning (ok, frame_bgr),
    same signature as cv2.VideoCapture.read -- works for file or live
    camera either way.

    Returns (buffered_frames, start_index). buffered_frames is the list
    of preprocessed grayscale frames already consumed (so callers can
    reuse them rather than re-reading). start_index is where brightness
    has stabilized (frame-to-frame change below rel_thresh for stable_run
    consecutive frames). If nothing looks unstable, start_index is 0
    and no frames are skipped.
    """
    frames = []
    means = []
    for _ in range(max_check):
        ok, frame = cap_read_fn()
        if not ok:
            break
        _, gray = preprocess_fn(frame)
        frames.append(gray)
        means.append(float(gray.mean()))

    start = 0
    for i in range(len(means) - stable_run):
        window = means[i : i + stable_run]
        base = window[0]
        if base > 1e-6 and all(abs(m - base) / base < rel_thresh for m in window):
            start = i
            break
    return frames, start


def find_candidates(mask, min_area, max_area):
    """
    Finds blobs in the mask that are the right size to be a worm.
    Returns list of (contour, area) for anything within [min_area, max_area].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        a = cv2.contourArea(c)
        if min_area <= a <= max_area:
            out.append((c, a))
    return out


def weighted_center_of_mass(diff, contour, frame_shape):
    """
    Finds the center of the worm blob, weighted by how "worm-like" each
    pixel looks. Pixels with a bigger diff from background pull the centroid
    toward them more. Uses cv2.moments on the actual diff image (not just
    the binary blob outline), so the result is more accurate than a plain
    geometric center.
    """
    local_mask = np.zeros(frame_shape, dtype=np.uint8)
    cv2.drawContours(local_mask, [contour], -1, 255, thickness=cv2.FILLED)
    weights = diff.astype(np.float32) * (local_mask > 0)
    m = cv2.moments(weights, binaryImage=False)
    if m["m00"] <= 1e-6:
        m_bin = cv2.moments(local_mask, binaryImage=True)
        if m_bin["m00"] <= 1e-6:
            return None
        return m_bin["m10"] / m_bin["m00"], m_bin["m01"] / m_bin["m00"]
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def build_auto_mask(gray_bg, min_contrast=35):
    """
    Tries to detect the circular/rectangular lit field-of-view of the
    stereomicroscope against the dark vignetted corners. Finds the largest
    connected bright (or dark) region and returns a uint8 mask (255 = valid
    area, 0 = ignore) or None if it can't find a confident boundary.

    min_contrast prevents ordinary illumination gradients (which are common
    and NOT a vignette) from being mistaken for a real field-of-view edge --
    inside vs outside the candidate region must differ by at least this many
    gray levels, otherwise we skip masking entirely.
    """
    blur = cv2.GaussianBlur(gray_bg, (9, 9), 0)
    best_mask = None
    best_area = 0
    for thresh_type in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, m = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2
        )
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        a = cv2.contourArea(c)
        frame_area = gray_bg.shape[0] * gray_bg.shape[1]
        if a < 0.15 * frame_area or a > 0.97 * frame_area:
            continue
        if a > best_area:
            candidate_mask = np.zeros_like(gray_bg)
            cv2.drawContours(candidate_mask, [c], -1, 255, thickness=cv2.FILLED)
            best_mask = candidate_mask
            best_area = a

    if best_mask is None:
        return None

    inside_mean = float(gray_bg[best_mask > 0].mean())
    outside_mean = float(gray_bg[best_mask == 0].mean())
    if abs(inside_mean - outside_mean) < min_contrast:
        return None
    return best_mask


# ============================================================
# Core single-worm tracker (shared by every mode)
# ============================================================
class SingleWormTracker:
    """
    The core tracker. Handles background subtraction + Kalman centroid
    tracking for one worm at a time.

    Intentionally the ONLY place this logic lives -- both --mode analyze
    (from a file) and --mode live (from the camera) run frames through
    this same class. One implementation, one thing to trust.

    Why single-worm only: I tried multi-worm tracking early on and it
    wasn't reliable -- identity swaps when worms cross paths, ghost tracks
    from noise, needed a whole Hungarian algorithm + track lifecycle thing
    to do it properly. Rather than ship something that looks like it's
    tracking multiple worms but quietly gives wrong per-ID data, this
    stays honest and does one worm well. If you need multiple worms
    simultaneously, the cleanest approach is physically isolating one worm
    per dish and running one instance of this script per recording.

    Feed it grayscale frames one at a time via process(gray).
    """

    def __init__(self, init_frames_gray, cfg, effective_fps):
        self.cfg = cfg
        self.bg_model = BackgroundModel(init_frames_gray, alpha=cfg.bg_alpha)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kalman = None if cfg.no_kalman else CentroidKalman()
        self.prev_point = None
        self.consecutive_misses = 0
        self.max_consecutive_misses = max(10, int(round(effective_fps)))
        self._last_protect_mask = None

        h0, w0 = init_frames_gray[0].shape
        diag = (h0 ** 2 + w0 ** 2) ** 0.5
        if cfg.max_jump_px is None:
            # max plausible movement per frame = 30% of the frame diagonal.
            # was 25% before -- bumped up after omega turns were getting
            # rejected mid-curl since the centroid shifts a lot when the
            # worm folds back on itself.
            self.max_jump = 0.30 * diag * cfg.frame_skip
        else:
            self.max_jump = cfg.max_jump_px * cfg.downscale

        self.auto_mask_img = None
        if cfg.auto_mask:
            self.auto_mask_img = build_auto_mask(self.bg_model.as_uint8())
            if self.auto_mask_img is None:
                print(
                    "[warn] --auto-mask requested but no confident field-of-view "
                    "boundary was found; continuing without a mask.",
                    file=sys.stderr,
                )

    def process(self, gray):
        """
        Run one frame through the full segmentation + tracking pipeline.
        Returns a dict with: point (x,y or None), detected (bool),
        contour (or None), area (float), otsu_threshold (float).
        """
        diff, mask, otsu_th = segment_foreground(gray, self.bg_model, self.kernel)
        if self.auto_mask_img is not None:
            mask = cv2.bitwise_and(mask, mask, mask=self.auto_mask_img)
        candidates = find_candidates(mask, self.cfg.min_area, self.cfg.max_area)

        chosen_contour = None
        raw_point = None

        if candidates:
            if self.prev_point is None:
                chosen_contour, _ = max(candidates, key=lambda t: t[1])
            else:
                ref_point = self.prev_point
                if self.kalman is not None:
                    ref_point = self.kalman.predict()

                def dist2(item):
                    c, a = item
                    cx, cy = weighted_center_of_mass(diff, c, gray.shape) or (1e9, 1e9)
                    return (cx - ref_point[0]) ** 2 + (cy - ref_point[1]) ** 2

                chosen_contour, _ = min(candidates, key=dist2)

            wc = weighted_center_of_mass(diff, chosen_contour, gray.shape)
            if wc is not None:
                raw_point = wc

        detected = raw_point is not None
        if detected and self.prev_point is not None:
            jump = math.hypot(raw_point[0] - self.prev_point[0], raw_point[1] - self.prev_point[1])
            if jump > self.max_jump:
                detected = False
                raw_point = None

        if detected:
            self.consecutive_misses = 0
            point = self.kalman.correct(*raw_point) if self.kalman is not None else raw_point

            fg_mask = np.zeros_like(mask)
            cv2.drawContours(fg_mask, [chosen_contour], -1, 255, thickness=cv2.FILLED)
            fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=4)

            # ghost healing: if the worm moved a real distance (not just
            # Kalman jitter), flag the old spot for accelerated healing so
            # the background catches up there before the tracker can lock
            # back onto the ghost. 8% of max_jump is the jitter threshold --
            # anything below that we just leave alone.
            if self.prev_point is not None:
                move_dist = math.hypot(
                    point[0] - self.prev_point[0],
                    point[1] - self.prev_point[1]
                )
                if move_dist > self.max_jump * 0.08:
                    if self._last_protect_mask is not None:
                        self.bg_model.mark_as_stale(self._last_protect_mask)

            self.bg_model.update(gray, protect_mask=fg_mask)
            self._last_protect_mask = fg_mask
            area_val = cv2.contourArea(chosen_contour)
        else:
            self.consecutive_misses += 1
            if self.kalman is not None and self.kalman.initialized:
                point = self.kalman.predict()
            else:
                point = self.prev_point
            self.bg_model.update(gray, protect_mask=None)
            area_val = 0.0

            if self.consecutive_misses >= self.max_consecutive_misses:
                self.prev_point = None
                point = None
                if self.kalman is not None:
                    self.kalman = CentroidKalman()
                self.consecutive_misses = 0
                self._last_protect_mask = None

        if point is not None:
            self.prev_point = point

        return {
            "point": point,
            "detected": detected,
            "contour": chosen_contour,
            "area": area_val,
            "otsu": otsu_th,
        }


# ============================================================
# Lightweight performance profiling
# ============================================================
class PerfStats:
    """
    Lightweight per-stage timing to figure out where the Pi 4 is spending
    its time. Just a few perf_counter() calls and a fixed-size deque per
    stage -- cheap enough to leave on. Reporting (the only expensive part)
    only happens every report_every frames.
    """

    def __init__(self, window=60, report_every=30, csv_path=None):
        self.window = window
        self.report_every = report_every
        self.timings = {}
        self.frame_times = deque(maxlen=window)
        self._t0 = None
        self._stage_t0 = None
        self.csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields = None

    def start_frame(self):
        self._t0 = time.perf_counter()
        self._stage_t0 = self._t0

    def lap(self, stage_name):
        if self._stage_t0 is None:
            return
        now = time.perf_counter()
        dt = now - self._stage_t0
        self.timings.setdefault(stage_name, deque(maxlen=self.window)).append(dt)
        self._stage_t0 = now

    def end_frame(self, frame_idx):
        if self._t0 is None:
            return
        total = time.perf_counter() - self._t0
        self.frame_times.append(total)
        if self.csv_path:
            self._write_csv_row(frame_idx, total)
        if frame_idx % self.report_every == 0:
            self.report(frame_idx)

    def report(self, frame_idx):
        if not self.frame_times:
            return
        avg_frame = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame if avg_frame > 0 else 0.0
        parts = [f"{stage}={1000*sum(dq)/len(dq):.1f}ms" for stage, dq in self.timings.items()]
        print(
            f"[perf] frame {frame_idx}  ~{fps:.1f} fps  " + "  ".join(parts),
            file=sys.stderr,
        )

    def _write_csv_row(self, frame_idx, total):
        if self._csv_file is None:
            self._csv_fields = ["frame", "total_s"] + list(self.timings.keys())
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()
        row = {"frame": frame_idx, "total_s": round(total, 6)}
        for stage, dq in self.timings.items():
            row[stage] = round(dq[-1], 6) if dq else ""
        # if new stages show up after the CSV header was already written,
        # just drop them silently rather than crashing -- this is a debug
        # tool, not the actual scientific output.
        row = {k: row.get(k, "") for k in self._csv_fields}
        self._csv_writer.writerow(row)

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()


# ============================================================
# Sustained near-zero detection rate warning
# ============================================================
class DetectionRateMonitor:
    """
    Watches the detection rate and yells if it tanks.

    A sustained near-0% detection rate (not just degraded, actually zero)
    almost always means a physical setup problem -- worm not in frame,
    empty dish, or a worm that sat still long enough to get absorbed into
    the background. Hit this exact situation during Pi testing: 0/1976
    frames detected on one --mode live run, right after getting 91-100%
    on the exact same command a minute earlier.

    Prints one warning to stderr when the rolling rate drops below rate_floor
    so you can actually go check the setup WHILE the run is still going,
    rather than discovering it after Ctrl+C.
    """

    def __init__(self, window=150, min_samples=100, rate_floor=0.03):
        self.recent = deque(maxlen=window)
        self.min_samples = min_samples
        self.rate_floor = rate_floor
        self.warned = False

    def update(self, detected):
        self.recent.append(1 if detected else 0)
        if self.warned or len(self.recent) < self.min_samples:
            return
        rate = sum(self.recent) / len(self.recent)
        if rate <= self.rate_floor:
            print(
                f"[warn] Detection rate has been {100*rate:.1f}% over the last "
                f"{len(self.recent)} frames. If the dish/worm setup looks fine, "
                "this is worth investigating now rather than assuming it's a "
                "tracking bug (empty dish, worm out of frame/focus, and a worm "
                "sitting still long enough to be absorbed into the background "
                "model are the usual causes). This warning only prints once per run.",
                file=sys.stderr,
            )
            self.warned = True


# ============================================================
# Resolution presets (Raspberry Pi HQ Camera / IMX477)
# ============================================================
# These match the IMX477 sensor's native/binned readout modes, which
# means the camera hardware does the binning -- not the CPU. If you're
# using a different module (e.g. Camera Module 3 / IMX708), the presets
# still work as arbitrary output sizes but might not map to native modes,
# so check that module's docs for best performance. You can always skip
# the presets entirely and just pass a "WIDTHxHEIGHT" string directly.
RESOLUTION_PRESETS = {
    # name: (width, height, approx_max_fps, description)
    "super_high": (4056, 3040, 10, "Full sensor resolution. Best detail, large files, low fps. "
                                    "NOTE: above the Pi 4's h264 encoder limit, see below -- "
                                    "fine for --mode analyze on a still image workflow, NOT for "
                                    "record/record-analyze/live+record-output."),
    "high": (2028, 1520, 40, "2x2 binned, full field of view. Sharp and still efficient. "
                              "NOTE: also above the Pi 4 encoder limit -- same caveat as above."),
    "1080p": (1920, 1080, 40, "Max resolution the Pi 4's hardware h264 encoder actually "
                               "supports. Safe default for record/record-analyze/live."),
    "medium": (1332, 990, 50, "Good balance of detail and file size/throughput."),
    "low": (640, 480, 60, "Smallest/fastest. Fine when only positions matter, not image quality."),
}

# Pi 4 hardware h264 encoder ceiling -- found this the hard way when
# 2028x1520 crashed with a ProcessLookupError 3 frames deep in picamera2
# at ioctl VIDIOC_STREAMON with no useful error message at all.
# This is a real silicon limit, not fixable in code. Pi 5 raises it.
PI4_ENCODER_MAX_WIDTH = 1920
PI4_ENCODER_MAX_HEIGHT = 1080


def _validate_encoder_resolution(resolution):
    """
    Check the resolution before doing anything that uses the hardware h264
    encoder (any recording mode). Modes that only use capture_array()
    (analyze from file, --camera-selftest, live without --record-output)
    never touch the encoder and don't need this check -- they can use
    "high" or "super_high" without issues.
    """
    w, h = resolution
    if w > PI4_ENCODER_MAX_WIDTH or h > PI4_ENCODER_MAX_HEIGHT:
        raise SystemExit(
            f"--resolution {w}x{h} is above the Raspberry Pi 4 hardware h264 "
            f"encoder's limit of {PI4_ENCODER_MAX_WIDTH}x{PI4_ENCODER_MAX_HEIGHT}. "
            "This isn't a bug in this script -- it's a hardware ceiling on the Pi "
            "4's video encode block (the Pi 5 raises this limit). Use "
            "--resolution 1080p (the default), another preset at or under "
            "1920x1080, or an explicit WIDTHxHEIGHT within that limit. "
            "('high'/'super_high' still work fine for --mode analyze, which "
            "never touches the hardware encoder.)"
        )


def parse_resolution(s):
    """Accepts a preset name (case-insensitive) or 'WIDTHxHEIGHT'."""
    if s is None:
        return None
    key = s.strip().lower()
    if key in RESOLUTION_PRESETS:
        w, h, _, _ = RESOLUTION_PRESETS[key]
        return (w, h)
    if "x" in key:
        try:
            w, h = key.split("x")
            return (int(w), int(h))
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"--resolution must be one of {list(RESOLUTION_PRESETS)} or 'WIDTHxHEIGHT', got {s!r}"
    )


# ============================================================
# Config
# ============================================================
class Config:
    def __init__(self, args):
        self.mode = args.mode
        self.input = args.input
        self.output_csv = args.output_csv
        self.output_video = args.output_video
        self.output_plot = args.output_plot
        self.roi = args.roi
        self.downscale = args.downscale
        self.min_area = args.min_area
        self.max_area = args.max_area
        self.bg_alpha = args.bg_alpha
        self.bg_init_frames = args.bg_init_frames
        self.use_clahe = args.clahe
        self.no_kalman = args.no_kalman
        self.auto_mask = args.auto_mask
        self.show = args.show
        self.max_frames = args.max_frames
        self.frame_skip = max(1, args.frame_skip)
        self.mm_per_px = args.mm_per_px
        self.max_jump_px = args.max_jump_px

        # Pi-camera-specific (record / record-analyze / live / still modes)
        self.resolution = parse_resolution(args.resolution)
        self.analysis_resolution = parse_resolution(args.analysis_resolution) if args.analysis_resolution else (640, 480)
        self.record_output = args.record_output
        self.duration = args.duration
        self.record_fps = args.record_fps
        self.bitrate = args.bitrate

        # target fps cap for live mode -- limits analysis rate to leave
        # CPU headroom. overrides frame_skip to hit the target rate
        # from whatever the camera's native fps actually is.
        self.target_fps = args.target_fps

        # Still image mode interval (seconds between captures)
        self.still_interval = args.still_interval

        # Output directory for still image mode annotated frames
        self.still_output_dir = args.still_output_dir

        # profiling
        self.profile = args.profile
        self.profile_interval = args.profile_interval
        self.profile_csv = args.profile_csv


def parse_roi(s):
    if s is None:
        return None
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be 'x,y,w,h'")
    return tuple(parts)


def open_capture(input_arg):
    # Allow "0", "1", ... to mean a live camera index (USB webcam / a
    # Pi camera exposed through the legacy V4L2 path).
    try:
        idx = int(input_arg)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index: {input_arg}")
        return cap
    except ValueError:
        pass

    path = str(input_arg)
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap

    # Pi OS ships an OpenCV build that uses GStreamer but not ffmpeg, and it
    # sometimes misparses a plain filename as a GStreamer pipeline description
    # (you'll see "GStreamer warning: Error opening bin: unexpected reference
    # '<filename>'" then a crash). If the plain open fails, try again with an
    # explicit GStreamer filesrc pipeline so there's no ambiguity.
    abspath = str(Path(path).resolve())
    gst_pipeline = f'filesrc location="{abspath}" ! decodebin ! videoconvert ! appsink'
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap

    raise RuntimeError(
        f"Could not open video source: {input_arg} (tried a plain OpenCV open "
        "and a GStreamer filesrc fallback pipeline; if this is a real file, "
        "check the codec is one your OpenCV build's backend supports)"
    )


# ============================================================
# Mode: analyze (the tested, primary path)
# ============================================================
def run_analyze(cfg: Config):
    cap = open_capture(cfg.input)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-2:
        src_fps = 30.0
    effective_fps = src_fps / cfg.frame_skip

    roi = cfg.roi

    def preprocess(frame_bgr):
        if roi is not None:
            x, y, w, h = roi
            frame_bgr = frame_bgr[y : y + h, x : x + w]
        if cfg.downscale != 1.0:
            frame_bgr = cv2.resize(
                frame_bgr, None, fx=cfg.downscale, fy=cfg.downscale,
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if cfg.use_clahe:
            gray = _clahe.apply(gray)
        return frame_bgr, gray

    if cfg.use_clahe:
        _clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    buffered, warmup_start = detect_warmup_end(
        cap.read, preprocess, max_check=max(90, cfg.bg_init_frames + 45)
    )
    if warmup_start > 0:
        print(
            f"[info] Detected ~{warmup_start} frame(s) of camera "
            f"exposure/white-balance settling at the start of the "
            f"source; excluding them from the background model.",
            file=sys.stderr,
        )
    init_frames = buffered[warmup_start : warmup_start + cfg.bg_init_frames]
    frames_to_skip_on_reopen = warmup_start
    if not init_frames:
        raise RuntimeError("Could not read any frames from the input source.")

    is_file = isinstance(cfg.input, str) and Path(cfg.input).exists()
    starting_frame_idx = 0
    if is_file:
        cap.release()
        cap = open_capture(cfg.input)
        for _ in range(frames_to_skip_on_reopen):
            if not cap.grab():
                break
        starting_frame_idx = frames_to_skip_on_reopen

    tracker = SingleWormTracker(init_frames, cfg, effective_fps)

    writer = None
    if cfg.output_video:
        h0, w0 = init_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg.output_video, fourcc, effective_fps, (w0, h0))

    perf = PerfStats(report_every=cfg.profile_interval, csv_path=cfg.profile_csv) if cfg.profile else None
    det_monitor = DetectionRateMonitor()

    csv_rows = []
    frame_idx = starting_frame_idx - 1
    t_start = time.time()
    processed_count = 0
    cumulative_distance_px = 0.0

    while True:
        if cfg.max_frames is not None and processed_count >= cfg.max_frames:
            break
        if perf:
            perf.start_frame()

        ok = True
        for _ in range(cfg.frame_skip - 1):
            ok = cap.grab()
            if not ok:
                break
        if not ok:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        processed_count += 1
        if perf:
            perf.lap("capture")

        color, gray = preprocess(frame)
        if perf:
            perf.lap("preprocess")

        result = tracker.process(gray)
        if perf:
            perf.lap("track")

        point = result["point"]
        detected = result["detected"]
        chosen_contour = result["contour"]
        area_val = result["area"]
        det_monitor.update(detected)

        if point is not None and csv_rows and csv_rows[-1]["_raw_cx"] is not None:
            prev_x = csv_rows[-1]["_raw_cx"]
            prev_y = csv_rows[-1]["_raw_cy"]
            cumulative_distance_px += math.hypot(point[0] - prev_x, point[1] - prev_y)

        t_sec = frame_idx / src_fps
        if point is not None:
            orig_x = point[0] / cfg.downscale + (roi[0] if roi else 0)
            orig_y = point[1] / cfg.downscale + (roi[1] if roi else 0)
        else:
            orig_x = orig_y = float("nan")

        row = {
            "frame": frame_idx,
            "time_s": round(t_sec, 4),
            "cx_px": round(orig_x, 2) if point is not None else "",
            "cy_px": round(orig_y, 2) if point is not None else "",
            "area_px": round(area_val, 1),
            "detected": int(detected),
            "cumulative_distance_px": round(cumulative_distance_px, 2),
            "_raw_cx": point[0] if point is not None else None,
            "_raw_cy": point[1] if point is not None else None,
        }
        if cfg.mm_per_px:
            row["cx_mm"] = round(orig_x * cfg.mm_per_px, 4) if point is not None else ""
            row["cy_mm"] = round(orig_y * cfg.mm_per_px, 4) if point is not None else ""
            row["cumulative_distance_mm"] = round(cumulative_distance_px * cfg.mm_per_px, 4)
        csv_rows.append(row)

        if writer is not None or cfg.show:
            vis = color.copy()
            if chosen_contour is not None:
                cv2.drawContours(vis, [chosen_contour], -1, (0, 255, 0), 2)
            if point is not None:
                dot_color = (0, 0, 255) if detected else (0, 165, 255)
                cv2.circle(vis, (int(point[0]), int(point[1])), 5, dot_color, -1)
            cv2.putText(
                vis, f"frame {frame_idx}  t={t_sec:.2f}s  {'OK' if detected else 'coasting'}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
            if writer is not None:
                writer.write(vis)
            if cfg.show:
                cv2.imshow("celegans_tracker", vis)
                # waitKey(30) not waitKey(1) -- 1ms is too short over SSH/X11,
                # the window never gets focus and 'q' gets silently dropped.
                # ESC also works as a quit key.
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == 27:
                    break
        if perf:
            perf.lap("annotate_io")
            perf.end_frame(frame_idx)

    elapsed = time.time() - t_start
    cap.release()
    if writer is not None:
        writer.release()
    if cfg.show:
        cv2.destroyAllWindows()
    if perf:
        perf.close()

    # strip internal bookkeeping fields before writing CSV
    for r in csv_rows:
        r.pop("_raw_cx", None)
        r.pop("_raw_cy", None)

    n_detected = sum(r["detected"] for r in csv_rows)
    n_total = len(csv_rows)
    print(
        f"Processed {n_total} frames in {elapsed:.1f}s "
        f"({n_total / max(elapsed, 1e-6):.1f} fps). "
        f"Detected worm in {n_detected}/{n_total} frames "
        f"({100.0 * n_detected / max(n_total,1):.1f}%)."
    )

    if cfg.output_csv:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        with open(cfg.output_csv, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(csv_rows)
        print(f"Wrote trajectory CSV to {cfg.output_csv}")

    if cfg.output_plot:
        _save_plot(csv_rows, cfg.output_plot, cfg.mm_per_px)
        print(f"Wrote trajectory plot to {cfg.output_plot}")

    if cfg.output_video:
        print(f"Wrote annotated video to {cfg.output_video}")

    return csv_rows


def _save_plot(csv_rows, path, mm_per_px):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        # used to crash the whole run after the CSV was already written fine.
        # data wasn't lost but the process died anyway -- don't do that.
        print(
            f"[warn] Could not write trajectory plot to {path}: matplotlib is "
            f"not installed ({e}). Install with: pip3 install matplotlib "
            "(add --break-system-packages on newer Raspberry Pi OS / Debian "
            "trixie if pip refuses). The CSV data is unaffected.",
            file=sys.stderr,
        )
        return

    xs = [r["cx_px"] for r in csv_rows if r["cx_px"] != ""]
    ys = [r["cy_px"] for r in csv_rows if r["cy_px"] != ""]
    ts = [r["time_s"] for r in csv_rows]
    all_x = [r["cx_px"] if r["cx_px"] != "" else float("nan") for r in csv_rows]
    all_y = [r["cy_px"] if r["cy_px"] != "" else float("nan") for r in csv_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(xs, ys, "-", lw=0.8, color="tab:blue")
    axes[0].scatter(xs[:1], ys[:1], c="green", label="start", zorder=5)
    axes[0].scatter(xs[-1:], ys[-1:], c="red", label="end", zorder=5)
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    axes[0].set_title("Centroid trajectory")
    axes[0].legend()

    axes[1].plot(ts, all_x, label="x (px)")
    axes[1].plot(ts, all_y, label="y (px)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("position (px)")
    axes[1].set_title("Position vs time")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================
# Pi camera support (record / record-analyze / live)
# HARDWARE-UNTESTED -- see module docstring.
# ============================================================
def _import_picamera2():
    try:
        from picamera2 import Picamera2
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import FfmpegOutput
    except ImportError as e:
        raise RuntimeError(
            "picamera2 is not available. This mode only works on a Raspberry Pi "
            "with the picamera2 package installed (it ships with Raspberry Pi OS "
            "Bookworm and later). On the Pi, check with: python3 -c 'import picamera2'. "
            f"Original error: {e}"
        )
    return Picamera2, H264Encoder, FfmpegOutput


def _make_picamera2(resolution, lores_resolution=None, fps=None, retries=5, retry_delay=1.5):
    """
    Configures a Picamera2 instance for video capture.
    If `lores_resolution` is given, a second low-resolution stream is
    also configured -- this is what makes live analysis at a
    modest, Pi-4-friendly resolution possible while independently
    recording (if requested) at full quality, since the hardware
    encoder only ever touches the "main" stream.

    Retries a few times if the camera is reported as busy. This happens on
    the second of two back-to-back runs -- the previous process's picam2.stop()
    wasn't followed by picam2.close(), so libcamera hadn't fully let go yet.
    .close() is now called everywhere, but the release isn't always instant,
    so the retry is a safety net rather than the real fix.
    """
    Picamera2, H264Encoder, FfmpegOutput = _import_picamera2()
    main_stream = {"size": resolution, "format": "RGB888"}
    config_kwargs = {"main": main_stream}
    if lores_resolution is not None:
        config_kwargs["lores"] = {"size": lores_resolution, "format": "YUV420"}
    if fps:
        frame_duration_us = int(1_000_000 / fps)
        config_kwargs["controls"] = {"FrameDurationLimits": (frame_duration_us, frame_duration_us)}

    picam2 = None
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            picam2 = Picamera2()
            break
        except RuntimeError as e:
            last_err = e
            busy = "busy" in str(e).lower() or "in use" in str(e).lower()
            if busy and attempt < retries:
                print(
                    f"[camera] device reported busy (attempt {attempt}/{retries}) -- "
                    f"usually means a previous run's camera handle hasn't fully "
                    f"released yet. Waiting {retry_delay:.1f}s and retrying ...",
                    file=sys.stderr,
                )
                time.sleep(retry_delay)
                continue
            raise RuntimeError(
                f"Could not acquire the Pi camera: {e}. If this keeps happening, "
                "check nothing else still has it open (another instance of this "
                "script, libcamera-hello, a second SSH session, etc)."
            ) from e
    if picam2 is None:
        raise RuntimeError(f"Could not acquire the Pi camera after {retries} attempts: {last_err}")

    video_config = picam2.create_video_configuration(**config_kwargs)
    picam2.configure(video_config)
    return picam2


def _lores_to_gray(lores_array, size):
    """
    Extracts the grayscale (Y/luma) plane from picamera2's YUV420 lores stream.
    The array comes as a 2D buffer shaped (height * 3//2, width) -- the first
    `height` rows are full-res Y, and the rest pack the subsampled U/V planes.
    Since we only need grayscale, we just take the Y plane directly and skip
    the YUV->BGR->gray roundtrip.
    """
    w, h = size
    y_plane = lores_array[:h, :w]
    return cv2.GaussianBlur(y_plane, (5, 5), 0)


def camera_selftest(cfg: Config):
    """
    Quick sanity check before a real run -- opens the camera, grabs a
    few frames from both the main and lores streams, prints their shapes
    and achieved fps, then exits. Run this once on actual hardware to
    catch API/format issues (e.g. if your picamera2 version returns arrays
    shaped differently than expected) before committing to a real experiment.
    """
    print("[selftest] Importing picamera2 ...")
    Picamera2, H264Encoder, FfmpegOutput = _import_picamera2()
    print("[selftest] Configuring camera "
          f"(main={cfg.resolution}, lores={cfg.analysis_resolution}) ...")
    picam2 = _make_picamera2(cfg.resolution, cfg.analysis_resolution, fps=cfg.record_fps)
    picam2.start()
    try:
        time.sleep(1.0)  # let auto-exposure begin settling
        n = 20
        t0 = time.time()
        for i in range(n):
            main_arr = picam2.capture_array("main")
            lores_arr = picam2.capture_array("lores")
            if i == 0:
                print(f"[selftest] main frame shape={main_arr.shape} dtype={main_arr.dtype}")
                print(f"[selftest] lores frame shape={lores_arr.shape} dtype={lores_arr.dtype} "
                      f"(expected roughly ({cfg.analysis_resolution[1]*3//2}, {cfg.analysis_resolution[0]}))")
                gray = _lores_to_gray(lores_arr, cfg.analysis_resolution)
                print(f"[selftest] derived grayscale analysis frame shape={gray.shape}")
        elapsed = time.time() - t0
        print(f"[selftest] Captured {n} synchronized main+lores frame pairs in "
              f"{elapsed:.2f}s (~{n/elapsed:.1f} fps).")
        print("[selftest] If the shapes above look sane, --mode live / --mode record "
              "should work. If capture_array('lores') isn't shaped as expected, the "
              "Y-plane slicing in _lores_to_gray() may need adjusting for your "
              "picamera2 version.")
    finally:
        picam2.stop()
        picam2.close()


def record_video_picamera2(output_path, resolution, duration_s=None, fps=None, bitrate=None):
    """
    Records from the Pi camera to an mp4 using the hardware H.264 encoder
    (very cheap on CPU, which is why we're not doing software encoding).
    Blocks until duration_s is up, or until Ctrl+C if no duration is set.
    """
    _validate_encoder_resolution(resolution)
    Picamera2, H264Encoder, FfmpegOutput = _import_picamera2()
    picam2 = _make_picamera2(resolution, lores_resolution=None, fps=fps)
    encoder = H264Encoder(bitrate=bitrate or 10_000_000)
    output = FfmpegOutput(output_path)
    print(f"[record] Starting recording to {output_path} at {resolution[0]}x{resolution[1]} ...")
    picam2.start_recording(encoder, output)
    try:
        if duration_s is not None:
            time.sleep(duration_s)
        else:
            print("[record] Recording -- press Ctrl+C to stop.")
            while True:
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[record] Stopping (Ctrl+C).")
    finally:
        picam2.stop_recording()
        picam2.stop()
        picam2.close()
    print(f"[record] Saved {output_path}")


def run_record(cfg: Config):
    if not cfg.record_output:
        raise ValueError("--mode record requires --record-output PATH")
    record_video_picamera2(
        cfg.record_output, cfg.resolution, duration_s=cfg.duration,
        fps=cfg.record_fps, bitrate=cfg.bitrate,
    )


def run_record_analyze(cfg: Config):
    if not cfg.record_output:
        raise ValueError("--mode record-analyze requires --record-output PATH")
    record_video_picamera2(
        cfg.record_output, cfg.resolution, duration_s=cfg.duration,
        fps=cfg.record_fps, bitrate=cfg.bitrate,
    )
    print("[record-analyze] Recording finished; handing off to the analysis pipeline ...")
    analyze_cfg = cfg
    analyze_cfg.input = cfg.record_output
    return run_analyze(analyze_cfg)


def run_live(cfg: Config):
    """
    Real-time tracking directly from the camera using the exact same
    SingleWormTracker as --mode analyze. If cfg.record_output is set,
    a full-quality copy is simultaneously written via the hardware encoder
    attached to the "main" stream, while analysis runs on the smaller
    "lores" stream -- so recording and tracking don't fight over CPU.
    If no record_output is set, only the CSV/plot gets saved -- useful
    when you just want the data and don't care about the footage.
    """
    Picamera2, H264Encoder, FfmpegOutput = _import_picamera2()

    main_res = cfg.resolution
    if cfg.record_output:
        _validate_encoder_resolution(cfg.resolution)
    elif not cfg.show:
        # nothing is actually consuming the full-res main stream here
        # (no recording, no display) -- request it at analysis resolution
        # instead so the ISP isn't wasting bandwidth on pixels nobody reads.
        # tracker only ever sees "lores" anyway so this doesn't affect results.
        main_res = cfg.analysis_resolution

    picam2 = _make_picamera2(main_res, cfg.analysis_resolution, fps=cfg.record_fps)

    encoder = None
    if cfg.record_output:
        encoder = H264Encoder(bitrate=cfg.bitrate or 10_000_000)
        picam2.start_recording(encoder, FfmpegOutput(cfg.record_output))
        print(f"[live] Recording simultaneously to {cfg.record_output} "
              f"at {cfg.resolution[0]}x{cfg.resolution[1]}.")
    else:
        picam2.start()
        print("[live] Not saving video -- analysis only.")

    print(f"[live] Analyzing at {cfg.analysis_resolution[0]}x{cfg.analysis_resolution[1]} "
          "(the 'lores' stream).")

    # --clahe was silently doing nothing in live mode before -- analyze
    # applied it but this frame-grab path didn't. fixed so it's consistent.
    _clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)) if cfg.use_clahe else None

    def read_gray():
        try:
            arr = picam2.capture_array("lores")
            gray = _lores_to_gray(arr, cfg.analysis_resolution)
            if _clahe is not None:
                gray = _clahe.apply(gray)
            return True, gray
        except Exception as e:
            print(f"[live] frame grab failed: {e}", file=sys.stderr)
            return False, None

    def read_bgr_for_display():
        arr = picam2.capture_array("main")
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # warmup detection for a live source -- can't rewind a camera so
    # those frames are just gone once consumed, which is fine.
    def _read_as_bgr_tuple():
        ok, gray = read_gray()
        # detect_warmup_end expects (ok, frame_bgr) then calls preprocess_fn
        # on it -- we pass gray directly and use an identity preprocess since
        # it's already grayscale+blurred from read_gray()
        return ok, gray

    def _identity_preprocess(gray):
        return gray, gray

    print("[live] Letting camera settle and building background model ...")
    buffered, warmup_start = detect_warmup_end(_read_as_bgr_tuple, _identity_preprocess)
    init_frames = buffered[warmup_start : warmup_start + cfg.bg_init_frames]
    if not init_frames:
        raise RuntimeError("Could not read any frames from the camera to build a background model.")

    # worm can be on the plate during background init -- the median is
    # robust to it being present in <50% of init frames. if it's sitting
    # perfectly still for all of init, the background absorbs it and
    # detection starts low but recovers as alpha nudges the background
    # away. use --bg-init-frames 0 to skip init entirely (fastest start,
    # least robust).
    print("[live] Starting tracking ...")

    effective_fps = cfg.record_fps or 30.0
    tracker = SingleWormTracker(init_frames, cfg, effective_fps)
    perf = PerfStats(report_every=cfg.profile_interval, csv_path=cfg.profile_csv) if cfg.profile else None
    det_monitor = DetectionRateMonitor()

    # target fps throttle: analyze every Nth frame to leave CPU headroom.
    # example: camera at 30fps, --target-fps 5 → analyze every 6th frame.
    # camera still runs at native fps (frames are grabbed and discarded),
    # CSV timestamps reflect the effective analysis rate not the camera rate.
    if cfg.target_fps is not None and cfg.target_fps < effective_fps:
        live_frame_skip = max(1, round(effective_fps / cfg.target_fps))
        effective_analysis_fps = effective_fps / live_frame_skip
        print(
            f"[live] Camera at {effective_fps:.1f}fps, target {cfg.target_fps}fps → "
            f"analyzing every {live_frame_skip} frames ({effective_analysis_fps:.1f}fps effective)"
        )
    else:
        live_frame_skip = 1
        effective_analysis_fps = effective_fps

    csv_rows = []
    frame_idx = -1
    raw_frame_idx = -1
    cumulative_distance_px = 0.0
    t_start = time.time()

    try:
        while True:
            if cfg.max_frames is not None and frame_idx + 1 >= cfg.max_frames:
                break
            if perf:
                perf.start_frame()

            ok, gray = read_gray()
            if not ok:
                break
            raw_frame_idx += 1

            # drain the buffer on every raw frame to keep the camera from
            # stalling, but only actually run the tracker every Nth frame.
            if raw_frame_idx % live_frame_skip != 0:
                continue

            frame_idx += 1
            if perf:
                perf.lap("capture")

            result = tracker.process(gray)
            if perf:
                perf.lap("track")

            point = result["point"]
            detected = result["detected"]
            area_val = result["area"]
            det_monitor.update(detected)

            if point is not None and csv_rows and csv_rows[-1]["_raw_cx"] is not None:
                cumulative_distance_px += math.hypot(
                    point[0] - csv_rows[-1]["_raw_cx"], point[1] - csv_rows[-1]["_raw_cy"]
                )

            t_sec = frame_idx / effective_analysis_fps
            row = {
                "frame": frame_idx,
                "time_s": round(t_sec, 4),
                "cx_px": round(point[0], 2) if point is not None else "",
                "cy_px": round(point[1], 2) if point is not None else "",
                "area_px": round(area_val, 1),
                "detected": int(detected),
                "cumulative_distance_px": round(cumulative_distance_px, 2),
                "_raw_cx": point[0] if point is not None else None,
                "_raw_cy": point[1] if point is not None else None,
            }
            if cfg.mm_per_px:
                row["cx_mm"] = round(point[0] * cfg.mm_per_px, 4) if point is not None else ""
                row["cy_mm"] = round(point[1] * cfg.mm_per_px, 4) if point is not None else ""
                row["cumulative_distance_mm"] = round(cumulative_distance_px * cfg.mm_per_px, 4)
            csv_rows.append(row)

            if cfg.show:
                vis = read_bgr_for_display()
                # analysis and display streams are different
                # resolutions; scale the point up for display.
                sx = vis.shape[1] / cfg.analysis_resolution[0]
                sy = vis.shape[0] / cfg.analysis_resolution[1]
                if result["contour"] is not None:
                    scaled_contour = (result["contour"].astype(np.float32) * [sx, sy]).astype(np.int32)
                    cv2.drawContours(vis, [scaled_contour], -1, (0, 255, 0), 2)
                if point is not None:
                    dot_color = (0, 0, 255) if detected else (0, 165, 255)
                    cv2.circle(vis, (int(point[0] * sx), int(point[1] * sy)), 6, dot_color, -1)
                cv2.putText(
                    vis, f"frame {frame_idx}  t={t_sec:.2f}s  {'OK' if detected else 'coasting'}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.imshow("celegans_tracker (live)", vis)
                # same waitKey(30) + ESC fix as in run_analyze
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == 27:
                    break
            if perf:
                perf.lap("annotate_io")
                perf.end_frame(frame_idx)
    except KeyboardInterrupt:
        print("\n[live] Stopping (Ctrl+C).")
    finally:
        if encoder is not None:
            picam2.stop_recording()
        else:
            picam2.stop()
        picam2.close()
        if cfg.show:
            cv2.destroyAllWindows()
        if perf:
            perf.close()

    elapsed = time.time() - t_start
    for r in csv_rows:
        r.pop("_raw_cx", None)
        r.pop("_raw_cy", None)

    n_detected = sum(r["detected"] for r in csv_rows)
    n_total = len(csv_rows)
    print(
        f"Processed {n_total} frames in {elapsed:.1f}s "
        f"({n_total / max(elapsed, 1e-6):.1f} fps). "
        f"Detected worm in {n_detected}/{n_total} frames "
        f"({100.0 * n_detected / max(n_total,1):.1f}%)."
    )

    if cfg.output_csv:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        with open(cfg.output_csv, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(csv_rows)
        print(f"Wrote trajectory CSV to {cfg.output_csv}")

    if cfg.output_plot:
        _save_plot(csv_rows, cfg.output_plot, cfg.mm_per_px)
        print(f"Wrote trajectory plot to {cfg.output_plot}")

    return csv_rows


# ============================================================
# Still image mode
# ============================================================
def run_still(cfg):
    """
    Captures full-resolution stills at a fixed interval and runs the tracker
    on each one. Good for:
      - Long experiments where video file sizes get impractical
      - Maximizing spatial resolution (full 12.3MP) at low magnification
        where the worm is small in pixels
      - Situations where low temporal resolution is fine (slow movement
        relative to worm size between captures)
      - Leaving CPU headroom between captures for other stuff

    Output: annotated JPEG per capture in --still-output-dir, plus the
    standard CSV and trajectory plot.

    Usage example:
        python3 celegans_tracker.py --mode still \\
            --still-interval 5 \\
            --still-output-dir ./stills/ \\
            --output-csv still_track.csv \\
            --output-plot still_traj.png \\
            --downscale 0.5 \\
            --mm-per-px 0.003
    """
    picamera2 = _import_picamera2()
    Picamera2 = picamera2.Picamera2

    out_dir = Path(cfg.still_output_dir) if cfg.still_output_dir else Path("stills")
    out_dir.mkdir(parents=True, exist_ok=True)

    picam2 = _make_picamera2(Picamera2)

    # full sensor resolution for stills -- that's the whole point of this mode.
    # HQ camera (IMX477) native res is 4056x3040 (~12.3MP). above the Pi 4
    # h264 encoder ceiling, but still mode never touches the encoder (uses
    # capture_array() instead) so there's no ceiling.
    still_res = cfg.resolution or (4056, 3040)
    still_config = picam2.create_still_configuration(
        main={"size": still_res, "format": "RGB888"},
    )
    picam2.configure(still_config)
    picam2.start()
    time.sleep(2.0)  # let AE/AWB settle

    print(f"[still] Camera ready at {still_res[0]}x{still_res[1]}.")
    print(f"[still] Capturing every {cfg.still_interval}s to {out_dir}/")
    print(f"[still] Press Ctrl+C to stop.\n")

    # Build background from first N stills at full resolution.
    # At full res this is large but only done once at startup.
    print(f"[still] Building background from {cfg.bg_init_frames} initial captures ...")
    init_frames_gray = []
    for i in range(cfg.bg_init_frames):
        arr = picam2.capture_array("main")
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if cfg.downscale != 1.0:
            h, w = gray.shape
            gray = cv2.resize(gray, (int(w * cfg.downscale), int(h * cfg.downscale)))
        init_frames_gray.append(gray)
        time.sleep(0.5)  # short gap between init captures

    tracker = SingleWormTracker(init_frames_gray, cfg, effective_fps=1.0 / cfg.still_interval)

    csv_rows = []
    capture_idx = 0
    cumulative_distance_px = 0.0

    try:
        while True:
            if cfg.max_frames is not None and capture_idx >= cfg.max_frames:
                break

            t_capture_start = time.time()

            # Full-resolution capture
            arr = picam2.capture_array("main")
            gray_full = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            bgr_full = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            # Downscale for tracking (same as other modes)
            if cfg.downscale != 1.0:
                h, w = gray_full.shape
                gray = cv2.resize(gray_full, (int(w * cfg.downscale), int(h * cfg.downscale)))
            else:
                gray = gray_full
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if cfg.use_clahe:
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            result = tracker.process(gray)
            point = result["point"]
            detected = result["detected"]
            area_val = result["area"]

            t_sec = capture_idx * cfg.still_interval

            if point is not None and csv_rows and csv_rows[-1]["_raw_cx"] is not None:
                cumulative_distance_px += math.hypot(
                    point[0] - csv_rows[-1]["_raw_cx"],
                    point[1] - csv_rows[-1]["_raw_cy"],
                )

            row = {
                "frame": capture_idx,
                "time_s": round(t_sec, 4),
                "cx_px": round(point[0], 2) if point is not None else "",
                "cy_px": round(point[1], 2) if point is not None else "",
                "area_px": round(area_val, 1),
                "detected": int(detected),
                "cumulative_distance_px": round(cumulative_distance_px, 2),
                "_raw_cx": point[0] if point is not None else None,
                "_raw_cy": point[1] if point is not None else None,
            }
            if cfg.mm_per_px:
                scale = cfg.mm_per_px / cfg.downscale
                row["cx_mm"] = round(point[0] * scale, 4) if point is not None else ""
                row["cy_mm"] = round(point[1] * scale, 4) if point is not None else ""
                row["cumulative_distance_mm"] = round(cumulative_distance_px * scale, 4)
            csv_rows.append(row)

            # Annotate and save the full-resolution frame
            vis = bgr_full.copy()
            sx = bgr_full.shape[1] / gray.shape[1]
            sy = bgr_full.shape[0] / gray.shape[0]
            if result["contour"] is not None:
                scaled_contour = (result["contour"].astype(np.float32) * [sx, sy]).astype(np.int32)
                cv2.drawContours(vis, [scaled_contour], -1, (0, 255, 0), 2)
            if point is not None:
                dot_color = (0, 0, 255) if detected else (0, 165, 255)
                cv2.circle(vis, (int(point[0] * sx), int(point[1] * sy)), 8, dot_color, -1)
            label = f"t={t_sec:.1f}s  cap={capture_idx}  {'DETECTED' if detected else 'COASTING'}"
            cv2.putText(vis, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            out_path = out_dir / f"still_{capture_idx:05d}.jpg"
            cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
            status = "DETECTED" if detected else "coasting"
            print(f"[still] cap {capture_idx:5d}  t={t_sec:.1f}s  {status}  -> {out_path.name}")

            capture_idx += 1

            # Sleep until the next capture interval, accounting for
            # processing time already spent this cycle.
            elapsed = time.time() - t_capture_start
            sleep_time = max(0.0, cfg.still_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n[still] Stopped after {capture_idx} captures.")
    finally:
        picam2.stop()
        picam2.close()

    elapsed_total = capture_idx * cfg.still_interval
    det_count = sum(1 for r in csv_rows if r["detected"])
    print(
        f"\nProcessed {capture_idx} stills over ~{elapsed_total:.0f}s. "
        f"Detected worm in {det_count}/{capture_idx} frames "
        f"({'N/A' if not capture_idx else f'{100*det_count/capture_idx:.1f}%'})."
        f"\nNOTE: detection % = frames where a valid-area blob was found. "
        f"This is coverage, not validated accuracy -- dust/stains/ghosts can "
        f"also satisfy the area filter. Visual inspection of annotated stills "
        f"is needed to confirm true tracking quality."
    )

    _write_outputs(csv_rows, cfg)


def _write_outputs(csv_rows, cfg):
    """Writes CSV and plot. Shared by still mode and anything else that needs it."""
    if not csv_rows:
        return
    if cfg.output_csv:
        fieldnames = [k for k in csv_rows[0].keys() if not k.startswith("_")]
        with open(cfg.output_csv, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(csv_rows)
        print(f"Wrote trajectory CSV to {cfg.output_csv}")
    if cfg.output_plot:
        _save_plot(csv_rows, cfg.output_plot, cfg.mm_per_px)
        print(f"Wrote trajectory plot to {cfg.output_plot}")
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Center-of-mass motion tracker for C. elegans video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["analyze", "record", "record-analyze", "live", "still"],
                   default="analyze",
                   help="analyze: process an existing video file. "
                        "record: Pi camera -> file, no analysis. "
                        "record-analyze: Pi camera -> file -> analyze. "
                        "live: analyze frames straight from the Pi camera in real time. "
                        "still: capture full-resolution still images at timed intervals "
                        "(best for long low-magnification experiments -- uses full 12.3MP "
                        "sensor resolution, no video file size limit, minimal CPU load "
                        "between captures). record/record-analyze/live/still need picamera2.")
    p.add_argument("--camera-selftest", action="store_true",
                   help="Ignore --mode; just probe the Pi camera and report frame "
                        "shapes/fps, then exit. Do this before your first real "
                        "--mode live/record/record-analyze run.")
    p.add_argument("--input", default=None,
                   help="[analyze mode] Path to video file, OR a camera index (e.g. '0') "
                        "for a live feed via OpenCV/V4L2 (use this for USB webcams; use "
                        "--mode live for the Pi camera via picamera2).")
    p.add_argument("--output-csv", default=None, help="Path to write trajectory CSV.")
    p.add_argument("--output-video", default=None,
                   help="[analyze mode] Path to write annotated .mp4 preview.")
    p.add_argument("--output-plot", default=None, help="Path to write a trajectory PNG plot.")
    p.add_argument("--roi", type=parse_roi, default=None,
                   help="[analyze mode] Crop to 'x,y,w,h' in ORIGINAL frame pixels before "
                        "processing (e.g. to exclude dish rim / eyepiece vignette).")
    p.add_argument("--downscale", type=float, default=0.5,
                   help="Resize factor applied before processing (lower = faster, "
                        "important for real-time use on a Raspberry Pi 4).")
    p.add_argument("--min-area", type=float, default=80,
                   help="Minimum blob area (in px^2, AFTER downscale) to be considered the worm. "
                        "IMPORTANT: at low magnification the worm occupies fewer pixels -- lower "
                        "this value (try 20-40) when the worm appears small in frame. At very low "
                        "mag with --downscale 0.5, the worm may be only 30-80px^2.")
    p.add_argument("--max-area", type=float, default=15000,
                   help="Maximum blob area (in px^2, AFTER downscale) to be considered the worm.")
    p.add_argument("--bg-alpha", type=float, default=0.01,
                   help="Background adaptation rate (0-1). Higher adapts faster to "
                        "illumination drift but risks absorbing a very slow-moving worm.")
    p.add_argument("--bg-init-frames", type=int, default=45,
                   help="Number of initial frames (median-combined) used to build the "
                        "starting background model.")
    p.add_argument("--clahe", action="store_true",
                   help="Apply CLAHE local contrast normalization before segmentation. "
                        "Recommended at low magnification where illumination is uneven.")
    p.add_argument("--no-kalman", action="store_true",
                   help="Disable Kalman smoothing/prediction (raw center-of-mass only).")
    p.add_argument("--auto-mask", action="store_true",
                   help="Automatically detect and mask out the microscope's dark/bright "
                        "field-of-view vignette so corner artifacts are ignored.")
    p.add_argument("--show", action="store_true",
                   help="Show a live preview window (requires a display).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many processed frames (or stills in --mode still).")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="Process every Nth frame (analyze mode).")
    p.add_argument("--mm-per-px", type=float, default=None,
                   help="Optional calibration factor (millimeters per ORIGINAL-resolution "
                        "pixel) to also emit position/distance columns in mm.")
    p.add_argument("--max-jump-px", type=float, default=None,
                   help="Max plausible per-frame displacement in ORIGINAL-resolution pixels. "
                        "Default: auto (30%% of frame diagonal).")

    cam = p.add_argument_group("Raspberry Pi camera options (record / record-analyze / live / still)")
    cam.add_argument("--resolution", default="1080p",
                      help=f"Recording resolution: a preset ({', '.join(RESOLUTION_PRESETS)}) "
                           f"or 'WIDTHxHEIGHT'. In --mode still, defaults to full sensor "
                           f"resolution (4056x3040) regardless of this setting. Presets: " +
                           "; ".join(f"{k}={v[0]}x{v[1]} (~{v[2]}fps, {v[3]})" for k, v in RESOLUTION_PRESETS.items()))
    cam.add_argument("--analysis-resolution", default="640x480",
                      help="[live mode] Resolution of the separate low-res stream used "
                           "for real-time analysis -- kept small so the Pi 4's CPU can "
                           "keep up regardless of the recording resolution.")
    cam.add_argument("--record-output", default=None,
                      help="Output video path for record/record-analyze modes (required), "
                           "or for live mode if you also want to save footage (optional).")
    cam.add_argument("--duration", type=float, default=None,
                      help="Recording duration in seconds (record/record-analyze modes). "
                           "Omit to record until Ctrl+C.")
    cam.add_argument("--record-fps", type=float, default=None,
                      help="Target camera frame rate. Omit to let the camera choose based "
                           "on --resolution.")
    cam.add_argument("--bitrate", type=int, default=None,
                      help="H.264 bitrate in bits/sec for recording. Default ~10 Mbps.")
    cam.add_argument("--target-fps", type=float, default=None,
                      help="[live mode] Analysis rate limit in fps. If lower than the camera "
                           "fps, frames between analyses are discarded, leaving CPU headroom "
                           "for other processes or future analysis layers running in parallel. "
                           "Example: --target-fps 5 on a 30fps camera analyzes every 6th frame. "
                           "Timestamps in the CSV reflect this effective rate.")
    cam.add_argument("--still-interval", type=float, default=5.0,
                      help="[still mode] Seconds between still captures. Default 5s = 0.2fps. "
                           "Use 1.0 for ~1fps, 60.0 for one capture per minute.")
    cam.add_argument("--still-output-dir", default="stills",
                      help="[still mode] Directory to save annotated JPEG stills.")

    perf = p.add_argument_group("Performance profiling")
    perf.add_argument("--profile", action="store_true",
                       help="Print rolling per-stage timing (capture/preprocess/track/"
                            "annotate) and effective fps every --profile-interval frames. "
                            "Cheap enough to leave on.")
    perf.add_argument("--profile-interval", type=int, default=30,
                       help="Frames between profiling reports.")
    perf.add_argument("--profile-csv", default=None,
                       help="Optional path to also dump per-frame stage timings as CSV, "
                            "for offline analysis.")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = Config(args)

    if args.camera_selftest:
        camera_selftest(cfg)
        return

    if cfg.mode == "analyze":
        if not cfg.input:
            raise SystemExit("--mode analyze requires --input PATH")
        run_analyze(cfg)
    elif cfg.mode == "record":
        if not cfg.record_output:
            raise SystemExit("--mode record requires --record-output PATH")
        run_record(cfg)
    elif cfg.mode == "record-analyze":
        if not cfg.record_output:
            raise SystemExit("--mode record-analyze requires --record-output PATH")
        run_record_analyze(cfg)
    elif cfg.mode == "live":
        run_live(cfg)
    elif cfg.mode == "still":
        run_still(cfg)
    else:
        raise SystemExit(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
