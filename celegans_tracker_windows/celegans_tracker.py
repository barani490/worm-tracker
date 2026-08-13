#!/usr/bin/env python3

"""
celegans_tracker.py

Center-of-mass motion tracker for C. elegans video recordings.
Full documentation, validation status, and usage guide: see README.md
Debugging history: see CHANGELOG.md

Quick start: python3 celegans_tracker.py --help
"""


import argparse
import csv
import math
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Background model
# ============================================================
# built this whole section leaning heavily on a bunch of OpenCV
# background-subtraction threads on StackOverflow -- the per-pixel
# outlier rejection, the healing-over-time logic, the noise floor
# estimate, all of it started from patterns I found there.
def build_robust_background(init_frames, local_ksize=31, occ_thresh=10.0, min_clean=3):
    """
    Two-pass, per-pixel outlier-rejecting background estimate. This is
    NOT the same problem collect_spread_background_frames() solves.
    That fix (spreading samples across the whole file) stops a worm
    that's simply stationary from being absorbed. This one handles a
    DIFFERENT, distinct case: a worm whose path, over the course of the
    recording, happens to revisit or dwell on roughly the same patch of
    dish often enough that it's present in a large fraction (sometimes
    a literal majority) of the spread samples at that exact location --
    which a plain per-pixel median cannot reject, since a majority-
    occupied pixel's median IS the worm's value, not the dish's.

    Confirmed by direct inspection on a real recording: reconstructing
    the plain-median background and mapping its deviation from the
    surrounding smooth dish showed a real, coherent, worm-shaped smudge
    (12+ gray levels of contrast, not noise) baked in at a fixed
    location. At one sampled pixel there, the worm was present in 60 of
    80 spread background samples -- 75%, a genuine majority. The
    downstream effect: that fixed location reads as "foreground" all
    on its own, independent of the actual worm's current position, so
    whenever the real worm's own path later happens to carry it back
    across that same patch (which is likely exactly BECAUSE it's a
    patch the worm tends to occupy), its real body arrives on top of a
    contour that was already sitting there -- easy to mistake for the
    tracker "predicting" the worm's path, when it's actually a
    background contamination artifact from the worm's own recurring
    behavior earlier (or later) in the same recording.

    Approach: for each of the sampled frames, compute that FRAME's own
    local median (a large-kernel median blur) as a per-frame "what
    should be here if this is just dish" estimate. A pixel whose actual
    value deviates a lot from its own frame's local smoothness is
    "occupied" (something -- presumably the worm -- covers real dish
    there in that one frame) and excluded from that pixel's background
    estimate. The final per-pixel value is the median of whichever
    samples were NOT flagged occupied, however few remain (even a
    handful of genuinely clean samples of a smooth, low-noise dish
    gives a good estimate) -- falling back to the plain median only if
    a pixel has fewer than min_clean clean samples, i.e. the worm was
    on it in nearly every single sample and there is no information to
    do better with from this recording alone. Any pixels that still
    fall back are inpainted from their (by then mostly-clean)
    surroundings, since an empty dish is locally smooth by nature.
    """
    stacked = np.stack(init_frames, axis=0).astype(np.float32)
    n = stacked.shape[0]
    if n < 8:
        # Not enough samples for per-pixel outlier rejection to be
        # meaningful -- fall back to the plain median (original
        # behavior), which is still robust to true minority presence.
        return np.median(stacked, axis=0)

    local_meds = np.stack(
        [cv2.medianBlur(f.astype(np.uint8), local_ksize).astype(np.float32) for f in init_frames],
        axis=0,
    )
    deviation = np.abs(stacked - local_meds)
    clean = deviation <= occ_thresh
    clean_count = clean.sum(axis=0)

    plain_median = np.median(stacked, axis=0)
    masked = np.where(clean, stacked, np.nan)
    with warnings.catch_warnings():
        # Pixels with zero clean samples produce an all-NaN slice here,
        # which nanmedian warns about even though the result is never
        # used -- have_enough (below) is False for exactly those
        # pixels, and they fall back to plain_median / get inpainted.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clean_median = np.nanmedian(masked, axis=0)

    have_enough = clean_count >= min_clean
    refined = np.where(have_enough, clean_median, plain_median).astype(np.float32)

    needs_inpaint = (~have_enough).astype(np.uint8) * 255
    if np.any(needs_inpaint):
        refined = cv2.inpaint(
            refined.astype(np.uint8), needs_inpaint, 15, cv2.INPAINT_TELEA
        ).astype(np.float32)

    return refined


class BackgroundModel:
    """
    Adaptive background model with foreground protection.

    The background is initialized via build_robust_background(): a
    two-pass, per-pixel outlier-rejecting estimate (see that function's
    docstring), not a plain median. A plain median is only robust to a
    worm present in a MINORITY of samples; it can still be fooled if
    the worm's path revisits/dwells on the same patch often enough to
    be a majority at some specific pixels, which build_robust_background
    additionally guards against.

    Each frame, step() nudges the background towards the current frame
    with a small learning rate `alpha` everywhere NOT currently inside
    `safety_mask` (i.e. the worm's current position plus a generous
    buffer around it). It also fast-heals any pixel that has been
    OUTSIDE that safety zone for several consecutive frames in a row --
    see step()'s docstring for why persistence (not a single frame) is
    required before healing.
    """

    def __init__(self, init_frames, alpha=0.01, heal_streak_threshold=10):
        if len(init_frames) == 0:
            raise ValueError("Need at least one frame to initialize background")
        self.bg = build_robust_background(init_frames)
        self.alpha = alpha
        self.heal_streak_threshold = heal_streak_threshold
        self._unprotected_streak = np.zeros(self.bg.shape, dtype=np.int32)

    def diff(self, gray):
        """Absolute difference between current frame and background (uint8)."""
        return cv2.absdiff(gray, self.bg.astype(np.uint8))

    def step(self, gray, protect_mask=None, no_heal_mask=None, heal_roi_mask=None):
        """
        Call exactly once per frame (whether the worm was detected or
        not).

        protect_mask: uint8 or None. The worm's current, tightly-fitted
        position -- frozen completely (no update at all). None means
        no confirmed detection this frame.

        no_heal_mask: uint8 or None. A WIDER buffer around the current
        position; still gets the normal slow drift, but is never fast-
        healed (gives frame-to-frame segmentation jitter at the worm's
        own boundary a chance to be reclaimed later). None disables
        fast healing entirely for this frame.

        heal_roi_mask: uint8 or None. Fast healing is further
        restricted to fall within this region -- a generous but BOUNDED
        neighborhood around the worm's current position (e.g. its
        recent trail), NOT the whole frame. This matters: an earlier
        version made every pixel in the ENTIRE frame that stayed
        outside no_heal_mask long enough eligible for fast healing, no
        matter how far from the worm it was. Given enough frames, that
        ends up snap-healing almost the whole background -- everywhere
        the worm isn't -- one pixel at a time, replacing the smooth,
        multi-frame MEDIAN background (quiet, low-noise by
        construction) with a patchwork of individual single-frame
        snapshots (which carry that one frame's own sensor/compression
        noise, unsmoothed). Confirmed on a real recording: this
        measurably raised the overall per-frame background noise level
        over the course of the video, which pushed the noise floor and
        Otsu threshold up and made an already-faint worm's marginal
        signal harder to clear -- cutting detection nearly in half.
        Restricting fast healing to a bounded area around the worm
        means only the trail actually behind it is ever touched; the
        rest of the frame stays exactly as smooth as the original
        median/slow-EMA background everywhere it's already working.
        """
        gray_f = gray.astype(np.float32)
        protect = (protect_mask > 0) if protect_mask is not None else np.zeros(self.bg.shape, dtype=bool)
        free = ~protect
        self.bg[free] = (1.0 - self.alpha) * self.bg[free] + self.alpha * gray_f[free]

        if no_heal_mask is None or heal_roi_mask is None:
            return

        no_heal = no_heal_mask > 0
        roi = heal_roi_mask > 0
        heal_eligible = free & ~no_heal & roi
        # Persistence streak is only tracked/reset within the ROI --
        # pixels outside it were never candidates for fast healing in
        # the first place, so there's no need to maintain state there.
        self._unprotected_streak[(protect | no_heal) & roi] = 0
        self._unprotected_streak[heal_eligible] += 1
        heal_now = heal_eligible & (self._unprotected_streak >= self.heal_streak_threshold)
        if np.any(heal_now):
            self.bg[heal_now] = gray_f[heal_now]
            self._unprotected_streak[heal_now] = 0

    def as_uint8(self):
        return self.bg.astype(np.uint8)


# ============================================================
# Kalman filter wrapper for centroid smoothing + prediction
# ============================================================
class CentroidKalman:
    """
    Constant-velocity Kalman filter over (x, y).
    Used to:
      - smooth the raw center-of-mass measurement,
      - predict where to look for the worm in the next frame,
      - "coast" for a few frames if the worm is briefly not detected.
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

    def zero_velocity(self):
        """
        Stop trusting the current velocity estimate -- used when
        coasting through a long miss streak so the constant-velocity
        model doesn't extrapolate indefinitely in a straight line.
        See the call site in SingleWormTracker.process() for why.
        """
        self.kf.statePost[2, 0] = 0.0
        self.kf.statePost[3, 0] = 0.0


# ============================================================
# Segmentation + candidate extraction
# ============================================================
def robust_noise_floor(diff, k=5.0, fallback=6.0):
    """
    Estimate a per-frame noise floor directly from the diff image using
    the median absolute deviation (MAD). MAD is robust to the worm's own
    contribution: a real worm blob is a small minority of the frame's
    pixels, so the bulk statistic (median/MAD) reflects background noise,
    not worm signal, even while the worm is present and moving.

    Returns max(k * MAD-based sigma, fallback), so a very clean/quiet
    recording (near-zero background noise) still gets a sane minimum
    floor instead of collapsing to ~0, which would let single-pixel
    compression artifacts pass as a "detection".
    """
    d = diff.astype(np.float32)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    sigma = 1.4826 * mad  # normal-equivalent std-dev estimate from MAD
    return max(k * sigma, fallback)


def segment_foreground(gray, bg_model, kernel, min_signal_floor=6):
    """
    Returns (diff, mask, otsu_threshold, noise_floor).
    mask is a clean binary uint8 image (0/255).

    Uses HYSTERESIS thresholding (the same idea Canny edge detection
    uses), not a single global cutoff: a pixel is only ever included if
    it is CONNECTED to at least one pixel clearing the higher threshold
    (otsu_threshold, or noise_floor if that's higher -- see below), but
    once a region qualifies that way, all of its connected pixels down
    to a lower floor are included too.

    This matters because a real worm's body routinely has substantial
    internal contrast variation -- a strongly-contrasting mid-body next
    to a thin, faint head or tail -- and Otsu's threshold is a single
    GLOBAL split point computed mostly from a whole frame of near-zero
    background. Confirmed on a real recording: only ~25-30% of the
    worm's true bounding-box area was clearing the plain Otsu cutoff,
    cutting the reported contour down to a small fragment of the actual
    body even though the fragment itself was genuinely on-worm.
    Hysteresis fixes this directly: as long as SOME part of the worm is
    unambiguous, the rest of its actually-connected body (down to a
    much lower floor) comes along with it, without lowering the bar
    for unrelated noise elsewhere in the frame (which never gets a
    strong seed to connect to in the first place).
    """
    diff = bg_model.diff(gray)
    otsu_th, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    noise_floor = robust_noise_floor(diff, fallback=min_signal_floor)
    high_th = max(otsu_th, noise_floor)
    low_th = max(noise_floor, high_th * 0.35)

    _, high_mask = cv2.threshold(diff, high_th, 255, cv2.THRESH_BINARY)
    _, low_mask = cv2.threshold(diff, low_th, 255, cv2.THRESH_BINARY)
    # MORPH_OPEN on the strong seed only: keep speckle rejection where
    # it matters (deciding what counts as a genuine seed) without
    # eroding the faint extremities we're trying to recover.
    high_mask = cv2.morphologyEx(high_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(low_mask)
    seeded_labels = np.unique(labels[high_mask > 0])
    seeded_labels = seeded_labels[seeded_labels != 0]
    if seeded_labels.size > 0:
        mask = np.isin(labels, seeded_labels).astype(np.uint8) * 255
    else:
        mask = high_mask

    # MORPH_CLOSE: bridge any remaining small gaps (e.g. a body segment
    # that dips below even the low threshold for a pixel or two).
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    return diff, mask, otsu_th, noise_floor


# ============================================================
# Candidate selection / fresh-lock gating
# ============================================================
# mostly my own logic, but had a lot of back-and-forth with AI working
# through the fresh-lock gating and contrast-check details below.
def detect_warmup_end(cap_read_fn, preprocess_fn, max_check=90, stable_run=5, rel_thresh=0.015):
    """
    Many cameras (including the Raspberry Pi camera modules) take a
    handful of frames at the very start of a recording to settle
    auto-exposure / auto-white-balance, producing a brief global
    brightness transient that has nothing to do with the worm. If
    that transient is included in the background model it corrupts
    the very first detections and can permanently derail tracking.

    `cap_read_fn` is a zero-arg callable returning (ok, frame_bgr),
    matching cv2.VideoCapture.read's signature -- this makes the
    function source-agnostic (works for a file capture or a live
    camera frame-grab callback alike).

    Returns (buffered_frames, start_index): `buffered_frames` is the
    list of preprocessed grayscale frames already consumed (so
    callers can reuse them instead of re-reading), and `start_index`
    is the first index into that list where brightness has stabilized
    (relative frame-to-frame change below `rel_thresh` for
    `stable_run` consecutive frames). If the signal never looks
    unstable, start index is 0 -- i.e. no frames are wasted.
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


def collect_spread_background_frames(path, preprocess_fn, skip_start=0, n_samples=80):
    """
    Build the background-model sample set from frames spread evenly
    across an ENTIRE pre-recorded file, rather than only the first
    `bg_init_frames` consecutive frames.

    THIS IS THE FIX for the root-cause bug confirmed on real recordings
    from this rig: if the worm is on the plate and stationary (or slow-
    moving) for the whole first ~1.5s of the video, EVERY one of the
    first-N-consecutive-frames init frames contains the worm at
    essentially the same spot. The median background then has the worm
    baked into it (verified visually: a faint worm-shaped smudge is
    present in the resulting background image), so diff at the worm's
    real location reads near-zero for the rest of the video while an
    unrelated static-but-imperfectly-modeled feature elsewhere (e.g. the
    field-of-view vignette edge) wins the "largest diff blob" contest
    instead. This reproduced on every one of the 7 sample recordings
    checked, all of which have the worm present at frame 0.

    Sampling spread across the full duration instead fixes this: the
    median is only fooled if the worm sits at the SAME pixel for more
    than half of the *sampled* frames, which requires it to barely move
    for practically the entire recording -- a much rarer situation than
    "doesn't move for the first second and a half". Confirmed on the
    same recordings: the worm-shaped smudge disappears from the
    resulting background image, and the diff image at the worm's real,
    visually-confirmed location jumps from ~0 to 30-90+ (well above
    background noise).

    This only works for a seekable file with a known frame count, which
    is why it's file-only -- a live camera stream has no "later" frames
    to sample from yet, so run_live still uses the consecutive-frame
    approach and instead prompts the user to place the worm AFTER the
    background is captured (see the mode-9 fix note at the top of this
    file).

    Returns a list of preprocessed grayscale frames (may be shorter than
    n_samples if the file is short or has fewer readable frames).
    """
    cap = open_capture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= skip_start + 1:
        cap.release()
        return []
    n_samples = max(1, min(n_samples, total - skip_start))
    target_idxs = set(np.linspace(skip_start, total - 1, n_samples).astype(int).tolist())
    frames = []
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i in target_idxs:
            ok, frame = cap.retrieve()
            if ok:
                _, gray = preprocess_fn(frame)
                frames.append(gray)
        i += 1
    cap.release()
    return frames


def find_candidates(mask, diff, min_area, max_area, min_mean_diff=0.0):
    """
    Returns list of dicts (contour, area, mean_diff, elongation) for
    connected components whose area falls within [min_area, max_area]
    AND whose own mean diff intensity clears min_mean_diff.

    Why re-check signal strength here even though `mask` was already
    thresholded: MORPH_CLOSE (used to bridge gaps in the worm's body) can
    fuse a genuinely strong patch together with adjacent pixels that were
    only included because they sit in the gap being bridged, not because
    they individually cleared the threshold. A large blob can therefore
    end up with a much weaker true average signal than the pixel that
    triggered its inclusion. Requiring the *whole blob's* mean diff to
    clear the floor (not just its brightest pixel) keeps weak/ghost
    regions from being selected just because they cover a large area.

    `elongation` (long axis / short axis from the minimum-area bounding
    rectangle) is reported so callers can prefer worm-like (thin,
    elongated) shapes over blob-like artifacts when picking among several
    valid candidates -- most useful for the very first lock-on, when
    there's no previous position yet to disambiguate with.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        a = cv2.contourArea(c)
        if not (min_area <= a <= max_area):
            continue
        local_mask = np.zeros(diff.shape, dtype=np.uint8)
        cv2.drawContours(local_mask, [c], -1, 255, thickness=cv2.FILLED)
        pixels = diff[local_mask > 0]
        if pixels.size == 0:
            continue
        mean_diff = float(pixels.mean())
        if mean_diff < min_mean_diff:
            continue
        if len(c) >= 5:
            (_, _), (w, h), _ = cv2.minAreaRect(c)
        else:
            _, _, w, h = cv2.boundingRect(c)
        elongation = max(w, h) / max(1.0, min(w, h))
        out.append({"contour": c, "area": a, "mean_diff": mean_diff, "elongation": elongation})
    return out


def local_spatial_contrast(gray, contour, frame_shape, ring_width=6):
    """
    Mean |inside - immediately surrounding ring| gray-level contrast for
    a candidate blob, computed directly on the RAW frame -- no temporal
    history involved at all, unlike everything else in this module's
    detection path.

    Why this is needed as a SECOND, independent check beyond the
    temporal diff floor: a real worm has a sharp visible edge against
    its immediate surroundings in every single frame. A slow,
    spatially-smooth illumination drift or field-of-view shift can
    produce a large and perfectly sustained *temporal* diff (current
    frame vs. background model) at a fixed location without ever having
    any real spatial edge there -- confirmed on a real recording where
    the field of view visibly shifted partway through a 46s session,
    leaving a ~400-frame span at the start where a smooth brightness
    mismatch (not the worm) was the single strongest temporal-diff
    region in the frame, with near-zero local spatial contrast (measured
    0.0-0.3 gray levels vs. 57+ at the real worm's location in the same
    recording). Rejecting candidates below a spatial-contrast floor
    catches this class of false lock entirely, independent of whatever
    is or isn't wrong with the temporal background model.
    """
    local_mask = np.zeros(frame_shape, dtype=np.uint8)
    cv2.drawContours(local_mask, [contour], -1, 255, thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_width + 1, 2 * ring_width + 1))
    dilated = cv2.dilate(local_mask, kernel)
    ring_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(local_mask))
    inside_px = gray[local_mask > 0]
    ring_px = gray[ring_mask > 0]
    if inside_px.size == 0 or ring_px.size == 0:
        return 0.0
    return abs(float(inside_px.mean()) - float(ring_px.mean()))


def weighted_center_of_mass(diff, contour, frame_shape):
    """
    True intensity-weighted center of mass of a single blob: pixels
    with a larger difference from the background ("more worm-like")
    contribute more "mass" to the centroid, using cv2.moments on the
    diff image restricted to the blob (not just the binary shape).

    For a curled or S-shaped worm, this raw moment-based centroid can
    land in the concave gap of the curve -- geometrically "the middle"
    of the blob's bounding region, but not actually on the worm's body,
    with near-zero real diff signal there. Confirmed on a real
    recording: an 18-frame span where a correctly-selected, strongly
    elongated worm blob (area 1818px, elongation 3.15, strong signal)
    nonetheless produced a centroid sitting in empty space between two
    arms of a curl, reading as background at that exact point. To avoid
    ever reporting a point that isn't actually on the worm, this snaps
    the centroid to the nearest pixel that is truly part of the blob.
    """
    local_mask = np.zeros(frame_shape, dtype=np.uint8)
    cv2.drawContours(local_mask, [contour], -1, 255, thickness=cv2.FILLED)
    weights = diff.astype(np.float32) * (local_mask > 0)
    m = cv2.moments(weights, binaryImage=False)
    if m["m00"] <= 1e-6:
        m_bin = cv2.moments(local_mask, binaryImage=True)
        if m_bin["m00"] <= 1e-6:
            return None
        cx, cy = m_bin["m10"] / m_bin["m00"], m_bin["m01"] / m_bin["m00"]
    else:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]

    h, w = local_mask.shape
    ix = int(np.clip(round(cx), 0, w - 1))
    iy = int(np.clip(round(cy), 0, h - 1))
    if local_mask[iy, ix] == 0:
        ys, xs = np.nonzero(local_mask)
        if len(xs):
            d2 = (xs.astype(np.float32) - cx) ** 2 + (ys.astype(np.float32) - cy) ** 2
            k = int(np.argmin(d2))
            cx, cy = float(xs[k]), float(ys[k])
    return cx, cy


# added this after my prof pointed me at Chen et al. 2021 (see README) --
# had AI help working out the implementation and picking a synthetic test
# (circle vs. elongated rectangle) to check the formula before
# trusting it on real footage.
def contour_circularity(contour, area=None):
    """
    Circularity of a blob's contour: 4*pi*A / P^2, where A is area and
    P is perimeter. A value of 1.0 is a perfect circle; values
    approaching 0 indicate an increasingly elongated shape.

    This is the same metric used by Chen et al. (Mol Neurodegener 2021,
    https://doi.org/10.1186/s13024-021-00497-6) to phenotype C. elegans
    locomotion, with reference values from that paper:
      ~0.2  -- normal sinusoidal crawling
      ~0.5  -- omega turn (a normal directional change, not pathological)
      >=0.6 -- coiling (their threshold for the abnormal "coiler" phenotype)
    Those reference values assumed their own camera/contour setup, so
    treat them as a sanity check to validate against, not a guarantee --
    confirm this tracker's own circularity values land in a similar
    range on known normal-crawl and known omega-turn footage before
    trusting the 0.6 cutoff as-is.

    area is accepted as an optional pre-computed value (callers already
    have it via cv2.contourArea) purely to avoid recomputing it; pass
    None to have this function compute it itself.
    """
    if contour is None:
        return None
    if area is None:
        area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter <= 1e-6:
        return None
    return float(4.0 * np.pi * area / (perimeter ** 2))



    """
    Detects the circular/rectangular illuminated field-of-view of a
    stereomicroscope against dark vignetted corners, by finding the
    single largest connected bright(or dark) region. Returns a uint8
    mask (255 = keep, 0 = ignore) or None if no confident, genuinely
    high-contrast vignette boundary is found.

    `min_contrast` guards against mild, ordinary illumination
    gradients (common, NOT a vignette) being mistaken for a hard
    field-of-view edge: mean brightness inside vs. outside the
    candidate mask must differ by at least this many gray levels,
    otherwise masking is skipped entirely.
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
    Encapsulates the per-frame background-subtraction + Kalman
    centroid-tracking logic. This is deliberately the ONLY place this
    logic is implemented: `--mode analyze` (reading a file) and
    `--mode live` (reading live camera frames) both drive frames
    through this exact same class, so there is exactly one tested
    implementation to trust, regardless of where the frames came from.

    Why single-worm: the original version of this tool tried
    multi-worm tracking and it was found to be unreliable (identity
    swaps when worms cross paths, "ghost" tracks from noise) without
    a lot more work (proper multi-object association, e.g. the
    Hungarian algorithm over a full frame's candidate blobs, track
    lifecycle management, etc). Rather than ship something that
    LOOKS like it's tracking several worms but silently produces
    unreliable per-ID data, this stays single-worm and honest about
    it. If you need multiple simultaneous worms, the cleanest path is
    usually to physically isolate one worm per field of view (a
    common approach for this kind of assay anyway), and run one
    instance of this tool per recording.

    Give it grayscale frames one at a time via `process(gray)`.
    """

    def __init__(self, init_frames_gray, cfg, effective_fps):
        self.cfg = cfg
        self.bg_model = BackgroundModel(
            init_frames_gray, alpha=cfg.bg_alpha, heal_streak_threshold=cfg.heal_streak_frames
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kalman = None if cfg.no_kalman else CentroidKalman()
        self.prev_point = None
        self.consecutive_misses = 0
        self.max_consecutive_misses = max(10, int(round(effective_fps)))
        self._last_protect_mask = None
        self.frame_h, self.frame_w = init_frames_gray[0].shape
        # Grace period (in consecutive missed frames) before the Kalman
        # filter's velocity estimate is zeroed during coasting. Short on
        # purpose: even a few frames of unchecked extrapolation from a
        # noisy velocity estimate is enough to reach the frame edge
        # (confirmed on a real recording), after which the position just
        # sits clamped at a corner for the rest of the miss streak --
        # not dangerous (still correctly flagged detected=False) but not
        # a meaningful "coasting" estimate either. Damping fast keeps
        # coasting positions closer to the worm's actual last known area.
        self.velocity_damp_after_misses = max(2, int(round(effective_fps * 0.1)))

        h0, w0 = init_frames_gray[0].shape
        diag = (h0 ** 2 + w0 ** 2) ** 0.5
        if cfg.max_jump_px is None:
            # 0.30 of frame diagonal per frame (was 0.25 before).
            # Increased to handle omega turn centroid shifts without
            # rejecting the real worm mid-curl.
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
        Run one frame through segmentation + tracking.
        Returns a dict: point (x,y) or None, detected (bool),
        contour (or None), area (float), otsu_threshold (float).
        """
        diff, mask, otsu_th, noise_floor = segment_foreground(gray, self.bg_model, self.kernel)
        if self.auto_mask_img is not None:
            mask = cv2.bitwise_and(mask, mask, mask=self.auto_mask_img)
        # Per-blob signal floor: a candidate's own mean diff must clear
        # this to be considered at all. See find_candidates() docstring
        # for why this check is still needed after mask thresholding.
        min_mean_diff = max(noise_floor, otsu_th * 0.5)
        candidates = find_candidates(
            mask, diff, self.cfg.min_area, self.cfg.max_area, min_mean_diff=min_mean_diff
        )
        # Second, independent validation pass: reject any candidate that
        # has no real spatial edge in the raw frame, even if it cleared
        # the temporal-diff floor above. See local_spatial_contrast()
        # docstring -- this specifically catches smooth illumination/
        # field-of-view drift, which the temporal check alone cannot
        # distinguish from a real (if low-contrast) worm.
        candidates = [
            c for c in candidates
            if local_spatial_contrast(gray, c["contour"], gray.shape) >= self.cfg.min_spatial_contrast
        ]
        # Third validation pass: reject small, round blobs that cleared
        # both floors above but still don't look anything like a worm --
        # C. elegans is long and thin even coiled, so a blob that is
        # BOTH round (low elongation) AND small is much more likely to
        # be dust, debris, or an air bubble. Confirmed necessary: a
        # perfectly round, ~120px blob in live_rec.mp4 sat at the exact
        # same sub-pixel position for 13 straight seconds (387 frames)
        # -- no real worm is that still, not even resting (pharyngeal
        # pumping alone produces small continuous shifts). Large round
        # blobs are still allowed through since a genuinely tightly
        # coiled worm can legitimately look close to round.
        candidates = [
            c for c in candidates
            if c["elongation"] >= self.cfg.min_elongation_small_blob
            or c["area"] >= self.cfg.small_blob_area
        ]

        chosen_contour = None
        raw_point = None

        if candidates:
            if self.prev_point is None:
                # Committing to a brand-new lock -- either the very
                # first detection of a session, or reacquiring after
                # having fully lost the worm (see the miss branch
                # below). There's no prior position to sanity-check
                # against here, which is exactly when it's easiest to
                # grab something that isn't the worm: real lab feedback
                # was that the tracker would readily lock onto dust or
                # debris in this situation, especially right after the
                # worm leaves the frame (the last thing left moving/
                # contrasting is often a speck of dust nearby) and at
                # the very start of a live session.
                #
                # A fresh lock is held to a STRICTER bar than ongoing
                # per-frame detection: higher minimum elongation (a
                # real worm is long and thin even coiled; dust/debris is
                # much more often close to round) and a higher signal
                # floor multiplier. This does not change what counts as
                # a "detection" once already tracking -- only whether a
                # brand-new lock gets committed to at all. If nothing
                # clears this stricter bar, the frame is reported
                # undetected rather than settling for the best of a bad
                # set of candidates -- i.e. coast (see the miss branch)
                # and then genuinely wait, rather than latch onto
                # whatever's nearby just because it's the least-bad
                # option.
                fresh_candidates = [
                    c for c in candidates
                    if c["elongation"] >= self.cfg.fresh_lock_min_elongation
                    and c["mean_diff"] >= min_mean_diff * self.cfg.fresh_lock_signal_multiplier
                ]

                if fresh_candidates:
                    def initial_score(item):
                        return item["mean_diff"] * min(item["elongation"], 8.0)

                    best = max(fresh_candidates, key=initial_score)
                else:
                    best = None
            else:
                ref_point = self.prev_point
                if self.kalman is not None:
                    ref_point = self.kalman.predict()

                def dist2(item):
                    cx, cy = weighted_center_of_mass(diff, item["contour"], gray.shape) or (1e9, 1e9)
                    return (cx - ref_point[0]) ** 2 + (cy - ref_point[1]) ** 2

                best = min(candidates, key=dist2)

            if best is not None:
                chosen_contour = best["contour"]
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

            # Generous buffer around the current position: pixels
            # inside this are never updated/healed even if this exact
            # frame's tight contour (fg_mask) doesn't cover them, so
            # ordinary frame-to-frame segmentation jitter at the worm's
            # own boundary gets a chance to be reclaimed on a later
            # frame instead of being permanently erased. See
            # BackgroundModel.step() for the full reasoning (persistence
            # requirement) and the failure mode this avoids.
            safety_zone = cv2.dilate(fg_mask, self.kernel, iterations=self.cfg.heal_safety_iterations)
            heal_roi = cv2.dilate(fg_mask, self.kernel, iterations=self.cfg.heal_roi_iterations)
            self.bg_model.step(
                gray, protect_mask=fg_mask, no_heal_mask=safety_zone, heal_roi_mask=heal_roi
            )
            self._last_protect_mask = fg_mask
            area_val = cv2.contourArea(chosen_contour)
            circ_val = contour_circularity(chosen_contour, area=area_val)
        else:
            self.consecutive_misses += 1
            if self.kalman is not None and self.kalman.initialized:
                # After a few consecutive misses, stop trusting the
                # extrapolated velocity -- a real worm that's briefly
                # lost doesn't keep accelerating away in a straight
                # line forever, and letting the constant-velocity model
                # keep extrapolating unchecked produces exactly that:
                # confirmed on a real recording where a brief noisy
                # jump left the filter with a small spurious velocity
                # that, uncorrected for ~30 frames, extrapolated clean
                # off the edge of the frame (coordinates in the
                # thousands / negative) before the miss-streak reset
                # below finally cleared it. Zeroing velocity after a
                # short grace period keeps any "coasting" position
                # (still correctly flagged detected=False) physically
                # plausible instead.
                if self.consecutive_misses > self.velocity_damp_after_misses:
                    self.kalman.zero_velocity()
                point = self.kalman.predict()
                point = (
                    float(np.clip(point[0], 0, self.frame_w - 1)),
                    float(np.clip(point[1], 0, self.frame_h - 1)),
                )
            else:
                point = self.prev_point
            # No detection this frame -- nothing is currently
            # considered near the worm, so everything accumulates
            # toward the persistence-gated fast heal (see step()).
            self.bg_model.step(gray, protect_mask=None, no_heal_mask=None)
            self._last_protect_mask = None
            area_val = 0.0
            circ_val = None

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
            "circularity": circ_val,
            "otsu": otsu_th,
        }


# ============================================================
# Lightweight performance profiling
# ============================================================
class PerfStats:
    """
    Cheap rolling per-stage timing, meant to answer "what is the Pi 4
    actually spending time on" without itself becoming a burden:
    a handful of time.perf_counter() calls and a fixed-size deque per
    stage. Reporting (the only part that costs more than a
    microsecond) only happens every `report_every` frames.
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
        # New stages appearing after the header was written just get
        # silently dropped from the CSV rather than crashing -- this
        # is a diagnostic aid, not the scientific output of the tool.
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
    A total, clean 0% detection rate over a sustained stretch (not a
    degraded-but-nonzero rate) is the signature of a PHYSICAL setup
    problem -- worm not actually in frame/focus, empty dish, or a worm
    sitting perfectly still long enough to get absorbed into the
    background model -- not a tracking bug. Saw exactly this during
    Pi testing (0/1976 frames on one --mode live run, right after
    getting 91-100% on the same command a minute earlier).

    This prints ONE warning to stderr the first time the rolling
    detection rate crashes below `rate_floor`, so it's something worth
    walking over and checking WHILE a --mode live run is still going,
    instead of finding out 60 seconds later after Ctrl+C.
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
# These map to the IMX477 sensor's well-known native/binned readout
# modes, which is what makes them efficient on a Pi 4 (the camera
# hardware -- not the CPU -- does the binning/cropping). If you're
# using a different camera module (e.g. Camera Module 3 / IMX708),
# these resolutions still "work" as arbitrary output sizes, but they
# won't necessarily map to a native sensor mode, so check your
# module's documented modes for the best performance. You can always
# skip the presets and pass an explicit "WIDTHxHEIGHT" instead.
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

# Real hardware ceiling on the Pi 4's V4L2/bcm2835-codec hardware h264
# encoder block -- found the hard way (2028x1520 crashed with a
# ProcessLookupError three call-frames deep inside picamera2, at the
# ioctl VIDIOC_STREAMON call, instead of any kind of clear "resolution
# too big" message). This is a real silicon limit, not something fixable
# in this script; the Pi 5's encoder block raises this ceiling.
PI4_ENCODER_MAX_WIDTH = 1920
PI4_ENCODER_MAX_HEIGHT = 1080


def _validate_encoder_resolution(resolution):
    """
    Call this before anything that hands frames to the Pi's hardware
    h264 encoder (recording, in any mode). Anything that only ever does
    capture_array() (analyze on a file, --camera-selftest, live analysis
    without --record-output) never touches the encoder and doesn't need
    this check -- those can legitimately use "high"/"super_high".
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
        self.bg_init_samples = args.bg_init_samples
        self.min_spatial_contrast = args.min_spatial_contrast
        self.min_elongation_small_blob = args.min_elongation_small_blob
        self.small_blob_area = args.small_blob_area
        self.heal_safety_iterations = args.heal_safety_iterations
        self.heal_streak_frames = args.heal_streak_frames
        self.heal_roi_iterations = args.heal_roi_iterations
        self.fresh_lock_min_elongation = args.fresh_lock_min_elongation
        self.fresh_lock_signal_multiplier = args.fresh_lock_signal_multiplier
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

        # Target fps: limits analysis rate in live mode to leave CPU
        # headroom for other processes / future analysis layers.
        # If set, overrides frame_skip in live mode to achieve the
        # target rate from whatever the camera's native fps is.
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

    # Some OpenCV builds -- notably the one that ships on Raspberry Pi
    # OS, which is compiled with a GStreamer backend but no ffmpeg
    # backend -- don't reliably treat a bare filename as "just open this
    # file" the way the ffmpeg backend does. It can instead misparse the
    # string as a GStreamer pipeline description (symptom: a "GStreamer
    # warning: Error opening bin: unexpected reference '<filename>'"
    # printed to stderr, then the same RuntimeError below). If the plain
    # open fails, retry with an explicit, unambiguous decode pipeline
    # naming the file via filesrc instead of letting OpenCV guess.
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
    frames_to_skip_on_reopen = warmup_start
    is_file = isinstance(cfg.input, str) and Path(cfg.input).exists()

    if is_file:
        # Pre-recorded file: sample the background from frames spread
        # across the WHOLE recording rather than just the first
        # bg_init_frames consecutive frames. See
        # collect_spread_background_frames() docstring for why this is
        # the fix for worm-absorbed-into-background lock-on -- confirmed
        # against real recordings from this rig.
        init_frames = collect_spread_background_frames(
            cfg.input, preprocess, skip_start=frames_to_skip_on_reopen,
            n_samples=cfg.bg_init_samples,
        )
        if not init_frames:
            # Fall back to the consecutive-frame buffer already read
            # by detect_warmup_end above (e.g. an unusual file where
            # frame count isn't reliably reported).
            init_frames = buffered[warmup_start : warmup_start + cfg.bg_init_frames]
    else:
        init_frames = buffered[warmup_start : warmup_start + cfg.bg_init_frames]

    if not init_frames:
        raise RuntimeError("Could not read any frames from the input source.")

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
        circ_val = result["circularity"]
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
            "circularity": round(circ_val, 4) if circ_val is not None else "",
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
                # waitKey(30) gives the display event loop time to process
                # keyboard input. waitKey(1) is too short over SSH/X11 --
                # the window never gets focus and 'q' is silently ignored.
                # ESC (keycode 27) is also accepted as a quit key.
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
        # This used to take the whole run down with a traceback -- AFTER
        # the CSV had already been written successfully. The data isn't
        # lost, so don't nuke the process over a plotting dependency.
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

    Retries acquiring the camera a few times if libcamera reports the
    device as busy. This shows up as "Device or resource busy" /
    "Camera __init__ sequence did not complete." on the SECOND of two
    back-to-back runs -- because the previous process's picam2.stop()
    was not (before this fix) followed by picam2.close(), so libcamera
    hadn't necessarily released the device yet. .close() is now called
    everywhere this function's caller tears the camera down, but the
    release still isn't guaranteed instantaneous, so this retry is a
    belt-and-suspenders backstop, not a substitute for the .close() fix.
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
    Picamera2's YUV420 "lores" array comes back as a single 2D buffer
    of shape (height * 3 // 2, width): the first `height` rows are the
    full-resolution Y (luma) plane, and the remaining rows pack the
    subsampled U/V planes. Since our whole pipeline only needs
    grayscale, we take the Y plane directly rather than paying for a
    YUV->BGR->gray round trip.
    """
    w, h = size
    y_plane = lores_array[:h, :w]
    return cv2.GaussianBlur(y_plane, (5, 5), 0)


def camera_selftest(cfg: Config):
    """
    Quick, low-stakes diagnostic: opens the Pi camera, grabs a few
    frames from both the main and lores streams, reports their
    shapes/dtypes and an achieved fps, then exits. Meant to be run
    once on real hardware to catch API/format issues (e.g. if a
    particular picamera2 version returns arrays shaped differently
    than documented) before trusting `--mode live` or `--mode record`
    for an actual experiment.
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
    Records from the Pi camera straight to an mp4 file using the
    Pi's hardware H.264 encoder (cheap on CPU -- this is why we don't
    do software encoding here). Blocks until `duration_s` elapses, or
    until Ctrl+C if `duration_s` is None.
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


# had a lot of AI help building this whole function -- the recording/
# analysis-loop split especially.
def run_live(cfg: Config):
    """
    Live analysis directly from the camera, frame by frame, using the
    exact same SingleWormTracker as `--mode analyze`. If
    `cfg.record_output` is set, a full-quality copy is simultaneously
    written to disk via the hardware encoder (attached only to the
    "main" stream, so it does not compete with the CPU-side analysis,
    which runs on the smaller "lores" stream). If not set, nothing is
    ever written to disk except the CSV/plot of tracked positions --
    useful when you only want the data and not the footage.
    """
    Picamera2, H264Encoder, FfmpegOutput = _import_picamera2()

    main_res = cfg.resolution
    if cfg.record_output:
        _validate_encoder_resolution(cfg.resolution)
    elif not cfg.show:
        # Nothing actually consumes the full-res "main" stream in this
        # configuration (no --record-output, no --show) -- request it
        # at the same size as the analysis stream instead of the full
        # recording resolution, so the ISP isn't spending bandwidth
        # moving pixels nothing will ever read. Pure resource/perf
        # optimization; the tracker only ever looks at "lores" either
        # way, so this doesn't change what gets analyzed.
        main_res = cfg.analysis_resolution

    picam2 = _make_picamera2(main_res, cfg.analysis_resolution, fps=cfg.record_fps)

    encoder = None
    recording_start_time = None
    if cfg.record_output:
        encoder = H264Encoder(bitrate=cfg.bitrate or 10_000_000)
        picam2.start_recording(encoder, FfmpegOutput(cfg.record_output))
        # FIX: the recorded video's own clock starts here, but the CSV's
        # t_start used to be set much later (after the background-warmup
        # read below), so live CSV timestamps were silently offset from
        # the recorded file by however long warmup took (~8-9s observed).
        # That made a recorded live session and its own offline re-analysis
        # disagree on timing even though neither individually had a bug --
        # they were just anchored to two different zero points. Anchoring
        # here instead means live's time_s now lines up with time-into-the-
        # actual-video, same as analyze mode does when reading that file back.
        # found this by actually diffing a live CSV against an offline
        # re-analysis of the same recording and noticing the gap didn't
        # close even on a clean single-worm run -- traced it with AI help.
        recording_start_time = time.time()
        print(f"[live] Recording simultaneously to {cfg.record_output} "
              f"at {cfg.resolution[0]}x{cfg.resolution[1]}.")
    else:
        picam2.start()
        print("[live] Not saving video -- analysis only.")

    print(f"[live] Analyzing at {cfg.analysis_resolution[0]}x{cfg.analysis_resolution[1]} "
          "(the 'lores' stream).")

    # --clahe used to silently do nothing here -- analyze/record-analyze's
    # preprocess() applies it but this frame-grab path never did. Fixed
    # so --clahe behaves the same regardless of which mode you run.
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

    # Warm-up detection, adapted for a live source: read directly
    # (no file rewind possible -- those frames are simply gone once
    # consumed, which is fine for a live stream).
    def _read_as_bgr_tuple():
        ok, gray = read_gray()
        # detect_warmup_end wants (ok, frame_bgr) then calls
        # preprocess_fn on it; we short-circuit preprocess_fn to be
        # an identity pass-through since `gray` is already what we
        # need (already grayscale+blurred).
        return ok, gray

    def _identity_preprocess(gray):
        return gray, gray

    print("[live] Building background model ...")
    buffered, warmup_start = detect_warmup_end(_read_as_bgr_tuple, _identity_preprocess)
    init_frames = buffered[warmup_start : warmup_start + cfg.bg_init_frames]
    if not init_frames:
        raise RuntimeError("Could not read any frames from the camera to build a background model.")

    # There used to be a hard requirement here that the plate be empty
    # during background capture, enforced with a 3-second "PLACE YOUR
    # WORM ON THE PLATE NOW" countdown after init_frames was read. Real
    # lab use found this didn't reliably help -- the tracker would still
    # often lock onto dust/debris instead of the worm regardless of the
    # choreographed timing, so the delay was just friction without the
    # payoff. Removed. build_robust_background() (see BackgroundModel)
    # already tolerates the worm being present during these init_frames
    # far better than a plain median would, and the fresh-lock gating in
    # SingleWormTracker.process() (see its docstring) is what actually
    # keeps a fresh lock-on from grabbing dust -- that's true whether
    # the very first lock happens now or after the worm leaves and
    # re-enters the frame later, so it doesn't depend on this init
    # window being worm-free.
    print("[live] Starting tracking ...")

    effective_fps = cfg.record_fps or 30.0
    tracker = SingleWormTracker(init_frames, cfg, effective_fps)
    perf = PerfStats(report_every=cfg.profile_interval, csv_path=cfg.profile_csv) if cfg.profile else None
    det_monitor = DetectionRateMonitor()

    # Target fps: if set, we only run analysis every Nth frame so the
    # CPU has breathing room for future analysis layers or other processes.
    # Example: camera at 30fps, --target-fps 5 → analyze every 6th frame.
    # This is strictly a CPU-overhead control -- the camera still runs at
    # its native fps (frames are captured and discarded between analyses),
    # and the CSV timestamps reflect the target fps, not the camera fps.
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
    t_start = recording_start_time if recording_start_time is not None else time.time()

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

            # Skip frames to enforce target fps overhead control.
            # We still call read_gray() on every raw frame (to drain the
            # camera buffer and keep it from stalling), but only run
            # the tracker on every live_frame_skip-th frame.
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
            circ_val = result["circularity"]
            det_monitor.update(detected)

            if point is not None and csv_rows and csv_rows[-1]["_raw_cx"] is not None:
                cumulative_distance_px += math.hypot(
                    point[0] - csv_rows[-1]["_raw_cx"], point[1] - csv_rows[-1]["_raw_cy"]
                )

            # Real wall-clock elapsed time, NOT frame_idx / assumed_fps.
            # A live capture loop's real rate can (and does) drift from
            # whatever fps was configured/assumed -- --show display
            # overhead, thermal throttling, USB/CSI hiccups, anything.
            # Dividing by an assumed-constant fps silently accumulates
            # that drift into every timestamp; measuring the actual
            # clock does not, regardless of what the real frame rate
            # turns out to be. Confirmed via real lab testing: CSV
            # timestamps were measurably off from a phone-timer-measured
            # elapsed time in --mode live specifically, while --mode
            # analyze (reading a file, where frame_idx / src_fps IS
            # correct -- see run_analyze) was fine.
            t_sec = time.time() - t_start
            row = {
                "frame": frame_idx,
                "time_s": round(t_sec, 4),
                "cx_px": round(point[0], 2) if point is not None else "",
                "cy_px": round(point[1], 2) if point is not None else "",
                "area_px": round(area_val, 1),
                "circularity": round(circ_val, 4) if circ_val is not None else "",
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
                # Same waitKey(30) + ESC fix as in run_analyze.
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
# also mostly built with AI help, same as run_live above.
def run_still(cfg):
    """
    Capture full-resolution still images from the Pi camera at a fixed
    interval (--still-interval seconds) and run the tracker on each one.

    This is specifically designed for:
      - Long experiments (5+ minutes) where video file size is impractical
      - Maximum spatial resolution (12.3MP HQ camera) for accurate position
        measurement at low magnification where the worm is small in pixels
      - Low temporal resolution is acceptable (e.g. 1 frame every 5s)
        because worm movement between stills is small relative to worm size
      - CPU headroom: between captures the Pi is idle, leaving room for
        other analysis to run

    Output: annotated JPEG per capture saved to --still-output-dir,
    plus the standard CSV and trajectory plot.

    Usage:
        python3 celegans_tracker.py --mode still \\
            --still-interval 5 \\
            --still-output-dir ./stills/ \\
            --output-csv still_track.csv \\
            --output-plot still_traj.png \\
            --downscale 0.5 \\
            --mm-per-px 0.003
    """
    # BUG (found via traceback on hardware, fixed with AI help): this used
    # to be `picamera2 = _import_picamera2(); Picamera2 = picamera2.Picamera2`
    # -- assigning the function's 3-item return value to a variable literally
    # named `picamera2` shadowed the real module, so `.Picamera2` on the
    # next line was asking a tuple for an attribute it doesn't have.
    # `_make_picamera2` below was also being called wrong (passed the
    # Picamera2 CLASS where it expects a resolution tuple) -- two stacked
    # bugs in three lines, both fixed here.
    Picamera2, H264Encoder, FfmpegOutput = _import_picamera2()
    # The HQ camera (IMX477) native full resolution is 4056x3040 (~12.3MP).
    # This is above the Pi 4's h264 encoder limit, but still mode never
    # uses the encoder -- it uses capture_array() which has no ceiling.
    still_res = cfg.resolution or (4056, 3040)

    picam2 = _make_picamera2(still_res)

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
    session_start = time.time()

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
            circ_val = result["circularity"]

            # Real wall-clock elapsed time, not capture_idx * still_interval.
            # The loop below sleeps to pace itself to still_interval, but
            # if any single capture+process cycle ever runs long (slow
            # autofocus, a big blob to segment, anything), sleep_time
            # clamps to 0 and that frame's real capture time silently
            # drifts past what capture_idx * still_interval assumes -- with
            # no mechanism to catch back up on later frames either. Same
            # class of bug as the live-mode fix above; still_interval is
            # usually large enough that this was low-risk here, but
            # there's no reason to leave an assumed-rate timestamp in
            # when the actual clock is right there.
            t_sec = time.time() - session_start

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
                "circularity": round(circ_val, 4) if circ_val is not None else "",
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
    """Shared CSV and plot writing for still mode and any future mode."""
    if not csv_rows:
        return
    if cfg.output_csv:
        # NOTE: csv_rows still carry internal "_raw_cx"/"_raw_cy" bookkeeping
        # keys (used above to compute cumulative_distance_px between frames)
        # that are deliberately excluded from fieldnames but were never
        # popped off the row dicts themselves -- DictWriter's default
        # extrasaction="raise" then blows up on the first real row. Found
        # this the hard way after a full still-mode run completed fine and
        # then crashed writing the CSV at the very end. run_analyze/run_live
        # avoid this by popping those keys per-row earlier; simplest fix
        # here is just telling the writer to ignore anything not listed.
        fieldnames = [k for k in csv_rows[0].keys() if not k.startswith("_")]
        with open(cfg.output_csv, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
                        "starting background model. Used as-is for live camera streams; "
                        "for --mode analyze on a file, --bg-init-samples is used instead "
                        "(see below) since the whole file is available up front.")
    p.add_argument("--bg-init-samples", type=int, default=80,
                   help="--mode analyze on a file only: number of frames, evenly spread "
                        "across the ENTIRE recording, median-combined to build the "
                        "starting background model. Spreading across the whole file "
                        "(rather than only the first bg-init-frames consecutive frames) "
                        "is what prevents a worm that is stationary at the very start "
                        "from being permanently baked into the background as a ghost.")
    p.add_argument("--min-spatial-contrast", type=float, default=12.0,
                   help="Minimum mean gray-level contrast a candidate blob must have "
                        "against its immediate surroundings IN THE RAW FRAME (no "
                        "temporal history) to be accepted. Rejects smooth illumination/ "
                        "field-of-view drift that can otherwise masquerade as a "
                        "sustained, strong temporal-diff 'blob' despite having no real "
                        "edge anywhere -- a real worm always has one.")
    p.add_argument("--min-elongation-small-blob", type=float, default=1.3,
                   help="A candidate blob whose elongation (long/short axis ratio) is "
                        "below this AND whose area is below --small-blob-area is "
                        "rejected as likely dust/debris/an air bubble rather than a "
                        "worm. Large round blobs still pass (a tightly coiled worm can "
                        "legitimately look close to round).")
    p.add_argument("--small-blob-area", type=float, default=200,
                   help="Area (in downscaled px^2) below which a round (low-elongation) "
                        "blob is rejected as likely debris. See --min-elongation-small-blob.")
    p.add_argument("--heal-safety-iterations", type=int, default=10,
                   help="Extra dilation iterations (beyond the current worm contour's own "
                        "protection mask) defining the buffer zone that's never fast-healed. "
                        "Larger = more tolerant of frame-to-frame segmentation jitter at the "
                        "worm's own boundary, but freezes background adaptation over a wider "
                        "area for as long as the worm is nearby.")
    p.add_argument("--heal-streak-frames", type=int, default=10,
                   help="Consecutive DETECTED frames a pixel must stay outside the safety "
                        "zone before it's fast-healed (snapped to the current frame) rather "
                        "than left to the slow default alpha drift. Higher = more tolerant "
                        "of segmentation jitter, but a genuine trail behind a moving worm "
                        "takes longer to clear.")
    p.add_argument("--heal-roi-iterations", type=int, default=25,
                   help="Fast healing is restricted to within this many dilation iterations "
                        "of the worm's current position (a bounded local neighborhood, e.g. "
                        "its recent trail) -- NOT the whole frame. Prevents distant, unrelated "
                        "parts of the frame from ever being snap-healed into single-frame "
                        "noisy values, which would otherwise raise overall background noise "
                        "over a long recording and hurt a faint worm's already-marginal "
                        "signal margin.")
    p.add_argument("--fresh-lock-min-elongation", type=float, default=1.8,
                   help="Minimum elongation (long/short axis ratio) required to commit to a "
                        "BRAND NEW lock -- the very first detection of a session, or "
                        "reacquiring after fully losing the worm. Does not affect ongoing "
                        "frame-to-frame tracking once already locked on. Higher = more "
                        "resistant to locking onto round dust/debris, at the cost of taking "
                        "longer to (re)acquire a worm that's tightly coiled.")
    p.add_argument("--fresh-lock-signal-multiplier", type=float, default=1.3,
                   help="A brand-new lock additionally requires mean diff signal to clear "
                        "this multiple of the normal per-candidate floor. Same rationale as "
                        "--fresh-lock-min-elongation: a fresh lock has no prior position to "
                        "sanity-check against, so it's held to a stricter bar than ongoing "
                        "tracking.")
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
