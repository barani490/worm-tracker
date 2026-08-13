# Changelog

Debugging history for `celegans_tracker.py`, kept verbatim from development.

ERRORS FOUND DURING PI 4 TEST DAY + FIXES (keeping this log, don't delete):
 1) python3 celegans_tracker.py --mode record --record-output test10.mp4 --duration 10
    -> ProcessLookupError inside picamera2's V4L2 h264 encoder (ioctl VIDIOC_STREAMON).
    root cause: default --resolution is "high" = 2028x1520, which is ABOVE the pi 4's
    hardware h264 encoder ceiling (1920x1080, this is a real silicon limit not a bug in
    my code, pi 5 raises it). fixed by adding _validate_encoder_resolution() so this now
    fails IMMEDIATELY with a clear message instead of a cryptic 3-frames-deep ioctl crash,
    and by adding a new "1080p" preset (now the default) so a fresh run works out of the box.
 2) "Device or resource busy" / "Camera __init__ sequence did not complete." on back to
    back runs -- picam2.stop() was never followed by picam2.close(), so libcamera could
    still be holding the device when the next invocation tried to acquire it. fixed:
    .close() everywhere + a retry-with-backoff in _make_picamera2 since even with .close()
    the release isn't always instant.
 3) --mode analyze refusing to open a Pi-recorded mp4 (GStreamer "unexpected reference"
    warning). root cause: the Pi's OpenCV build is GStreamer-only (no ffmpeg backend) and
    mis-parses a bare filename. fixed: open_capture() now falls back to an explicit
    GStreamer decode pipeline if the plain open fails.
 4) --clahe silently doing nothing in --mode live (found while auditing, not from the log)
    -- run_live's frame grab never applied CLAHE unlike analyze/record-analyze. fixed.
 5) matplotlib missing on the pi crashed the whole run right after the CSV was already
    written. fixed: _save_plot now fails soft with an actionable message, doesn't take
    the rest of the run down with it.
 6) 'q' key non-functional in --show mode (confirmed on hardware, both modes).
    root cause: cv2.waitKey(1) is too short -- 1ms is not enough for the OS event loop
    to deliver a keyboard event over SSH/X11 forwarding, and the OpenCV window needs
    focus anyway. fixed: changed to waitKey(30) and also accept ESC (keycode 27) as a
    quit key in both run_analyze and run_live.
 7) Ghost / old-spot lock-on: tracker gets stuck at previously occupied worm position.
    root cause (confirmed): two mechanisms.
    (a) Background healing lag -- after worm moves, the old position heals at alpha=0.01/frame
        which takes ~100 frames (~3.3s at 30fps). During this time the stale patch reads as
        foreground and can be mistaken for the worm.
    (b) Vicious cycle -- if tracker locks onto the ghost, protect_mask keeps the ghost patch
        frozen in the background indefinitely, so it NEVER heals.
    fixed:
    (a) BackgroundModel.mark_as_stale() + accelerated healing (5x alpha) for 20 frames after
        worm moves significantly. Called from SingleWormTracker.process() on confirmed movement.
    (b) stationary_frames counter in SingleWormTracker: if the tracked point has not moved
        beyond 12% of max_jump for max_stationary_frames (1.5s by default), force a
        background update WITHOUT protection -- absorbs the ghost/static artifact.
 8) Head/tail extremities missing from contour. root cause: thin/faint extremities fall
    below Otsu threshold; morphological close (5x5, 2 iterations) didn't bridge the gap.
    fixed: separate larger close kernel (7x7, 3 iterations) specifically for closing gaps,
    while keeping the open kernel small (5x5, 1 iteration) for speckle removal. Confirmed
    improvement on hardware: full worm body covered at high magnification.
 9) --mode live tracker locks onto worm's STARTING position (worm absorption into background
    model during init). root cause: if the worm is on the plate during the bg_init_frames
    window, the median can include the worm (if present >50% of frames), so the tracker
    starts with the worm's original position already "baked in" as background. The diff at
    that position is then near zero, and ghost patches from early worm movement become the
    strongest foreground candidates.
    STATUS AS OF THIS COMMENT BEING WRITTEN (superseded, see #12): a 3-second delay was
    described here as implemented ("PLACE YOUR WORM ON THE PLATE NOW"). It was NOT actually
    present in run_live()'s code -- the comment was aspirational, not a description of what
    shipped. This is now actually implemented (see #12).
10) max_jump too tight for omega turns. root cause: during an omega turn the worm's centroid
    shifts significantly as the body curls, which the jump-rejection interpreted as an
    implausible teleport. fixed: increased default max_jump from 0.25 to 0.30 * frame_diagonal.
    This gives more room for genuine curl behavior without meaningfully increasing vulnerability
    to ghost lock-on (the ghost is at a fixed old position, not moving).
11) Timestamp warnings in record-analyze ("Timestamps are unset in a packet", "Non-monotonic DTS").
    root cause: FfmpegOutput in picamera2 does not always set PTS/DTS correctly, especially
    on first few frames. This is a known upstream picamera2/ffmpeg integration issue, not
    fixable within this script without rewriting the entire output pipeline.
    STATUS: confirmed in hardware testing. Playback check PENDING -- if recorded files play
    back smoothly and seeking works, these are cosmetic warnings only. If playback is broken,
    escalate as a real bug requiring output pipeline change.
12) [Verified against all 7 real sample recordings from this rig, not just claimed] The
    "WHAT'S TESTED" section below previously claimed the analysis pipeline was verified
    correct at "97.6% of frames tracked, zero mistaken jumps." That claim was false --
    actually running it against the recordings on hand, then independently checking the
    tracked (x,y) against the raw video by hand (pixel intensity at the reported point vs.
    its surroundings, and a second, differently-implemented detector sharing no code with
    the tracker) showed it was tracking real worm tissue in 0 of the 7 recordings. It had
    locked onto a static background artifact in every one and was reporting high confidence
    the entire time. Root causes, all confirmed by direct inspection of the background model
    and diff images (not just inferred):
    (a) Worm-absorbed-into-background (see #9's root cause, but this hits --mode analyze on
        a pre-recorded file too, and there the "place the worm after" workaround is impossible
        since the recording already happened with the worm present at frame 0). Visually
        confirmed: a faint worm-shaped smudge is baked into the median background image when
        built from only the first bg_init_frames consecutive frames.
        FIX: collect_spread_background_frames() -- for --mode analyze on a file, the
        background is now built from frames spread across the ENTIRE recording (default 80,
        --bg-init-samples) instead of only the first --bg-init-frames consecutive frames. The
        median is only fooled if the worm sits at the same pixel for >50% of the whole
        recording, which is far less likely than ">50% of the first ~1.5s". Confirmed this
        removes the baked-in smudge and restores real diff signal at the worm's true location.
    (b) No minimum signal-strength check on candidate blobs -- Otsu always finds *a* split
        point even in a frame with no real worm signal, so slow illumination drift or a
        stale/ghost background patch could pass as a "detection" purely because it was the
        single strongest thing in the frame, without ever being checked against how strong
        real worm signal actually looks.
        FIX: robust_noise_floor() (per-frame MAD-based) plus a per-blob mean-diff floor in
        find_candidates() that requires the WHOLE blob's average diff (not just its brightest
        pixel, which morphological closing can inflate) to clear the floor.
    (c) A slow, spatially-smooth illumination/field-of-view shift can produce a large,
        perfectly sustained TEMPORAL diff at a fixed location with almost no real spatial
        edge -- confirmed on live_rec.mp4, where the field of view visibly shifted partway
        through a 46s session, leaving ~13s at the start where this (not the worm) was the
        strongest diff region in the frame.
        FIX: local_spatial_contrast() -- an independent, non-temporal check (raw-frame
        contrast against immediate surroundings) that a real worm always passes and a smooth
        drift artifact never does.
    (d) Small round blobs (dust/debris/air bubbles) can clear both floors above while still
        looking nothing like a worm -- confirmed on live_rec.mp4: a ~120px, perfectly round,
        motionless-to-the-sub-pixel blob held "detected" for 13 straight seconds (no real
        worm, not even resting, is that still).
        FIX: shape-plausibility gate in SingleWormTracker.process() -- rejects blobs that are
        BOTH round (low elongation) AND small; large round blobs still pass since a tightly
        coiled worm can legitimately look close to round.
    (e) Once locked onto a genuinely bad position early on, the original per-frame nearest-
        candidate selection had nothing to compare against except that bad position, so the
        error was self-reinforcing rather than self-correcting. The first (unanchored) lock-on
        now scores candidates by signal strength x elongation instead of raw area, which
        matters most exactly when there's no prior position yet to sanity-check against.
    (f) Kalman coasting during a miss streak had no bound: a small spurious velocity, once
        established, extrapolated in a straight line for the whole miss streak (confirmed:
        one recording reached coordinates in the thousands / negative before the streak
        finally reset). Still correctly flagged detected=False throughout, but not a
        meaningful position. FIX: coasting predictions are now clamped to the frame, and the
        velocity estimate is zeroed after a couple of missed frames instead of compounding.
    (g) The intensity-weighted centroid of a curled/S-shaped worm blob can land in the
        concave gap of the curve -- geometrically central, but not actually on the worm's
        body, with near-zero real signal there. Confirmed on nice.mp4: an 18-frame span where
        a correctly-selected, strongly elongated (3.15) worm blob still produced a centroid
        reading as background. FIX: weighted_center_of_mass() now snaps to the nearest pixel
        that is actually part of the selected blob.
    All 7 sample recordings were re-checked after these fixes using the same independent,
    non-shared-code verification method: in every recording, sampled DETECTED frames land on
    real worm tissue 85-100% of the time (most at 100%; see the tool's own inline comments
    and the accompanying summary for exact per-video numbers and the two recordings with
    legitimate, honestly-flagged gaps -- a worm that's plausibly out of frame or resting
    through a large fraction of the whole recording is a real limitation of any single
    global background model, not something patched over here).
13) [Found from watching the actual annotated output, not just checking the tracked point]
    #12 verified the reported (x,y) centroid was landing on real worm tissue, which it was --
    but watching the annotated video overlays surfaced problems in the SEGMENTATION/contour
    quality that a point-only check doesn't catch: the green outline trailing the worm's old
    position, fusing with its current position; the outline shrinking to a small fragment of
    the body over the course of a video even as the reported point kept moving correctly with
    the worm; and the outline never covering the "full body" at all in some recordings.
    (a) TRAIL FUSION: a continuously-tracked, smoothly-moving worm never triggered the
        existing ghost-healing code at all. It was gated on (i) a full miss (no detection that
        frame) and (ii) a single-frame jump over 8% of max_jump -- neither condition is met by
        ordinary continuous crawling, which is detected every frame and moves gradually. Net
        effect: the old position kept healing at the ~100-frame default alpha rate long after
        the worm had moved on, so a visible trailing smear persisted behind its real path,
        which morphological closing then fused with the current-frame contour -- confirmed as
        the cause of "past position and body" being merged into one blob, skewing the
        centroid toward stale pixels.
        FIX: BackgroundModel.step() replaces the old jump-threshold-gated mark_as_stale() with
        precise, persistence-gated healing every frame: any pixel outside a generous buffer
        around the worm's current position, for several consecutive DETECTED frames in a row,
        gets snapped straight to the current frame. Two safeguards were necessary after the
        first attempt at this measurably hurt a faint/low-contrast recording:
          - the buffer (not just the worm's tight contour) is exempt from fast healing, so
            ordinary frame-to-frame segmentation jitter at the worm's own boundary (Otsu
            picking up slightly more or less of a thin tail from one frame to the next) isn't
            mistaken for "moved away" and permanently erased -- this alone dropped one
            recording's detection from 85% to 9% before the buffer was added;
          - fast healing is further restricted to a BOUNDED local neighborhood around the
            worm (its recent trail), not the whole frame -- without this, every distant,
            unrelated part of the frame eventually gets individually snap-healed to a single
            noisy frame's raw pixel values (as opposed to the smooth, multi-frame median/EMA
            background), measurably raising overall background noise over a long recording
            and pushing the noise floor up enough to hurt an already-faint worm's margin.
        Misses are handled separately and conservatively: since we can't tell "worm genuinely
        gone from here" from "worm still here but undetected this frame", a miss only ever
        gets the gentle slow drift, never fast healing -- an earlier attempt at fast-healing
        during misses too re-baked an undetected-but-present worm into the background after a
        single ~10-frame miss streak, reproducing the exact ghost-lock bug from #12(a).
    (b) UNDER-SEGMENTATION ("shrinking morph" / never a "full body" outline): a real worm's
        body routinely has substantial internal contrast variation -- a strong mid-body next
        to a faint head or tail -- but segmentation used a single GLOBAL Otsu threshold.
        Confirmed on a real recording: only ~25-30% of the worm's true bounding-box area was
        clearing that cutoff, cutting the reported contour down to a small fragment of the
        actual body (the reported CENTROID was still landing correctly on that fragment, which
        is genuinely part of the worm -- this was a segmentation-completeness problem, not a
        wrong-location one).
        FIX: segment_foreground() now uses hysteresis thresholding (the same idea Canny edge
        detection uses): a pixel is only included if it's connected to at least one pixel
        clearing the high (Otsu/noise-floor) threshold, but once a region qualifies that way,
        all of ITS connected pixels down to a much lower floor come with it. This recovers
        faint extremities that are genuinely part of an already-confirmed worm without
        lowering the bar for unrelated noise elsewhere in the frame, which never gets a strong
        seed to connect to. Median reported area increased 1.5-3.5x across all 7 recordings
        after this change, with on-worm accuracy unchanged (still 85-100%, most at 100%) --
        confirming this recovered real body, not new false positives.
    One recording (live_rec.mp4) has a remaining ambiguous case worth flagging explicitly
    rather than glossing over: hysteresis thresholding newly picks up a small, elongated,
    nearly motionless blob near frame 0 that persists with the same shape for 13+ seconds.
    It passes every validation check here (real spatial contrast, not round enough to be
    filtered as debris), but "worm resting in one spot for 13 seconds" and "a stationary,
    elongated static feature like a scratch or fiber" are not distinguishable by any check in
    this file, and this one hasn't been visually confirmed either way. Treat detections in
    that specific window with appropriate skepticism until someone looks at the actual video.
14) [Reported as "the tracker perfectly predicts where the worm is about to go" -- it wasn't
    prediction, it was a second, distinct background-contamination bug] A user watching
    live_test.mp4 noticed the green contour, near the start of the video, included an extra
    S-shaped region beyond the visible worm -- which the worm's real body then appeared to
    "grow into" over the following seconds, looking exactly like the software was forecasting
    its path. Two wrong explanations were offered first (a translucent/faint tail, and the
    worm somehow resting/reversing) before actually reconstructing the exact background image
    build_robust_background produces for this file and inspecting it directly: it contained a
    real, coherent, worm-shaped smudge (12+ gray levels of contrast against a dish that is
    otherwise flat to within ~1 gray level) baked in at a fixed location, independent of
    wherever the worm actually was in the current frame.
    Root cause, confirmed by checking actual per-pixel sample values across the 80 spread
    background frames: collect_spread_background_frames (#12(a)) stops a worm that's simply
    STATIONARY from being absorbed, but does nothing about a worm whose path, over the course
    of the recording, happens to revisit or dwell on roughly the same patch of dish often
    enough that it's present in a large fraction -- sometimes a literal majority -- of the
    spread samples at that specific location. A plain per-pixel median cannot reject that: if
    a pixel is worm-covered in the majority of samples, the worm's value IS the median. At one
    checked pixel, the worm was present in 60 of 80 samples (75%). The downstream effect is
    exactly what was reported: that location reads as "foreground" on its own, independent of
    the real worm's current position, and when the worm's own path later carries it back
    across that same patch (likely BECAUSE it's a patch the worm tends to occupy), its real
    body arrives on top of a contour that was already sitting there.
    FIX: build_robust_background() replaces the plain per-pixel median with a two-pass,
    per-pixel outlier-rejecting estimate. For each of the sampled frames, its own local median
    (large-kernel median blur) serves as a per-frame "what should be here if this is just
    dish" estimate; any pixel whose actual value deviates a lot from that is "occupied" in
    that one frame and excluded. The final value is the median of whichever samples were NOT
    flagged occupied, however few remain -- even a handful of genuinely clean samples of a
    smooth, low-noise dish gives a good estimate, so the fallback to the (contaminated) plain
    median only applies with fewer than 3 clean samples out of the whole set, and any pixel
    that still falls back is inpainted from its (by then mostly-clean) surroundings, since an
    empty dish is locally smooth by construction.
    Verified directly against the same recording: the previously-ghosted region dropped from
    903 contaminated pixels (>12 gray levels off) to 3 out of 7500 checked, and the reported
    area, which had swung from 1318 down to 95 (a 14x range) over the first ~280 frames as the
    ghost slowly bled out via ordinary background adaptation, is now stable within a tight
    1.0-1.15x band across the ENTIRE video. Re-ran all 7 recordings after this fix: 6 of 7 now
    detect the worm in 100% of frames (up from 85-100%) with on-worm accuracy unchanged (still
    100% on sampled detected frames, confirming these are real detections, not new false
    positives from a lower effective threshold); the 7th (live_rec.mp4) improved from 36.7% to
    65.6%, consistent with the genuine, separately-diagnosed gaps described in #12/#13 rather
    than a new problem.
15) [From real lab use of --mode live over multiple sessions, not just the 7 sample files]
    Three reports, none of which required touching segmentation/detection at all -- all three
    are about WHEN the tracker commits to a brand-new lock, not what counts as a detection
    once already tracking:
    (a) "when worm goes out of cam it tracks specs or dust, it should coast around there for a
        bit then just coast away" and "only concern is literally in the beginning it usually
        doesn't lock on worm but other stuff, shaking cam usually fixes this". Both are the
        same underlying gap: the moment nothing is currently being tracked (self.prev_point is
        None -- true at the very start of a session, and true again after fully losing the
        worm), the tracker committed to whatever candidate scored best among ALL of them, with
        no extra scrutiny. Dust/debris can clear the normal per-frame floors (they're real
        physical particles with genuine contrast) while still not looking like a worm at all,
        and once locked onto, ongoing nearest-to-prediction tracking would happily keep
        following that same dust speck indefinitely, since a settled dust speck is just as
        "trackable" frame-to-frame as a real worm is.
        FIX: a fresh lock (first detection of a session, or reacquiring after the existing
        coast-then-reset miss-streak logic gives up) is now held to a stricter bar than
        ongoing per-frame detection: minimum elongation 1.8 (a real worm is long and thin even
        coiled; dust/debris tends to be much closer to round) and mean diff signal at least
        1.3x the normal floor. If nothing clears this stricter bar, the frame is reported
        undetected -- coasts near the last known position via the existing Kalman logic for a
        bit (max_consecutive_misses, ~1s), then genuinely stops and waits, rather than
        settling for the best of a bad set of candidates just because something cleared the
        (lower) bar that's appropriate for continuing an already-confirmed track. This changes
        NOTHING about candidate generation, segmentation, or ongoing-track selection -- same
        mask, same floors, same hysteresis, same nearest-to-prediction logic once locked on.
        Verified against all 7 sample recordings: identical results (6/7 still 100% detection,
        same on-worm accuracy) -- this gate only bites when nothing is currently locked on,
        which essentially never happens mid-recording in the 7 sample files, so this is
        additive protection for the live/edge-of-frame case the samples don't exercise much,
        not a change to what those recordings measure.
    (b) "tracker usually tracks dusts even after picking up worm and putting worm during time
        allocated - should remove the 'adding worm part' as it'll be hard": the 3-second
        "PLACE YOUR WORM ON THE PLATE NOW" countdown added in #12 for --mode live wasn't
        reliably solving the problem it was meant to in actual lab use, and was just added
        friction. Removed. run_live() now starts tracking immediately after building the
        background from init_frames, whatever is or isn't on the plate at that moment --
        relying on build_robust_background() (tolerant of the worm being present) and the
        fresh-lock gating in (a) above (which doesn't care whether this is the start of a
        session or a later reacquisition) instead of a choreographed timing window.
    (c) Long unattended runs, including stretches where the worm barely moves for extended
        periods, were reported as running cleanly with no ghosting and no trail leaking into
        the background -- a real-world field confirmation of the healing fix in #13(a) holding
        up over much longer, less controlled sessions than the 7 sample recordings cover.
16) [Found via real lab testing: ran --mode live for a phone-timer-measured 30s and 120s and
    compared against the CSV's own last timestamp] --mode live's CSV time_s column was computed
    as frame_idx / effective_analysis_fps -- i.e. it ASSUMED the capture loop sustained exactly
    the configured/nominal fps (30fps by default) every single frame, rather than measuring
    actual elapsed time. A live capture loop's real rate can and does drift from whatever's
    configured (--show display overhead, thermal throttling, camera/USB/CSI hiccups, anything),
    and that drift accumulates silently into every single timestamp. Confirmed reproducible
    across both the 30s and 120s tests, and present with or without --record-output (the video
    file's own timestamps were fine -- only the CSV's were wrong, since only the CSV used this
    assumed-fps math). --mode analyze does NOT have this bug: frame_idx / src_fps is CORRECT
    there, because a recorded file's frame rate is a fixed, known property of the file itself,
    not something that has to be measured live.
    FIX: --mode live's time_s is now time.time() - t_start (actual measured wall-clock elapsed
    time), not an assumed-rate calculation. Applied the same fix to --mode still's timestamps
    (capture_idx * still_interval before this) for the same underlying reason, even though it
    wasn't reported broken -- still_interval is normally large enough relative to per-capture
    processing time that the drift is low-risk there, but it's the identical class of bug
    (assumed pacing vs. actual clock) and there's no reason to leave it once the live-mode
    instance of it was found.
    Separately reported same day: --mode analyze (and record-analyze, which calls it) got
    noticeably slower after the #14 background-construction fix -- one real test measured 161s
    to analyze a 30s recording. Not reproduced on the machine this was developed on (analyze
    consistently runs FASTER than real-time there across all 7 sample recordings), so this
    looks like a real, hardware-dependent cost rather than a logic bug: collect_spread_back
    ground_frames (#14) reads through the file via cap.grab() from wherever it starts all the
    way to the last sample point to build the background BEFORE the main tracking pass starts,
    which -- because grab() still has to decode sequentially through compressed video even
    when not retrieving most frames -- means analyzing a file now costs roughly two full
    decode-length passes instead of one, on top of the added per-frame hysteresis/
    connectedComponents cost from #13(b). Both costs scale with video length and would hit a
    Raspberry Pi's much weaker decoder far harder than a dev machine, which would also explain
    why a shorter re-test "was quicker" (correctly noticed, but the video being shorter is WHY
    it was quicker, not evidence the underlying cost is fixed). NOT fixed here -- changing the
    background-construction pass without being able to measure the result on the actual
    hardware risks trading a correctness bug for a performance one, or vice versa. Needs a
    --profile run captured directly on the affected hardware to confirm which of the two added
    costs actually dominates before touching either.
IMPORTANT: --camera-selftest PASS DOESNT GUARANTEE --mode record WILL WORK (selftest never
touches the hardware encoder at all, so it can't catch bug #1 above)


