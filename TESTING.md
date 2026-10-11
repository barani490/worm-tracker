# Testing protocol

Run these on your own rig before trusting output. Commands assume the repo root.

## Automated (no camera needed)
```bash
python3 tests/test_basics.py
```
Checks the circularity math and that the still-mode CSV writer doesn't crash on internal bookkeeping keys.

## Manual, on the Pi

1. **Camera check.** `python3 celegans_tracker.py --camera-selftest`. Pass: frames come back and an fps is reported.
2. **Live timing.** `python3 celegans_tracker.py --mode live --output-csv t.csv --show`. Stop after exactly 30 s by phone timer. Pass: the last `time_s` in `t.csv` matches your timer (rows start after the ~9–10 s warmup, so the first `time_s` is not 0).
3. **Live vs. offline.**
   ```bash
   python3 celegans_tracker.py --mode live --record-output same.mp4 --output-csv live.csv --show
   python3 celegans_tracker.py --mode analyze --input same.mp4 --output-csv offline.csv
   ```
   Use a single worm. Pass: the last `time_s` of both CSVs agrees within about a second.
4. **Still mode.** `python3 celegans_tracker.py --mode still --still-interval 2.5 --output-csv s.csv`, then Ctrl+C. Pass: images are saved *and* `s.csv` is written without a traceback.
5. **Behavior checks** (any live run): worm leaves the frame; re-enters; rests about 2 min; moves fast; makes an omega turn; remove the worm entirely and confirm the tracker does not lock onto dust.
6. **Circularity sanity.** On footage with a visible omega turn or coil, pull the frames at the circularity peaks and confirm the shape matches the number (crawling ≈ 0.2–0.3, omega turn ≈ 0.5, coil ≥ 0.6).
7. **Speed.** Add `--profile` to any mode to print capture / preprocess / track timings.

## Recording tips for pickup or disturbance experiments

The CSV cannot tell *why* tracking was interrupted: a deliberate pickup and a worm wandering off-screen look identical. Log pickup and placement times by hand (phone timer or notebook) for every trial.
