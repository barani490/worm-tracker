"""
Small automated checks. Run from the repo root:

    python3 tests/test_basics.py

Needs opencv-python and numpy (see requirements-core.txt). No camera needed.
This is NOT a full test suite -- it covers two specific things that broke or
were added during development. The hardware/behavior checks are manual; see
TESTING.md.
"""
import csv
import os
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import celegans_tracker as ct  # noqa: E402


def contour_of(draw):
    img = np.zeros((200, 200), np.uint8)
    draw(img)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts[0]


def test_circularity():
    circle = ct.contour_circularity(contour_of(lambda im: cv2.circle(im, (100, 100), 50, 255, -1)))
    thin = ct.contour_circularity(contour_of(lambda im: cv2.rectangle(im, (30, 95), (170, 105), 255, -1)))
    square = ct.contour_circularity(contour_of(lambda im: cv2.rectangle(im, (70, 70), (130, 130), 255, -1)))
    # A pixelated circle lands a bit under 1.0; an elongated, worm-like shape is low.
    assert 0.85 < circle <= 1.0, circle
    assert thin < 0.3, thin
    assert 0.7 < square < 0.85, square
    assert ct.contour_circularity(None) is None
    print(f"ok  circularity: circle={circle:.3f} thin={thin:.3f} square={square:.3f}")


def test_still_csv_write_ignores_internal_keys():
    # Regression test: rows carry internal _raw_cx/_raw_cy keys that are excluded
    # from the CSV columns. This used to crash with ValueError at the very end of
    # a still-mode run.
    rows = [
        {"frame": 0, "time_s": 0.0, "cx_px": 10.0, "cy_px": 20.0, "detected": 1,
         "_raw_cx": 10.0, "_raw_cy": 20.0},
        {"frame": 1, "time_s": 2.5, "cx_px": 11.0, "cy_px": 21.0, "detected": 1,
         "_raw_cx": 11.0, "_raw_cy": 21.0},
    ]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.csv")
        cfg = SimpleNamespace(output_csv=out, output_plot=None, mm_per_px=None)
        ct._write_outputs(rows, cfg)
        with open(out) as f:
            written = list(csv.DictReader(f))
    assert len(written) == 2
    assert "_raw_cx" not in written[0] and "_raw_cy" not in written[0]
    assert written[1]["time_s"] == "2.5"
    print("ok  still-mode CSV write ignores internal keys")


if __name__ == "__main__":
    test_circularity()
    test_still_csv_write_ignores_internal_keys()
    print("all checks passed")
