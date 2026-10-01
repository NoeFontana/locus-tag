"""Controlled corner-bias experiment: linear vs sRGB photometry with analytically exact GT.

Renders single tag36h11 tags with exact area coverage (edge convention = Locus:
pixel ``(i, j)`` spans ``[j, j+1] x [i, i+1]``), blurs in **linear** light, optionally
sRGB-encodes, adds noise, quantizes to 8 bit, then reports the mean *radial* corner
bias (outward +) and corner RMSE for each shipped profile. Because the GT is exact by
construction, any bias is the detector's (or the photometry's), not the dataset's.

Usage::

    PYTHONPATH=. uv run --group bench python tools/bench/photometric_corner_bias.py [--trials 30]

See ``docs/engineering/lessons/rotation-tail-and-edge-refinement.md`` (2026-10-01).
"""

from __future__ import annotations

import argparse

import cv2
import locus
import numpy as np

SS = 6  # supersampling factor per axis for area coverage
W, H = 640, 480


def _canvas() -> np.ndarray:
    tag = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 7, 8, borderBits=1
    )
    canvas = np.full((12, 12), 255, np.uint8)
    canvas[2:10, 2:10] = tag  # 8x8-cell tag with a 2-cell white quiet zone
    return canvas


def render(
    rng: np.random.Generator, canvas: np.ndarray, side: float, blur: float, srgb: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(uint8 image, GT outer corners (4, 2) in Locus convention)``."""
    c = np.array([W / 2 + rng.uniform(-80, 80), H / 2 + rng.uniform(-60, 60)])
    a = np.deg2rad(rng.uniform(0, 90))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * side / 2
    sq[2:, 0] *= 1 - rng.uniform(0, 0.2)  # mild keystone
    gt = sq @ rot.T + c
    hm = cv2.getPerspectiveTransform(
        np.array([[2, 2], [10, 2], [10, 10], [2, 10]], dtype=np.float32),
        gt.astype(np.float32),
    )
    hi = np.linalg.inv(hm)
    x0, y0 = np.floor(gt.min(0) - side * 0.5).astype(int).clip(0)
    x1, y1 = np.ceil(gt.max(0) + side * 0.5).astype(int)
    x1, y1 = min(x1, W), min(y1, H)
    ys, xs = np.mgrid[y0 * SS : y1 * SS, x0 * SS : x1 * SS]
    u, v = (xs + 0.5) / SS, (ys + 0.5) / SS
    den = hi[2, 0] * u + hi[2, 1] * v + hi[2, 2]
    cu = (hi[0, 0] * u + hi[0, 1] * v + hi[0, 2]) / den
    cv_ = (hi[1, 0] * u + hi[1, 1] * v + hi[1, 2]) / den
    ci, cj = np.floor(cv_).astype(int), np.floor(cu).astype(int)
    inside = (ci >= 0) & (ci < 12) & (cj >= 0) & (cj < 12)
    val = np.ones(ci.shape, np.float32)
    val[inside] = canvas[ci[inside], cj[inside]] / 255.0
    lin = np.ones((H, W), np.float32)
    lin[y0:y1, x0:x1] = val.reshape(y1 - y0, SS, x1 - x0, SS).mean((1, 3))
    lin = 0.08 + 0.84 * lin
    if blur > 0:
        lin = cv2.GaussianBlur(lin, (0, 0), blur)
    enc = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) if srgb else lin
    img = np.clip(np.round(enc * 255 + rng.normal(0, 1.0, enc.shape)), 0, 255).astype(np.uint8)
    return img, gt


def main() -> None:
    ap = argparse.ArgumentParser(description="Linear vs sRGB corner-bias experiment.")
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    canvas = _canvas()
    dets = {p: locus.Detector(profile=p) for p in ("standard", "high_accuracy")}
    print("photometry  blur  side | " + " | ".join(f"{p}: radial px / RMSE px / n" for p in dets))
    for srgb in (False, True):
        for blur in (0.5, 1.0, 1.5):
            for side in (40, 120):
                acc: dict[str, tuple[list[float], list[float]]] = {p: ([], []) for p in dets}
                for _ in range(args.trials):
                    img, gt = render(rng, canvas, side, blur, srgb)
                    cen = gt.mean(0)
                    for p, det in dets.items():
                        for c in np.asarray(det.detect(img).corners).reshape(-1, 4, 2):
                            c = min(
                                (np.roll(c, r, 0) for r in range(4)),
                                key=lambda cc: float(np.abs(cc - gt).sum()),
                            )
                            if np.abs(c - gt).max() > 4:
                                continue
                            out = [(gt[k] - cen) / np.linalg.norm(gt[k] - cen) for k in range(4)]
                            acc[p][0].extend(float((c[k] - gt[k]) @ out[k]) for k in range(4))
                            acc[p][1].append(float(np.sqrt(((c - gt) ** 2).sum(1).mean())))
                cells = [
                    f"{np.mean(r):+.3f} / {np.mean(e):.3f} / {len(e)}" if e else "—"
                    for r, e in acc.values()
                ]
                print(
                    f"{'sRGB' if srgb else 'linear':10s} {blur:5.1f} {side:5d} | "
                    + " | ".join(cells)
                )


if __name__ == "__main__":
    main()
