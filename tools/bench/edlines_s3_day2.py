"""S3 day-2 falsification — anchor extraction + chain walking on scene_0008.

Per ``docs/engineering/edlines_s3_anchor_walk_design_2026-05-04.md §2.2-2.3``:

- **Stage A**: Sobel gradient → anchor = pixel with |∇|² > τ²_anchor and local
  maximum along its gradient direction (Canny-style NMS).
- **Stage B**: from each anchor, walk along the edge tangent (perpendicular
  to gradient) building a chain. Smart-routing: try ahead / ahead-left /
  ahead-right; pick highest-gradient candidate that is an anchor; mark
  consumed.

**Day-2 acceptance**: scene_0008 produces 4 chains matching the 4 tag edges,
each with ≥ 100 anchors. If the chains fragment (>4 chains for a single
edge) or merge (single chain spans multiple edges), the algorithm needs
revision before proceeding to day 3.

Usage::

    PYTHONPATH=. uv run --group bench tools/bench/edlines_s3_day2.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


# Default parameters from the design memo.
TAU_ANCHOR = 16.0  # |∇| threshold (intensity units)
MAX_CHAIN_LEN = 10000


def extract_anchors(
    gray: np.ndarray, bbox: tuple[int, int, int, int], tau_anchor: float = TAU_ANCHOR
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stage A — Sobel gradients + anchor pixel extraction with NMS.

    Returns (xs, ys, gxs, gys, |grad|s) for anchor pixels in the bbox.
    """
    x_min, y_min, x_max, y_max = bbox
    # Pad bbox by 2 px so we can compute gradients + NMS at the boundary.
    pad = 2
    x_lo = max(x_min - pad, 0)
    y_lo = max(y_min - pad, 0)
    x_hi = min(x_max + pad, gray.shape[1] - 1)
    y_hi = min(y_max + pad, gray.shape[0] - 1)
    roi = gray[y_lo : y_hi + 1, x_lo : x_hi + 1].astype(np.float64)

    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    # Quantise gradient direction to 4 octants for NMS:
    #  0 = vertical-ish edge   (|gy| dominant; NMS along ±y)
    #  1 = NE-SW diagonal      (gx,gy same sign, magnitudes within 2× — NMS along (+1,+1))
    #  2 = horizontal-ish edge (|gx| dominant; NMS along ±x)
    #  3 = NW-SE diagonal      (NMS along (+1,-1))
    abs_gx = np.abs(gx)
    abs_gy = np.abs(gy)

    # Anchor mask: |∇| > τ AND local max along gradient direction.
    h, w = grad_mag.shape
    anchor_xs: list[int] = []
    anchor_ys: list[int] = []
    anchor_gx: list[float] = []
    anchor_gy: list[float] = []
    anchor_mag: list[float] = []
    # Restrict scan to the inner bbox (so dx,dy steps stay in roi).
    bx0 = x_min - x_lo
    by0 = y_min - y_lo
    bx1 = x_max - x_lo
    by1 = y_max - y_lo
    for j in range(by0, by1 + 1):
        for i in range(bx0, bx1 + 1):
            m = grad_mag[j, i]
            if m < tau_anchor:
                continue
            # NMS direction perpendicular to the edge → along the gradient.
            # Determine octant.
            ax = abs_gx[j, i]
            ay = abs_gy[j, i]
            if ay > 2 * ax:
                # Vertical edge: gradient is along ±y → compare neighbours along y.
                dx, dy = 0, 1
            elif ax > 2 * ay:
                # Horizontal edge: gradient is along ±x → compare along x.
                dx, dy = 1, 0
            else:
                # Diagonal edge.
                if (gx[j, i] >= 0) == (gy[j, i] >= 0):
                    dx, dy = 1, 1
                else:
                    dx, dy = 1, -1
            if 0 <= j - dy < h and 0 <= j + dy < h and 0 <= i - dx < w and 0 <= i + dx < w:
                if m < grad_mag[j - dy, i - dx] or m < grad_mag[j + dy, i + dx]:
                    continue
            anchor_xs.append(i + x_lo)
            anchor_ys.append(j + y_lo)
            anchor_gx.append(float(gx[j, i]))
            anchor_gy.append(float(gy[j, i]))
            anchor_mag.append(float(m))

    return (
        np.asarray(anchor_xs, dtype=np.int32),
        np.asarray(anchor_ys, dtype=np.int32),
        np.asarray(anchor_gx, dtype=np.float64),
        np.asarray(anchor_gy, dtype=np.float64),
        np.asarray(anchor_mag, dtype=np.float64),
    )


def walk_chains(
    xs: np.ndarray,
    ys: np.ndarray,
    gxs: np.ndarray,
    gys: np.ndarray,
    mags: np.ndarray,
) -> list[list[int]]:
    """Stage B — walk anchors into chains following the edge-tangent direction.

    Returns a list of chains; each chain is a list of indices into the input
    anchor arrays.
    """
    n = len(xs)
    # Spatial index: (x, y) -> anchor index.
    pos_to_idx: dict[tuple[int, int], int] = {(int(xs[i]), int(ys[i])): i for i in range(n)}
    consumed = np.zeros(n, dtype=bool)

    # Sort anchors by gradient magnitude (descending) so we start chains from
    # the strongest anchors (Akinlar-Topal smart routing).
    order = np.argsort(-mags)

    chains: list[list[int]] = []

    def quantised_tangent(idx: int, sign: int) -> tuple[int, int]:
        """8-connected step direction along the edge tangent at anchor `idx`,
        with a sign indicating which way to walk along the tangent."""
        # Tangent = (-gy, gx) / |g|. For 8-connected stepping we want the
        # discrete (dx, dy) ∈ {-1, 0, +1}² that's closest to the tangent.
        tx = -gys[idx] * sign
        ty = gxs[idx] * sign
        # Pick the unit-step direction by sign + dominance.
        ax, ay = abs(tx), abs(ty)
        if ax > 2 * ay:
            dx, dy = (1 if tx > 0 else -1), 0
        elif ay > 2 * ax:
            dx, dy = 0, (1 if ty > 0 else -1)
        else:
            dx = 1 if tx > 0 else -1
            dy = 1 if ty > 0 else -1
        return dx, dy

    def rotate_45(dx: int, dy: int, sign: int) -> tuple[int, int]:
        """Rotate an 8-connected step by ±45°."""
        # 8-connected directions in CCW order:
        #   (1,0) (1,1) (0,1) (-1,1) (-1,0) (-1,-1) (0,-1) (1,-1)
        dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        idx_dir = dirs.index((dx, dy))
        new_idx = (idx_dir + sign) % 8
        return dirs[new_idx]

    for start_in_order in order:
        if consumed[start_in_order]:
            continue
        chain = [int(start_in_order)]
        consumed[start_in_order] = True

        for direction_sign in (+1, -1):
            cursor = int(start_in_order)
            step_dx, step_dy = quantised_tangent(cursor, direction_sign)
            for _ in range(MAX_CHAIN_LEN):
                # Three candidates: ahead, ahead-left, ahead-right (±45°).
                cx, cy = int(xs[cursor]), int(ys[cursor])
                candidates = [
                    (cx + step_dx, cy + step_dy),
                    (cx + rotate_45(step_dx, step_dy, +1)[0], cy + rotate_45(step_dx, step_dy, +1)[1]),
                    (cx + rotate_45(step_dx, step_dy, -1)[0], cy + rotate_45(step_dx, step_dy, -1)[1]),
                ]
                # Pick the candidate whose gradient direction is most similar
                # to the cursor's (within MAX_ANGLE_DELTA), with |∇| as tiebreak.
                # The orientation gate prevents the chain from walking across a
                # corner where the gradient rotates ~90° to a perpendicular edge.
                MAX_ANGLE_COS = np.cos(np.radians(30.0))  # 30° tolerance
                cursor_g_norm = np.hypot(gxs[cursor], gys[cursor])
                best_idx = -1
                best_score = -1.0
                best_step = (0, 0)
                for c in candidates:
                    if c not in pos_to_idx:
                        continue
                    ci = pos_to_idx[c]
                    if consumed[ci]:
                        continue
                    cand_g_norm = np.hypot(gxs[ci], gys[ci])
                    if cand_g_norm < 1e-9 or cursor_g_norm < 1e-9:
                        continue
                    # Cosine similarity of unit gradient vectors. Two
                    # antiparallel gradients (sign flip across the edge centre)
                    # also indicate the same edge — use abs() of the dot product.
                    cos_sim = abs(
                        gxs[ci] * gxs[cursor] + gys[ci] * gys[cursor]
                    ) / (cand_g_norm * cursor_g_norm)
                    if cos_sim < MAX_ANGLE_COS:
                        continue
                    score = mags[ci] * cos_sim
                    if score > best_score:
                        best_score = score
                        best_idx = ci
                        best_step = (c[0] - cx, c[1] - cy)
                if best_idx < 0:
                    break
                # Append to chain (front for direction -1, back for +1).
                if direction_sign > 0:
                    chain.append(best_idx)
                else:
                    chain.insert(0, best_idx)
                consumed[best_idx] = True
                cursor = best_idx
                step_dx, step_dy = best_step

        chains.append(chain)

    return chains


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO_ROOT
        / "tests/data/hub_cache/locus_v1_tag36h11_1920x1080/images/scene_0008_cam_0000.png",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default="881,456,1048,606",
        help="Component bbox 'x_min,y_min,x_max,y_max' (default: scene_0008 tag bbox from S1 telemetry).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU_ANCHOR,
        help=f"Anchor gradient-magnitude threshold (default: {TAU_ANCHOR}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "diagnostics/edlines_s3_day2",
        help="Output directory for visualisation + chain dumps.",
    )
    args = parser.parse_args()

    img_raw = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        raise FileNotFoundError(f"could not read image: {args.image}")
    bbox = tuple(int(v) for v in args.bbox.split(","))
    assert len(bbox) == 4, "--bbox needs 4 ints"

    print(f"image:  {args.image.name}")
    print(f"bbox:   {bbox} ({bbox[2] - bbox[0]} x {bbox[3] - bbox[1]} px)")
    print(f"τ_anchor: {args.tau}")

    # Stage A.
    xs, ys, gxs, gys, mags = extract_anchors(img_raw, bbox, args.tau)
    print(f"\nStage A — anchor extraction:")
    print(f"  total anchors: {len(xs)}")
    print(f"  |∇| range:    {mags.min():.1f} – {mags.max():.1f}")
    print(f"  |∇| mean:      {mags.mean():.1f}, median: {np.median(mags):.1f}")

    # Stage B.
    chains = walk_chains(xs, ys, gxs, gys, mags)
    chain_lens = sorted([len(c) for c in chains], reverse=True)
    print(f"\nStage B — chain walking:")
    print(f"  total chains: {len(chains)}")
    print(f"  chain lengths (top 10): {chain_lens[:10]}")
    print(f"  chains ≥ 100 anchors: {sum(1 for l in chain_lens if l >= 100)}")
    print(f"  chains ≥ 50 anchors:  {sum(1 for l in chain_lens if l >= 50)}")

    # Day-2 acceptance check.
    long_chains = [c for c in chains if len(c) >= 50]
    long_chains.sort(key=len, reverse=True)
    print(f"\nDay-2 acceptance check:")
    print(f"  Top 4 chains by length:")
    for i, ch in enumerate(long_chains[:4]):
        cxs = xs[ch].astype(np.float64)
        cys = ys[ch].astype(np.float64)
        x_var = float(cxs.var())
        y_var = float(cys.var())
        orient = "horizontal" if x_var > y_var else "vertical"
        print(
            f"    chain {i}: {len(ch)} anchors, "
            f"x∈[{int(cxs.min())},{int(cxs.max())}], y∈[{int(cys.min())},{int(cys.max())}], "
            f"{orient}"
        )
    n_ge_100 = sum(1 for l in chain_lens if l >= 100)
    if n_ge_100 == 4:
        print(f"  ✓ exactly 4 chains ≥ 100 anchors — PASS")
    elif n_ge_100 < 4:
        print(f"  ✗ only {n_ge_100} chains ≥ 100 anchors — chain walking too aggressive (fragments)")
    else:
        print(f"  ⚠ {n_ge_100} chains ≥ 100 anchors — top-4 selection still likely OK")

    # Visualisation.
    args.out.mkdir(parents=True, exist_ok=True)
    vis = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
    # Draw bbox.
    cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
    # Plot all anchors (small grey dots).
    for x, y in zip(xs, ys, strict=True):
        cv2.circle(vis, (int(x), int(y)), 0, (160, 160, 160), -1)
    # Plot top-K chains in distinct colours.
    palette = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 165, 255),
        (255, 0, 255),
        (255, 255, 0),
        (200, 200, 100),
        (100, 200, 200),
    ]
    for k, ch in enumerate(long_chains[:8]):
        colour = palette[k % len(palette)]
        for idx in ch:
            cv2.circle(vis, (int(xs[idx]), int(ys[idx])), 1, colour, -1)
    crop = vis[
        max(bbox[1] - 10, 0) : bbox[3] + 10,
        max(bbox[0] - 10, 0) : bbox[2] + 10,
    ]
    out_full = args.out / "scene_0008_chains_full.png"
    out_crop = args.out / "scene_0008_chains_crop.png"
    cv2.imwrite(str(out_full), vis)
    cv2.imwrite(str(out_crop), crop)
    print(f"\nVisualisation written to:")
    print(f"  {out_full.relative_to(REPO_ROOT)}")
    print(f"  {out_crop.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
