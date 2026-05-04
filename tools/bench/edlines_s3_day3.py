"""S3 day-3 — segment split + top-4 selection + sub-pixel line fit + corner.

Extends day-2 (anchor extraction + chain walking) with Stages C-F per
``docs/engineering/edlines_s3_anchor_walk_design_2026-05-04.md §2.4-2.7``:

- **Stage C** — chain → linear segments via iterative split-at-worst-RMS.
- **Stage D** — top-4 segment selection (longest 4 with pairwise-angle gate).
- **Stage E** — sub-pixel TLS line fit per segment (anchor positions adjusted
  via 3-point parabolic vertex of |∇| along the gradient direction).
- **Stage F** — corner extraction by intersecting adjacent CW-ordered lines.

End-to-end day-3 acceptance: 4 corners produced on scene_0008's tag, each
within 1 px of GT (per memo §4.2 day-4 criterion, surfaced early).

Usage::

    PYTHONPATH=. uv run --group bench tools/bench/edlines_s3_day3.py
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


# ---------------------------------------------------------------------------
# Stage C — chain → linear segments (split at worst-RMS)
# ---------------------------------------------------------------------------


def fit_line_tls(pts: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """TLS line through 2D points; returns (nx, ny, d, _cx, _cy, rms)."""
    cx_v = float(pts[:, 0].mean())
    cy_v = float(pts[:, 1].mean())
    cov = np.cov(pts.T, ddof=0)
    _, eigvecs = np.linalg.eigh(cov)
    n = eigvecs[:, 0]  # smallest-eigenvalue eigenvector = line normal
    nx, ny = float(n[0]), float(n[1])
    d = -(nx * cx_v + ny * cy_v)
    rms = float(np.sqrt(np.mean((pts @ n + d) ** 2)))
    return nx, ny, d, cx_v, cy_v, rms


def split_chain_into_segments(
    chain_pts: np.ndarray, max_rms: float = 1.5, min_seg_len: int = 8
) -> list[np.ndarray]:
    """Iteratively split a chain at its worst-fitting point until each segment
    has RMS perpendicular distance ≤ max_rms or is shorter than 2·min_seg_len."""
    if len(chain_pts) < 2 * min_seg_len:
        return [chain_pts] if len(chain_pts) >= min_seg_len else []
    nx, ny, d, _, _, rms = fit_line_tls(chain_pts.astype(np.float64))
    if rms <= max_rms:
        return [chain_pts]
    residuals = np.abs(chain_pts.astype(np.float64) @ np.array([nx, ny]) + d)
    split_idx = int(np.argmax(residuals))
    if split_idx < min_seg_len:
        return split_chain_into_segments(chain_pts[split_idx + 1 :], max_rms, min_seg_len)
    if split_idx > len(chain_pts) - min_seg_len:
        return split_chain_into_segments(chain_pts[:split_idx], max_rms, min_seg_len)
    left = split_chain_into_segments(chain_pts[:split_idx], max_rms, min_seg_len)
    right = split_chain_into_segments(chain_pts[split_idx + 1 :], max_rms, min_seg_len)
    return left + right


# ---------------------------------------------------------------------------
# Stage D — top-4 segment selection
# ---------------------------------------------------------------------------


def select_top4_segments(
    segments: list[np.ndarray], min_len_px: float = 20.0
) -> list[np.ndarray] | None:
    """Pick the 4 longest segments with at least 2 distinct orientations."""
    candidates = []
    for seg in segments:
        if len(seg) < 2:
            continue
        span = float(np.linalg.norm(seg[-1].astype(np.float64) - seg[0].astype(np.float64)))
        if span >= min_len_px:
            candidates.append((seg, span))
    if len(candidates) < 4:
        return None
    candidates.sort(key=lambda t: -t[1])
    top4 = [seg for seg, _ in candidates[:4]]
    angles = []
    for seg in top4:
        nx, ny, *_ = fit_line_tls(seg.astype(np.float64))
        angle = (np.degrees(np.arctan2(-nx, ny)) + 180.0) % 180.0
        angles.append(angle)
    angles_arr = np.array(angles)
    diffs = np.abs(angles_arr[:, None] - angles_arr[None, :])
    diffs = np.minimum(diffs, 180.0 - diffs)
    if diffs.max() < 30.0:
        return None
    return top4


# ---------------------------------------------------------------------------
# Stage E — sub-pixel TLS line fit
# ---------------------------------------------------------------------------


def subpixel_adjust_anchor(
    gray: np.ndarray, x: int, y: int, gx: float, gy: float
) -> tuple[float, float]:
    """Adjust anchor (x, y) to the sub-pixel max of |∇| along gradient direction
    via 3-point parabolic vertex on (x±nx, y±ny) for unit-normal (nx, ny)."""
    g_norm = float(np.hypot(gx, gy))
    if g_norm < 1e-9:
        return float(x), float(y)
    nx_g, ny_g = gx / g_norm, gy / g_norm
    h, w = gray.shape

    def grad_mag(xf: float, yf: float) -> float:
        xi, yi = int(round(xf)), int(round(yf))
        if not (1 <= xi < w - 1 and 1 <= yi < h - 1):
            return 0.0
        gxv = 0.5 * (gray[yi, xi + 1] - gray[yi, xi - 1])
        gyv = 0.5 * (gray[yi + 1, xi] - gray[yi - 1, xi])
        return float(np.hypot(gxv, gyv))

    g_m = grad_mag(x - nx_g, y - ny_g)
    g_0 = grad_mag(float(x), float(y))
    g_p = grad_mag(x + nx_g, y + ny_g)
    denom = g_p + g_m - 2.0 * g_0
    if abs(denom) < 1e-9:
        return float(x), float(y)
    delta = (g_m - g_p) / (2.0 * denom)
    delta = max(-0.5, min(0.5, delta))
    return float(x) + delta * nx_g, float(y) + delta * ny_g


def fit_segment_subpixel(
    gray: np.ndarray,
    seg_xy: np.ndarray,
    seg_chain_idx: np.ndarray,
    gxs: np.ndarray,
    gys: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Stage E — sub-pixel TLS line fit for one segment."""
    sub_pts = np.empty((len(seg_xy), 2), dtype=np.float64)
    for k in range(len(seg_xy)):
        ai = int(seg_chain_idx[k])
        sub_pts[k, 0], sub_pts[k, 1] = subpixel_adjust_anchor(
            gray, int(seg_xy[k, 0]), int(seg_xy[k, 1]), float(gxs[ai]), float(gys[ai])
        )
    return fit_line_tls(sub_pts)


# ---------------------------------------------------------------------------
# Stage F — corner extraction
# ---------------------------------------------------------------------------


def intersect_lines(l1, l2) -> tuple[float, float] | None:
    """Intersect two homogeneous lines (nx, ny, d). Returns (x, y) or None."""
    nx1, ny1, d1, *_ = l1
    nx2, ny2, d2, *_ = l2
    det = nx1 * ny2 - ny1 * nx2
    if abs(det) < 1e-9:
        return None
    x = (ny1 * d2 - ny2 * d1) / det
    y = (nx2 * d1 - nx1 * d2) / det
    return x, y


def order_lines_cw(lines):
    """Sort lines by atan2 of their centroid relative to the centroid-of-centroids.
    Returns a permutation of [0..3]."""
    centres = np.array([(line[3], line[4]) for line in lines])
    mid = centres.mean(axis=0)
    angles = np.arctan2(centres[:, 1] - mid[1], centres[:, 0] - mid[0])
    return list(np.argsort(angles))


def extract_corners(top4_lines):
    """Order 4 lines CW, intersect adjacent pairs, return 4 corners CW."""
    cw_order = order_lines_cw(top4_lines)
    ordered = [top4_lines[i] for i in cw_order]
    corners = []
    for i in range(4):
        c = intersect_lines(ordered[i], ordered[(i + 1) % 4])
        if c is None:
            return None
        corners.append(c)
    return corners


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
        default=REPO_ROOT / "diagnostics/edlines_s3_day3",
        help="Output directory for visualisation + chain dumps.",
    )
    parser.add_argument(
        "--gt-corners",
        type=str,
        default="903.44,456.41,1049.71,455.25,1031.40,607.67,880.73,602.61",
        help=(
            "Ground-truth tag corners as 8 comma-separated floats: "
            "x1,y1,x2,y2,x3,y3,x4,y4 in canonical order (BL=c0, BR=c1, TR=c2, TL=c3). "
            "Default: scene_0008 from root-cause memo §1."
        ),
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

    # ===================================================================
    # Stage C — split chains into linear segments
    # ===================================================================
    img_f = img_raw.astype(np.float64)
    print(f"\nStage C — chain → linear segments (max_rms=1.5 px):")
    all_segments: list[tuple[np.ndarray, np.ndarray]] = []  # (xy, chain_idx_array)
    for chain in chains:
        if len(chain) < 8:
            continue
        chain_xy = np.column_stack([xs[chain], ys[chain]])
        chain_idx = np.array(chain, dtype=np.int32)
        # Track index mapping during recursive split.
        sub_segs = split_chain_into_segments(chain_xy, max_rms=1.5, min_seg_len=8)
        # Re-derive chain_idx for each sub-segment (find positions in original chain).
        cursor = 0
        # split_chain_into_segments returns segments in order, but the recursion
        # introduces gaps at split points. Reconstruct by linear scan.
        all_pts_in_chain = chain_xy
        for seg in sub_segs:
            # find starting index in chain.
            for s in range(cursor, len(all_pts_in_chain) - len(seg) + 1):
                if np.array_equal(all_pts_in_chain[s : s + len(seg)], seg):
                    seg_idx = chain_idx[s : s + len(seg)]
                    all_segments.append((seg, seg_idx))
                    cursor = s + len(seg) + 1  # +1 to skip the split-out point
                    break
    seg_lens = sorted([len(s[0]) for s in all_segments], reverse=True)
    print(f"  total segments: {len(all_segments)}")
    print(f"  segment lengths (top 10): {seg_lens[:10]}")

    # ===================================================================
    # Stage D — top-4 segment selection
    # ===================================================================
    bbox_short = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
    min_len = max(20.0, bbox_short / 4.0)
    seg_xys = [s[0] for s in all_segments]
    top4_xys = select_top4_segments(seg_xys, min_len_px=min_len)
    if top4_xys is None:
        print(f"\nStage D FAILED — no valid 4-segment set with min_len_px={min_len:.1f}")
        return
    # Match top4_xys back to (xy, chain_idx) tuples.
    top4 = []
    for txy in top4_xys:
        for sxy, sci in all_segments:
            if sxy is txy:
                top4.append((sxy, sci))
                break
    print(f"\nStage D — top-4 segments:")
    for i, (sxy, _) in enumerate(top4):
        nx, ny, _, cx, cy, rms = fit_line_tls(sxy.astype(np.float64))
        angle = (np.degrees(np.arctan2(-nx, ny)) + 180.0) % 180.0
        print(
            f"  seg {i}: n={len(sxy):4d}, "
            f"x∈[{int(sxy[:,0].min())},{int(sxy[:,0].max())}], "
            f"y∈[{int(sxy[:,1].min())},{int(sxy[:,1].max())}], "
            f"angle={angle:.1f}°, integer-RMS={rms:.3f} px"
        )

    # ===================================================================
    # Stage E — sub-pixel TLS line fit
    # ===================================================================
    print(f"\nStage E — sub-pixel TLS line fit:")
    fitted_lines = []
    for i, (sxy, sci) in enumerate(top4):
        line = fit_segment_subpixel(img_f, sxy, sci, gxs, gys)
        nx, ny, d, cx, cy, rms = line
        angle = (np.degrees(np.arctan2(-nx, ny)) + 180.0) % 180.0
        print(
            f"  line {i}: nx={nx:+.4f}, ny={ny:+.4f}, d={d:+.3f}, "
            f"angle={angle:.1f}°, sub-pixel RMS={rms:.4f} px"
        )
        fitted_lines.append(line)

    # ===================================================================
    # Stage F — corner extraction
    # ===================================================================
    print(f"\nStage F — corner extraction:")
    corners = extract_corners(fitted_lines)
    if corners is None:
        print(f"  FAILED — adjacent line pair was parallel")
        return
    # Parse GT corners.
    gt_vals = [float(v) for v in args.gt_corners.split(",")]
    assert len(gt_vals) == 8, "--gt-corners needs 8 floats"
    gt_corners = [(gt_vals[2 * i], gt_vals[2 * i + 1]) for i in range(4)]
    # Match S3 corners (CW from line ordering) to GT corners (canonical order).
    # Find the GT permutation that minimises sum-of-Δ.
    from itertools import permutations
    s3 = np.array(corners)
    gt = np.array(gt_corners)
    best_perm = None
    best_total = float("inf")
    for perm in permutations(range(4)):
        s3_p = s3[list(perm)]
        total = float(np.sum(np.linalg.norm(s3_p - gt, axis=1)))
        if total < best_total:
            best_total = total
            best_perm = list(perm)
    s3_aligned = s3[best_perm]
    deltas = np.linalg.norm(s3_aligned - gt, axis=1)
    print(f"\nCorner residuals (S3 vs GT, best permutation alignment):")
    for i, ((sx, sy), (gx_v, gy_v), d) in enumerate(
        zip(s3_aligned, gt_corners, deltas, strict=True)
    ):
        print(f"  c{i}: S3=({sx:.3f}, {sy:.3f}), GT=({gx_v:.3f}, {gy_v:.3f}), ‖Δ‖={d:.3f} px")
    print(f"\n  max ‖Δ‖: {deltas.max():.3f} px")
    print(f"  mean ‖Δ‖: {deltas.mean():.3f} px")

    print()
    print("=" * 70)
    print("S3 day-3 end-to-end acceptance")
    print("=" * 70)
    print(f"  Target:   max corner ‖Δ‖ < 1.0 px")
    print(f"  Observed: max corner ‖Δ‖ = {deltas.max():.3f} px → "
          f"{'PASS ✓' if deltas.max() < 1.0 else 'FAIL'}")
    print(f"  Current detector for scene_0008 corner 1: 3.83 px (S1 baseline)")
    if deltas.max() < 1.0:
        print(f"  → S3 architecture VALIDATED end-to-end on Blender PSF.")
    elif deltas.max() < 2.0:
        print(f"  → Partial fix; document as conditional ship per memo §4.3.")
    else:
        print(f"  → S3 falsified — synthetic PSF floor higher than expected.")


if __name__ == "__main__":
    main()
