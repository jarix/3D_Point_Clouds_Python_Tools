#!/usr/bin/env python3
"""
Generate a small single-object point cloud (a 'duck') and save as .xyzrgb.

Each line in the output file has:
    x y z r g b
- x, y, z: float (meters)
- r, g, b: integers in [0, 255]

Usage examples:
    python gen_duck_xyzrgb.py
    python gen_duck_xyzrgb.py --out duck.xyzrgb --mode rainbow
    python gen_duck_xyzrgb.py --points-body 3000 --points-head 1200 --jitter 0.002 --mode patches
"""

import argparse
import numpy as np
from typing import Tuple

def sample_ellipsoid(center: Tuple[float, float, float],
                     radii: Tuple[float, float, float],
                     n_points: int) -> np.ndarray:
    """
    Sample approximately uniform points on an ellipsoid surface using spherical params.
    Returns points as (N, 3).
    """
    u = np.random.rand(n_points) * 2.0 * np.pi
    v = np.random.rand(n_points) * np.pi
    x = radii[0] * np.cos(u) * np.sin(v)
    y = radii[1] * np.sin(u) * np.sin(v)
    z = radii[2] * np.cos(v)
    pts = np.stack([x, y, z], axis=1) + np.array(center, dtype=float)
    return pts

def classic_colors(n_body, n_head, n_beak, n_eye) -> np.ndarray:
    """Yellow body/head, orange beak, black eyes."""
    body_rgb = np.tile([255, 215, 0],   (n_body, 1))
    head_rgb = np.tile([255, 215, 0],   (n_head, 1))
    beak_rgb = np.tile([255, 140, 0],   (n_beak, 1))
    eye_rgb  = np.tile([10, 10, 10],    (n_eye, 1))
    return np.vstack([body_rgb, head_rgb, beak_rgb, eye_rgb, eye_rgb])

def rainbow_colors(points: np.ndarray) -> np.ndarray:
    """Rainbow gradient along X-axis (head↔tail)."""
    x = points[:, 0]
    x_norm = (x - x.min()) / max(1e-9, (x.max() - x.min()))
    # Simple HSV→RGB-ish mapping without external libs:
    # We'll map hue=x_norm, s=1, v=1, then convert to RGB.
    h = x_norm * 6.0
    c = np.ones_like(h)
    xk = 1 - np.abs((h % 2) - 1)  # triangle wave
    zeros = np.zeros_like(h)

    seg = (h.astype(int) % 6)
    r = np.select([seg==0, seg==1, seg==2, seg==3, seg==4, seg==5],
                  [c,       xk,     0,       0,       xk,     c     ], default=0)
    g = np.select([seg==0, seg==1, seg==2, seg==3, seg==4, seg==5],
                  [xk,     c,      c,       xk,     0,       0     ], default=0)
    b = np.select([seg==0, seg==1, seg==2, seg==3, seg==4, seg==5],
                  [0,      0,      xk,      c,      c,       xk    ], default=0)

    rgb = np.stack([r, g, b], axis=1)
    rgb = (rgb * 255.0).clip(0, 255).astype(int)
    return rgb

def patch_colors(points: np.ndarray,
                 idx_slices: dict) -> np.ndarray:
    """
    Region-based colors:
      - body: yellow
      - head: greenish
      - beak: orange
      - eyes: black
      - add a light belly stripe (y near 0, z below 0) on body region
    """
    n_total = points.shape[0]
    rgb = np.full((n_total, 3), [255, 215, 0], dtype=int)  # default body yellow

    sl_body = idx_slices['body']
    sl_head = idx_slices['head']
    sl_beak = idx_slices['beak']
    sl_eye1 = idx_slices['eye1']
    sl_eye2 = idx_slices['eye2']

    rgb[sl_head] = [80, 200, 120]      # mint green head
    rgb[sl_beak] = [255, 140, 0]       # orange beak
    rgb[sl_eye1] = [10, 10, 10]        # black eye
    rgb[sl_eye2] = [10, 10, 10]        # black eye

    # Make a lighter belly stripe on the body (z below 0 and |y| small)
    body_pts = points[sl_body]
    belly_mask = (body_pts[:, 2] < 0) & (np.abs(body_pts[:, 1]) < 0.08)
    body_indices = np.arange(n_total)[sl_body][belly_mask]
    rgb[body_indices] = [255, 240, 150]

    return rgb

def main():
    ap = argparse.ArgumentParser(description="Generate a colored duck point cloud and save as .xyzrgb")
    ap.add_argument("--out", type=str, default="duck_single_object.xyzrgb",
                    help="Output .xyzrgb filename")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--points-body", type=int, default=3000, help="Points on body surface")
    ap.add_argument("--points-head", type=int, default=1200, help="Points on head surface")
    ap.add_argument("--points-beak", type=int, default=600, help="Points on beak surface")
    ap.add_argument("--points-eye", type=int, default=150, help="Points per eye surface")
    ap.add_argument("--jitter", type=float, default=0.002, help="Gaussian jitter std (meters)")
    ap.add_argument("--mode", type=str, default="classic",
                    choices=["classic", "rainbow", "patches"],
                    help="Coloring mode")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # --- Geometry (five ellipsoids) ---
    body_pts = sample_ellipsoid(center=(0.0, 0.0, 0.0),    radii=(0.50, 0.35, 0.30), n_points=args.points_body)
    head_pts = sample_ellipsoid(center=(0.45, 0.0, 0.15),  radii=(0.18, 0.15, 0.15), n_points=args.points_head)
    beak_pts = sample_ellipsoid(center=(0.62, 0.0, 0.12),  radii=(0.08, 0.05, 0.04), n_points=args.points_beak)
    eye1_pts = sample_ellipsoid(center=(0.50, 0.06, 0.20), radii=(0.015, 0.015, 0.015), n_points=args.points_eye)
    eye2_pts = sample_ellipsoid(center=(0.50, -0.06, 0.20),radii=(0.015, 0.015, 0.015), n_points=args.points_eye)

    points = np.vstack([body_pts, head_pts, beak_pts, eye1_pts, eye2_pts])

    # Optional small jitter for realism
    if args.jitter > 0:
        points += np.random.normal(scale=args.jitter, size=points.shape)

    # --- Coloring ---
    n_body, n_head, n_beak, n_eye = args.points_body, args.points_head, args.points_beak, args.points_eye
    idx_slices = {
        "body":  slice(0, n_body),
        "head":  slice(n_body, n_body + n_head),
        "beak":  slice(n_body + n_head, n_body + n_head + n_beak),
        "eye1":  slice(n_body + n_head + n_beak, n_body + n_head + n_beak + n_eye),
        "eye2":  slice(n_body + n_head + n_beak + n_eye, n_body + n_head + n_beak + 2*n_eye),
    }

    if args.mode == "classic":
        rgb = classic_colors(n_body, n_head, n_beak, n_eye)
    elif args.mode == "rainbow":
        rgb = rainbow_colors(points)
    else:  # patches
        rgb = patch_colors(points, idx_slices)

    # --- Save as .xyzrgb (ASCII) ---
    out = np.hstack([points, rgb.astype(int)])
    with open(args.out, "w") as f:
        for x, y, z, r, g, b in out:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    mins = points.min(axis=0); maxs = points.max(axis=0)
    print(f"Saved: {args.out}")
    print(f"Points: {points.shape[0]}")
    print(f"BBox x[{mins[0]:.3f}, {maxs[0]:.3f}] y[{mins[1]:.3f}, {maxs[1]:.3f}] z[{mins[2]:.3f}, {maxs[2]:.3f}]")
    print(f"Color mode: {args.mode}")

if __name__ == "__main__":
    main()