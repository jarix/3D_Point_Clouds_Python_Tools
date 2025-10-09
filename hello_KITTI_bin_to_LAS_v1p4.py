# Prerequisites
import sys
import argparse
from pathlib import Path
import numpy as np
import laspy

def convert_one_file(bin_path: Path, scale_xyz=(0.001, 0.001, 0.001)):
    '''Convert one KITTI .bin file to LAS 1.4 (point format 6)
    Args:
        bin_path: Path to the input .bin file
        scale_xyz: Tuple of (scale_x, scale_y, scale_z) for LAS file
    '''
    data = np.fromfile(bin_path, dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"{bin_path} size is not divisible by 4 floats; got {data.size} floats.")
    pts = data.reshape(-1, 4)
    x, y, z, intensity = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    n = x.shape[0]

    # --- LAS 1.4, point format 6 (XYZ, intensity, returns, GPS time, etc.) ---
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = scale_xyz
    header.offsets = (float(x.min()), float(y.min()), float(z.min()))

    las = laspy.LasData(header)

    # Coordinates (laspy handles scaling/offset to integer storage)
    las.x = x
    las.y = y
    las.z = z

    # Intensity: map KITTI float (~0..1) to uint16 0..65535
    las.intensity = np.clip(intensity * 65535.0, 0, 65535).astype(np.uint16)

    # Required LAS 1.4 (pf 6) fields: set reasonable defaults
    # Return info: single return for all points
    las.return_number = np.ones(n, dtype=np.uint8)
    las.number_of_returns = np.ones(n, dtype=np.uint8)

    # Scanner channel (0..3). KITTI has one sensor stream here.
    las.scanner_channel = np.zeros(n, dtype=np.uint8)

    # Classification (1 = Unclassified per ASPRS)
    las.classification = np.ones(n, dtype=np.uint8)

    # GPS time is present in pf 6; unknown -> 0.0 is fine
    las.gps_time = np.zeros(n, dtype=np.float64)

    # Optional: point_source_id (0) and user_data (0) left at defaults

    out_path = bin_path.with_suffix(".las")
    las.write(out_path)
    print(f"Wrote {out_path}  (points: {n}, LAS 1.4 pf6, scale={scale_xyz}, offsets={header.offsets})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Convert KITTI .bin files to LAS v1.4 (point format 6)")
        print("Usage: python hello_KITTI_bin_to_LAS_v1p4.py <file_or_folder>")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Convert KITTI .bin to LAS v1.4 (point format 6)")
    parser.add_argument("path", help="Path to a .bin file or a directory containing .bin files")
    parser.add_argument("--scale", nargs=3, type=float, default=[0.001, 0.001, 0.001],
                        help="LAS XYZ scales, e.g. --scale 0.001 0.001 0.001")
    args = parser.parse_args()

    target = Path(args.path)
    scale_xyz = tuple(args.scale)

    if target.is_file() and target.suffix.lower() == ".bin":
        convert_one_file(target, scale_xyz)
    elif target.is_dir():
        bins = sorted(target.glob("*.bin"))
        if not bins:
            print(f"No .bin files found in {target}")
            sys.exit(1)
        for p in bins:
            convert_one_file(p, scale_xyz)
    else:
        print("Provide a KITTI .bin file or a folder containing .bin files.")
        sys.exit(1)

