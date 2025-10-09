# Prerequisites
import sys
import numpy as np
import laspy
from pathlib import Path

def convert_one_file(bin_path: Path, las_scale=(0.001, 0.001, 0.001)):
    """
    Convert one KITTI .bin file to LAS 1.2 point format 0 (XYZ + intensity).
    
    Args:
        bin_path: Path to input .bin file.
        las_scale: Tuple of (x, y, z) scale factors for LAS file.
    """
    data = np.fromfile(bin_path, dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"{bin_path} size is not divisible by 4*float32; got {data.size} floats.")
    pts = data.reshape(-1, 4)
    x, y, z, intensity = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    # Prepare LAS 1.2, point format 0 (XYZ + intensity)
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = las_scale
    # Offsets help keep integers small; here choose mins (or use means)
    header.offsets = (float(x.min()), float(y.min()), float(z.min()))

    las = laspy.LasData(header)

    # Assign coordinates (laspy applies scales/offsets automatically)
    las.x = x
    las.y = y
    las.z = z

    # Map KITTI intensity float (typically 0..~1) to uint16
    inten_u16 = np.clip(intensity * 65535.0, 0, 65535).astype(np.uint16)
    las.intensity = inten_u16

    out_path = bin_path.with_suffix(".las")
    las.write(out_path)
    print(f"Wrote {out_path}  (points: {len(las)})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Convert KITTI .bin files to LAS v1.2 (point format 0)")
        print("Usage: python hello_KITTI_bin_to_LAS_v1p2.py <file_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_file() and target.suffix.lower() == ".bin":
        convert_one_file(target)
    elif target.is_dir():
        for p in sorted(target.glob("*.bin")):
            convert_one_file(p)
    else:
        print("Provide a KITTI .bin file or a folder containing .bin files.")

    
