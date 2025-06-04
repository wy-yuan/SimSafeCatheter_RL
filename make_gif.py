"""
make_gif.py
-----------
Create an animated GIF from a folder of PNG (or JPG) frames.

Usage
-----
python make_gif.py --frames ./frames --pattern "*.png" --fps 30 --out path.gif
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio  # pip install imageio

# -------------------------------------------------------------
def build_gif(frame_dir: Path, pattern: str, fps: int, outfile: Path):
    """Read frames in lexicographic order and write an animated GIF."""
    files = sorted(frame_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No images matching {pattern} in {frame_dir}")
    print(f"Found {len(files)} frames, writing {outfile} …")

    # duration = seconds per frame
    with imageio.get_writer(outfile, mode="I", duration=1 / fps) as writer:
        for f in files:
            writer.append_data(imageio.imread(f))
    print("GIF complete.")


# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make GIF from image sequence")
    parser.add_argument("--frames", type=str, default="frames",
                        help="Folder containing images")
    parser.add_argument("--pattern", type=str, default="*.png",
                        help='Glob pattern (e.g. "*.png")')
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second in the GIF")
    parser.add_argument("--out", type=str, default="animation.gif",
                        help="Output GIF filename")

    args = parser.parse_args()
    build_gif(Path(args.frames), args.pattern, args.fps, Path(args.out))
