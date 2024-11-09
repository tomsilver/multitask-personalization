"""Horizontally combine two videos of the same shape and length."""

import argparse
from pathlib import Path

from moviepy.editor import VideoFileClip, clips_array


def _main(infile1: Path, infile2: Path, outfile: Path):
    video1 = VideoFileClip(str(infile1))
    video2 = VideoFileClip(str(infile2))
    final_video = clips_array([[video1, video2]])
    final_video.write_videofile(str(outfile), codec="libx264")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile1", type=Path)
    parser.add_argument("infile2", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.infile1, args.infile2, args.outfile)
