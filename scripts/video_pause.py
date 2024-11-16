"""Create a new version of a video that pauses when the environment says so."""

import argparse
from pathlib import Path

import pandas as pd
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import freeze  # pylint: disable=no-name-in-module


def _main(infile: Path, results_file: Path, outfile: Path, pause_duration: float):
    df = pd.read_csv(results_file)
    video = VideoFileClip(str(infile))
    dt = 1 / video.fps
    clips = []
    for step in range(1, max(df.step) + 1):
        row = df[df.step == step - 1]
        clip = video.subclip(step * dt, (step + 1) * dt)
        if row.env_video_should_pause.item():
            clip = freeze(clip, freeze_duration=pause_duration)
        clips.append(clip)

    final_video = concatenate_videoclips(clips)
    final_video.write_videofile(str(outfile), codec="libx264")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path)
    parser.add_argument("results_file", type=Path)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--pause_duration", default=2.0, type=float, required=False)
    args = parser.parse_args()
    _main(args.infile, args.results_file, args.outfile, args.pause_duration)
