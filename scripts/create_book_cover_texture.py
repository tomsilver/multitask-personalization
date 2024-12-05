"""Script to help create book cover textures that can be loaded into
pybullet."""

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from numpy.typing import NDArray
from PIL import Image


def _main(
    front_cover_file: Path,
    back_cover_file: Path,
    spine_file: Path,
    outfile: Path,
    scale: int = 250,
    page_width: int = 2,
    page_gray_color: float = 80,
    page_white_color: float = 225,
) -> None:

    # Read the front and back cover images
    front_cover = iio.imread(front_cover_file)
    back_cover = iio.imread(back_cover_file)
    spine = iio.imread(spine_file)

    # Normalize the height of the front and back covers to match
    front_cover_img = _resize_image(front_cover, scale, scale)
    back_cover_img = _resize_image(back_cover, scale, scale)
    spine_img = _resize_image(spine, scale, scale)

    # Create pages.
    pages_img = np.zeros_like(front_cover_img)
    for c in range(0, pages_img.shape[1], 2 * page_width):
        pages_img[:, c : c + page_width] = page_gray_color
        pages_img[:, c + page_width : c + 2 * page_width] = page_white_color

    # Create an empty canvas for the texture
    canvas = 255 * np.ones((scale * 4, scale * 4, 3), dtype=np.uint8)

    # Horizontal flip both covers
    front_cover_img = front_cover_img[:, ::-1, :]
    back_cover_img = back_cover_img[:, ::-1, :]
    spine_img = spine_img[:, ::-1, :]

    # Rotate the front cover 90 degrees
    front_cover_img = np.rot90(front_cover_img, k=1)

    # Rotate the spine 180 degrees
    spine_img = np.rot90(spine_img, k=2)

    # Place the images on the texture map
    canvas[scale * 2 : scale * 3, 0:scale, :] = front_cover_img  # Front cover
    canvas[scale * 3 : scale * 4, scale * 2 : scale * 3, :] = (
        back_cover_img  # Back cover
    )
    canvas[scale * 2 : scale * 3, scale : scale * 2, :] = spine_img  # spine
    canvas[scale * 3 : scale * 4, 0:scale, :] = pages_img
    canvas[scale * 3 : scale * 4, scale : scale * 2, :] = pages_img
    canvas[scale * 3 : scale * 4, scale * 3 : scale * 4, :] = pages_img

    # Save the final texture map to the specified file
    texture_img = Image.fromarray(canvas)
    texture_img.save(outfile)
    print(f"Book cover texture saved to {outfile}")


def _resize_image(image: NDArray, height: int, width: int) -> NDArray:
    """Resize an image to match the specified height while keeping aspect
    ratio."""
    img = Image.fromarray(image)
    resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(resized_img)[..., :3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("front_cover", type=Path)
    parser.add_argument("back_cover", type=Path)
    parser.add_argument("spine", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    _main(args.front_cover, args.back_cover, args.spine, args.outfile)
