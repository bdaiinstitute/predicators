"""Hacky code for developing very object-specific image-based samplers."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from predicators.structs import Image

OBJECT_CROPS = {
    # min_x, max_x, min_y, max_y
    "hammer": (160, 450, 160, 350),
    "hex_key": (160, 450, 160, 350),
    "brush": (100, 400, 350, 480),
    "hex_screwdriver": (100, 400, 350, 480),
}

OBJECT_COLOR_BOUNDS = {
    # (min B, min G, min R), (max B, max G, max R)
    "hammer": ((0, 0, 50), (40, 40, 200)),
    "hex_key": ((0, 50, 50), (40, 150, 200)),
    "brush": ((0, 100, 200), (80, 255, 255)),
    "hex_screwdriver": ((0, 0, 50), (40, 40, 200)),
}


def _find_center(img: Image,
                 obj_name: str,
                 outfile: Optional[Path] = None) -> Tuple[int, int]:
    # Crop
    crop_min_x, crop_max_x, crop_min_y, crop_max_y = OBJECT_CROPS[obj_name]
    cropped_img = img[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

    # Uncomment for debugging
    cv2.imshow("Cropped image", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Mask color.
    lo, hi = OBJECT_COLOR_BOUNDS[obj_name]
    lower = np.array(lo)
    upper = np.array(hi)
    mask = cv2.inRange(cropped_img, lower, upper)

    # Apply blur.
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Uncomment for debugging
    cv2.imshow("Masked image", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Connected components with stats.
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=4)

    # Find the largest non background component.
    # Note: range() starts from 1 since 0 is the background label.
    max_label, _ = max(
        ((i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)),
        key=lambda x: x[1])

    # Uncomment for debugging
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    cv2.imshow("Biggest component", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cropped_x, cropped_y = map(int, centroids[max_label])

    x = cropped_x + crop_min_x
    y = cropped_y + crop_min_y

    if outfile is not None:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imwrite(str(outfile), img)

    return (x, y)


def _main() -> None:
    # # Hammer
    # obj_name = "hammer"
    # img_nums = [2, 6, 7, 8, 9, 10]
    # for n in img_nums:
    #     img_file = Path(f"sampler_images/wall/img{n}.png")
    #     img = cv2.imread(str(img_file))
    #     outfile = Path(f"sampler_images/wall/labelled_{obj_name}{n}.png")
    #     _find_center(img, obj_name, outfile)

    # # Hex Key
    # obj_name = "hex_key"
    # img_nums = [2, 6, 7, 8, 9, 10]
    # for n in img_nums:
    #     img_file = Path(f"sampler_images/wall/img{n}.png")
    #     img = cv2.imread(str(img_file))
    #     outfile = Path(f"sampler_images/wall/labelled_{obj_name}{n}.png")
    #     _find_center(img, obj_name, outfile)

    # # Brush
    # obj_name = "brush"
    # img_nums = [1, 3, 4, 5, 12, 13, 14, 15, 16]
    # for n in img_nums:
    #     img_file = Path(f"sampler_images/table/img{n}.png")
    #     img = cv2.imread(str(img_file))
    #     outfile = Path(f"sampler_images/table/labelled_{obj_name}{n}.png")
    #     _find_center(img, obj_name, outfile)

    # Screwdriver
    obj_name = "hex_screwdriver"
    img_nums = [12, 13, 14, 15, 16]
    for n in img_nums:
        img_file = Path(f"sampler_images/table/img{n}.png")
        img = cv2.imread(str(img_file))
        outfile = Path(f"sampler_images/table/labelled_{obj_name}{n}.png")
        _find_center(img, obj_name, outfile)


if __name__ == "__main__":
    _main()
