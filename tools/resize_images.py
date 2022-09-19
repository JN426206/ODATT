# Script for resize all images in source path with export to the target path
# Example run:
# python tools/resize_images.py "20161019 BM/images" "HD/20161019 BM/images"

import glob
import os
import sys
from pathlib import Path
import cv2


def convert(source_path, target_path, target_image_width=0, target_image_height=0):
    source_files = [os.path.basename(x) for x in glob.glob(source_path + "/*")]
    num_of_files = len(source_files)
    print("Number of files to convert: {}".format(num_of_files))
    if not os.path.exists(target_path):
        Path(target_path).mkdir(parents=True, exist_ok=True)
    for index, source_file in enumerate(source_files):
        print(f"Converting file {source_file} {index + 1}/{num_of_files}: ", end="")
        source_image = cv2.imread(os.path.join(source_path, source_file))
        # aspect_ratio = source_image.shape[0] / source_image.shape[1]
        # aspect_ratio = source_image.shape[0] if source_image.shape[1] > source_image.shape[0] else source_image.shape[1] / source_image.shape[0] if source_image.shape[0] < source_image.shape[1] else source_image.shape[1]
        if target_image_width == 0 or target_image_height == 0:
            if target_image_width == 0:
                target_image_width = int(source_image.shape[1] * (target_image_height/source_image.shape[0]))
            if target_image_height == 0:
                target_image_height = int(source_image.shape[0] * (target_image_width/source_image.shape[1]))
        print(f"{source_image.shape[1]}x{source_image.shape[0]} -> {target_image_width}x{target_image_height}")
        target_image = cv2.resize(source_image, (target_image_width, target_image_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(target_path, source_file), target_image)

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Podaj jako 1 argument katalog źródłowy a jako 2 katalog przeszukiwany w którym zostaną usunięte pliki!")
        exit(0)

    convert(sys.argv[1], sys.argv[2], target_image_width=720)
