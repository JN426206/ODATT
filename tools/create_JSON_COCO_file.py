import glob
import json
import os
import sys
from pathlib import Path
import cv2


def get_markers_poss(file_format):
    seperator = " "
    if "\t" in file_format:
        seperator = "\t"
    elif ";" in file_format:
        seperator = ";"
    markers = file_format.split(seperator)
    class_marker_pos = -1
    score_marker_pos = -1
    if "C" in markers:
        class_marker_pos = markers.index("C")
    if "S" in markers:
        score_marker_pos = markers.index("S")
    if "X" in markers:
        X_marker_poss = [i for i in range(len(markers)) if markers[i] == "X"]
    else:
        print("Wrong inputted file format!")
        exit(1)
    if len(X_marker_poss) != 4:
        print("Wrong inputted file format!")
        exit(1)
    return class_marker_pos, score_marker_pos, X_marker_poss, seperator, len(markers)


def convert(source_path, target_file, source_file_format, bbox_mode=1, image_width=None, image_height=None,
            images_source_path=None):
    source_files = [os.path.basename(x) for x in glob.glob(source_path+"/*")]
    image_files = None
    if images_source_path:
        image_files = {".".join(os.path.basename(x).split(".")[:-1]): os.path.basename(x) for x in glob.glob(images_source_path+"/*")}
    elif image_width is None and image_height is None:
        print("Set image width and height!")
        exit(1)
    num_of_files = len(source_files)
    print("Number of files to convert: {}".format(num_of_files))
    s_class_marker_pos, s_score_marker_pos, s_X_marker_poss, s_seperator, _ = get_markers_poss(file_format=source_file_format)
    images_label = {}
    for index, source_file in enumerate(source_files):
        print(f"Reading file {source_file} {index+1}/{num_of_files}: ")
        if image_files:
            image_name = ".".join(source_file.split(".")[:-1])
            image = cv2.imread(os.path.join(images_source_path, image_files[image_name]))
            image_height = image.shape[0]
            image_width = image.shape[1]
        image_label = {
            "image_id": index,
            "file_name": image_files[image_name] if image_files else None,
            "height": image_height,
            "width": image_width,
            "annotations": []
        }
        with open(os.path.join(source_path, source_file)) as file:
            for line in file.readlines():
                values = line.split(s_seperator)
                if s_class_marker_pos != -1:
                    object_class = values[s_class_marker_pos]
                    if CHANGE_OBJECT_CLASSID_TO_CLASSLABEL and object_class == "player":
                        object_class = 0
                    if CHANGE_OBJECT_CLASSID_TO_CLASSLABEL and object_class == "ball":
                        object_class = 1
                if s_score_marker_pos != -1:
                    score = values[s_score_marker_pos]
                x1 = int(values[s_X_marker_poss[0]])
                x2 = int(values[s_X_marker_poss[1]])
                x3 = int(values[s_X_marker_poss[2]])
                x4 = int(values[s_X_marker_poss[3]])
                bbox = [x1, x2, x3, x4] if bbox_mode == 1 else [x1, x2, x1+x3, x2+x4]
                annotation = {
                    "bbox": bbox,
                    "bbox_mode": bbox_mode,
                    "category_id": object_class
                }
                image_label["annotations"].append(annotation)
        images_label[os.path.basename(source_file)] = image_label
    with open(os.path.join(target_file), "w") as output_file:
        json.dump(images_label, output_file)
        print("Done!")

CHANGE_OBJECT_CLASSID_TO_CLASSLABEL = True

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Podaj jako 1 argument katalog źródłowy a jako 2 plik JSON gdzie zostaną zapisane dane")
        exit(0)

    if len(sys.argv) == 4:
        convert(sys.argv[1], sys.argv[2], "C X X X X", images_source_path=sys.argv[3])
    else:
        convert(sys.argv[1], sys.argv[2], "C X X X X")
