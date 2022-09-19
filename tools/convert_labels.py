import glob
import os
import sys
from pathlib import Path

class BoundingBox:
    def __init__(self):
        self.x1 = 0  # Always absolute min x (top left corner)
        self.y1 = 0  # Always absolute min y (top left corner)
        self.x2 = 0  # Always absolute max x (bottom right corner)
        self.y2 = 0  # Always absolute max y (bottom right corner)
        self.image_width = 0
        self.image_height = 0
        self.format = None

    def set_from_YOLO(self, xcn, ycn, wn, hn, image_width, image_height):  # Relative format <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
        """
        Args:
            xcn (float): Normalized center x of bounding box
            ycn (float): Normalized center y of bounding box
            wn (float): Normalized width of bounding box
            hn (float): Normalized height of bounding box
            image_width (float): Source image width
            image_height (float): Source image height
        """
        self.format = "YOLO"
        self.image_width = image_width
        self.image_height = image_height
        self.x1 = (xcn * image_width) - (wn * image_width)/2
        self.y1 = (ycn * image_height) - (hn * image_height)/2
        self.x2 = (xcn * image_width) + (wn * image_width)/2
        self.y2 = (ycn * image_height) + (hn * image_height)/2

    def set_from_x1_y1_x2_y2(self, x1, y1, x2, y2, image_width, image_height):
        """
        Args:
            x1 (): absolute min x (top left corner)
            y1 (): absolute min y (top left corner)
            x2 (): absolute max x (bottom right corner)
            y2 (): absolute max y (bottom right corner)
            image_width (): Source image width
            image_height (): Source image height
        """
        self.format = "x1y1x2y2"
        self.image_width = image_width
        self.image_height = image_height
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def set_from_x_y_w_h(self, x, y, w, h, image_width, image_height):
        """
        Args:
            x (): absolute min x (top left corner)
            y (): absolute min y (top left corner)
            w (): absolute bounding box width
            h (): absolute bounding box height
            image_width (): Source image width
            image_height (): Source image height
        """
        self.format = "xywh"
        self.image_width = image_width
        self.image_height = image_height
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y + h

    def set_from_cx_cy_w_h(self, cx, cy, w, h, image_width, image_height):
        """
        Args:
            cx (): absolute center x (center point of bounding box)
            cy (): absolute center y (center point of bounding box)
            w (): absolute bounding box width
            h (): absolute bounding box height
            image_width (): Source image width
            image_height (): Source image height
        """
        self.format = "cxcywh"
        self.image_width = image_width
        self.image_height = image_height
        self.x1 = cx - w/2
        self.y1 = cy - h/2
        self.x2 = cx + w/2
        self.y2 = cy + h/2

    def get_YOLO(self, target_image_width=0, target_image_height=0):
        """

        Args:
            target_image_width (): If you want to scale bbox you can set scaled image width
            target_image_height (): If you want to scale bbox you can set scaled image height

        """
        if len(self.format) == 0:
            message = "First set bounding box by some set function!"
            print(message)
            return message

        ratiow = 1
        ratioh = 1
        if target_image_width and target_image_height:
            ratiow = target_image_width / self.image_width
            ratioh = target_image_height / self.image_height

        xn = (self.x1*ratiow + (self.x2*ratiow - self.x1*ratiow)/2) / self.image_width*ratiow
        yn = (self.y1*ratioh + (self.y2*ratioh - self.y1*ratioh)/2) / self.image_height*ratioh
        wn = (self.x2*ratiow - self.x1*ratiow) / self.image_width*ratiow
        hn = (self.y2*ratioh - self.y1*ratioh) / self.image_height*ratioh

        return xn, yn, wn, hn

    def get_x1_y1_x2_y2(self, target_image_width=0, target_image_height=0):
        """

        Args:
            target_image_width (): If you want to scale bbox you can set scaled image width
            target_image_height (): If you want to scale bbox you can set scaled image height

        """
        if len(self.format) == 0:
            message = "First set bounding box by some set function!"
            print(message)
            return message

        ratiow = 1
        ratioh = 1
        if target_image_width and target_image_height:
            ratiow = target_image_width / self.image_width
            ratioh = target_image_height / self.image_height

        return int(self.x1 * ratiow), int(self.y1 * ratioh), int(self.x2 * ratiow), int(self.y2 * ratioh)

    def get_cx_cy_w_h(self, target_image_width=0, target_image_height=0):
        """

        Args:
            target_image_width (): If you want to scale bbox you can set scaled image width
            target_image_height (): If you want to scale bbox you can set scaled image height

        """
        if len(self.format) == 0:
            message = "First set bounding box by some set function!"
            print(message)
            return message

        ratiow = 1
        ratioh = 1
        if target_image_width and target_image_height:
            ratiow = target_image_width / self.image_width
            ratioh = target_image_height / self.image_height

        return int((self.x1-(self.x2-self.x1)/2)*ratiow), int((self.y1-(self.y2-self.y1)/2)*ratioh), int((self.x2-self.x1)*ratiow), int((self.y2-self.y1)*ratioh)

    def get_x_y_w_h(self, target_image_width=0, target_image_height=0):
        """

        Args:
            target_image_width (): If you want to scale bbox you can set scaled image width
            target_image_height (): If you want to scale bbox you can set scaled image height

        """
        if len(self.format) == 0:
            message = "First set bounding box by some set function!"
            print(message)
            return message

        ratiow = 1
        ratioh = 1
        if target_image_width and target_image_height:
            ratiow = target_image_width / self.image_width
            ratioh = target_image_height / self.image_height

        return int(self.x1*ratiow), int(self.y1*ratioh), int((self.x2-self.x1)*ratiow), int((self.y2-self.y1)*ratioh)

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


def convert(source_path, target_path, set_function, get_function, source_file_format, target_file_format, image_width,
            image_height, target_image_width = 0, target_image_height = 0, default_object_class = 0, default_score = 1.0):
    source_files = [os.path.basename(x) for x in glob.glob(source_path+"/*")]
    num_of_files = len(source_files)
    print("Number of files to convert: {}".format(num_of_files))
    s_class_marker_pos, s_score_marker_pos, s_X_marker_poss, s_seperator, _ = get_markers_poss(file_format=source_file_format)
    t_class_marker_pos, t_score_marker_pos, t_X_marker_poss, t_seperator, markers = get_markers_poss(
        file_format=target_file_format)
    object_class = default_object_class
    score = default_score
    if not os.path.exists(target_path):
        Path(target_path).mkdir(parents=True, exist_ok=True)
    for index, source_file in enumerate(source_files):
        print(f"Converting file {source_file} {index+1}/{num_of_files}: ", end="")
        with open(os.path.join(source_path, source_file)) as file:
            with open(os.path.join(target_path, source_file), "w") as output_file:
                for line in file.readlines():
                    values = line.split(s_seperator)
                    if s_class_marker_pos != -1:
                        object_class = values[s_class_marker_pos]
                        if SAVE_ONLY_PEOPLE_BALL_OBJECTS and object_class != "0" and object_class != "1":
                            continue
                        if CHANGE_OBJECT_CLASSID_TO_CLASSLABEL and object_class == "0":
                            object_class = "player"
                        if CHANGE_OBJECT_CLASSID_TO_CLASSLABEL and object_class == "1":
                            object_class = "ball"
                    if s_score_marker_pos != -1:
                        score = values[s_score_marker_pos]
                    x1 = values[s_X_marker_poss[0]]
                    x2 = values[s_X_marker_poss[1]]
                    x3 = values[s_X_marker_poss[2]]
                    x4 = values[s_X_marker_poss[3]]
                    # print("source: ", x1, x2, x3, x4)
                    set_function(float(x1), float(x2), float(x3), float(x4), image_width=image_width, image_height=image_height)
                    # print(get_function(target_image_width=target_image_width, target_image_height=target_image_height))
                    x1, x2, x3, x4 = get_function(target_image_width=target_image_width, target_image_height=target_image_height)

                    new_line = ""
                    for marker in range(markers):
                        marker_str = ""
                        if marker == t_class_marker_pos:
                            marker_str = object_class
                        if marker == t_score_marker_pos:
                            marker_str = score
                        if marker in t_X_marker_poss:
                            marker_str = t_X_marker_poss.index(marker)
                            if marker_str == 0:
                                marker_str = x1
                            elif marker_str == 1:
                                marker_str = x2
                            elif marker_str == 2:
                                marker_str = x3
                            elif marker_str == 3:
                                marker_str = x4
                        new_line = f"{t_seperator}".join([new_line, str(marker_str)])
                        new_line = new_line[1:] if new_line[0] == t_seperator else new_line
                    new_line += "\n"
                    output_file.write(new_line)
        print("Done!")

CHANGE_OBJECT_CLASSID_TO_CLASSLABEL = True
SAVE_ONLY_PEOPLE_BALL_OBJECTS = True

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Podaj jako 1 argument katalog źródłowy a jako 2 katalog przeszukiwany w którym zostaną usunięte pliki!")
        exit(0)

    boundingbox = BoundingBox()
    # Set file format where:
    ## C - means object class
    ## S - means score
    ## X - means cords (any x1/x2/y1/y2/cx/cy/w/h) order depends of used functions
    ##   - menas space seperator
    ## \t - means tab seperator
    ## ; - means semicolon seperator
    ## Example on line of file with YOLO format:
    ## 0 0.26299071311950684 0.7514517307281494 0.03339359909296036 0.1159067153930664
    ## Will be coded as "C X X X X"
    ## Other example with object class and score before cords and seperated by tab:
    ## 0	0.99	876	283	19	37
    ## Will be coded as "C\tS\tX\tX\tX\tX"
    # python tools/convert_labels.py "data/SoccerDB/jsladjf1_good" "data/SoccerDB/jsladjf1_good_d"
    # convert(sys.argv[1], sys.argv[2], boundingbox.set_from_YOLO, boundingbox.get_YOLO, "C X X X X", "C X X X X", image_width=1280, image_height=720, target_image_width=1280/2, target_image_height=720/2)
    convert(sys.argv[1], sys.argv[2], boundingbox.set_from_YOLO, boundingbox.get_x_y_w_h, "C X X X X", "C X X X X",
            image_width=1280, image_height=720)
    # python tools/convert_labels.py "data/SoccerNET/2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich/2_HQ_p/frames_labels" "data/SoccerNET/2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich/2_HQ_p/frames_labels_d"
    # convert(sys.argv[1], sys.argv[2], boundingbox.set_from_x1_y1_x2_y2, boundingbox.get_x_y_w_h, "C X X X X", "C X X X X",
    #         image_width=1280, image_height=720)
    # convert(sys.argv[1], sys.argv[2], boundingbox.set_from_x1_y1_x2_y2, boundingbox.get_x1_y1_x2_y2, "C\tS\tX\tX\tX\tX", "C S X X X X", image_width=1280, image_height=720, target_image_width=1280/2, target_image_height=720/2)
    # convert(sys.argv[1], sys.argv[2], boundingbox.set_from_x_y_w_h, boundingbox.get_x_y_w_h, "C\tS\tX\tX\tX\tX",
    #         "C S X X X X", image_width=1280, image_height=720)
    # Simple remove score from labels:
    # convert(sys.argv[1], sys.argv[2], boundingbox.set_from_x_y_w_h, boundingbox.get_x_y_w_h, "C S X X X X",
    #         "C X X X X", image_width=1280, image_height=720)
    # Example of resizing labels of bboxes for scaled images
    # convert(sys.argv[1], sys.argv[2], boundingbox.set_from_x_y_w_h, boundingbox.get_x_y_w_h, "C X X X X",
    #         "C X X X X", image_width=1920, image_height=1080, target_image_width=720, target_image_height=405)
    # convert(sys.argv[1], sys.argv[2], boundingbox.set_from_x_y_w_h, boundingbox.get_x_y_w_h, "C X X X X",
    #         "C X X X X", image_width=1280, image_height=720)
