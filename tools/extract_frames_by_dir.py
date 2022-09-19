# Tool for extract frames from video based on content of direcotry with files extracted by soccer_net.py
import glob
import os
import sys
import cv2

def scan(source, target, video_file):
    files = [os.path.basename(x) for x in sorted(glob.glob(source+"/*", recursive=True))]
    numoffiles = len(files)
    print("Number of files: {}".format(numoffiles))
    id_frames_to_extract = {}
    for file in files:
        id_frames_to_extract[file.split("_")[0]]= file

    vidcap = cv2.VideoCapture(video_file)
    success = True
    id_frame = 0
    os.makedirs(target, exist_ok=True)
    while success:
        success, clear_image = vidcap.read()
        if str(id_frame) in id_frames_to_extract:
            cv2.imwrite(os.path.join(target, id_frames_to_extract[str(id_frame)]), clear_image)
        id_frame += 1


if __name__ == "__main__":
    #print(len(sys.argv))
    if len(sys.argv)<2:
        print("Pass source directory as first argument and target directory where frames will be saved as second argument and video file as third argument!")
        exit(0)

    scan(sys.argv[1],sys.argv[2],sys.argv[3])