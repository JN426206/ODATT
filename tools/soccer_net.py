# Skrypt do generowania obrazów z adnotacjami z nagrania meczu z datasetu SoccerNet
# Na pierwszej klatce za pomocą przycisków
# W - wczytanie następnej klatki
# A - Poprzednia predykcja
# S - Następna predykcja
# Można ustawić początkową klatkę pokrywającą się z predykcją. Z reguły będzie to jakaś dalsza klatka.... Każde przjście do następnej klatki jest printowane tak aby zapamiętać na jakiej klatce było dobrz.
# Jeżeli mamy już dobrz ustawioną pierwszą predykcję to możemy wcisnąć Q. Wtedy program będzie skakać co ustawiony step do kolejnej predykcji.
# Q - Skakanie pmiędzy oznaczonymi już klatkami z zatrzymywaniem na każdej
# S - Zapisanie klatki z adnotacjami i jeżeli ustawiliśmy zmienną "save_pred_frame_output_path" to także klatki z bboxami
# E - Automatyczne skakanie i zapisywanie klatek
# Przed generowaniem plików można przekonwertować plik wideo z meczu za pomocą polecenia:
# Jeżeli masz GPU Nvidii:
# ffmpeg -hwaccel nvdec -i "1_HQ.mkv" -c:v h264_nvdec -vf yadif=1 -refs 2 -r 25 -vcodec h264_nvenc -b:v 8000k "1_HQ_p.mkv"
# ffmpeg -hwaccel nvdec -i "2_HQ.mkv" -c:v h264_nvdec -vf yadif=1 -refs 2 -r 25 -vcodec h264_nvenc -b:v 8000k "2_HQ_p.mkv"
# Jeżeli masz tylko CPU:
# ffmpeg -i "1_HQ.mkv" -c:v libx264 -vf yadif=1 -refs 2 -r 25 -vcodec libx264 -b:v 8000k "1_HQ_p.mkv"
# Pozbędziemy się takich pasków przy ruchu
# Trzeba kilka razy uruchamiać program żeby dobrze synchronizować
# Lepiej na samym wideo znaleźć kiedy rozpoczyna się połowa i wpisać do part_start_time, automatycznie obliczy się, która to jest klatka.
import numpy as np
import cv2 as cv
import json
import os
from pathlib import Path
from datetime import timedelta
# import matplotlib.pyplot as plt
# from caffe.proto import caffe_pb2

# lmdb_env = lmdb.open('data/bbox_lmdb')
# lmdb_txn = lmdb_env.begin()
# lmdb_cursor = lmdb_txn.cursor()

# for key, value in lmdb_cursor:
#     print(key, value)
#     break

main_dir = "data/SoccerNET/2016-10-19 - 21-45 Barcelona 4 - 0 Manchester City"
video_name = "2_HQ_p.mp4"
labels_file = "2_HQ_25_player_bbox.json"
output_frames_dir = ".".join(video_name.split(".")[:-1])
if not os.path.exists(os.path.join(main_dir, video_name)):
    print(f"Error! video {video_name} not exists in {main_dir}")
    exit(1)
if main_dir[-1] == "/":
    main_dir = main_dir[:-1]
vidcap = cv.VideoCapture(f"{main_dir}/{video_name}")

part_start_time = timedelta(hours=0, minutes=2, seconds=30).seconds * vidcap.get(cv.CAP_PROP_FPS)
start_from_frame_number = 3939 # part_start_time  # Use part_start_time or provide own start frame number
print("Start from frame number: ", start_from_frame_number)
step_extract_frame = 100

success,clear_image = vidcap.read()
count = 0
while success and count != start_from_frame_number:
  success,clear_image = vidcap.read()
  count += 1
# cv.imshow("Image",image)
# cv.waitKey(0)
prediction=0
trzynastka = 0
co_12_13_count = 0
start = False
run = True
count_step = 0
save_frame_output_path = f"{main_dir}/{output_frames_dir}/frames"
save_frame_labels = f"{main_dir}/{output_frames_dir}/frames_labels"
save_pred_frame_output_path = f"{main_dir}/{output_frames_dir}/pred_frames"
if save_frame_output_path[-1] == "/":
    save_frame_output_path = main_dir[:-1]
if save_frame_labels[-1] == "/":
    save_frame_labels = save_frame_labels[:-1]
if save_pred_frame_output_path[-1] == "/":
    save_pred_frame_output_path = save_pred_frame_output_path[:-1]

if not os.path.exists(save_frame_output_path):
    Path(save_frame_output_path).mkdir(parents=True, exist_ok=True)
if len(save_pred_frame_output_path) > 0 and not os.path.exists(save_pred_frame_output_path):
    Path(save_pred_frame_output_path).mkdir(parents=True, exist_ok=True)
if not os.path.exists(save_frame_labels):
    Path(save_frame_labels).mkdir(parents=True, exist_ok=True)

autosave = False
with open(f'{main_dir}/{labels_file}') as json_file:
    data = json.load(json_file)
    while True:
        if start:
            if not trzynastka and co_12_13_count == 12:
                trzynastka = 1
                co_12_13_count = 0

                prediction += 1
                p = data['predictions'][prediction]
                if len(p['bboxes']) > 6:
                    if count_step > step_extract_frame and run:
                        run = False
                        count_step = 0
            elif co_12_13_count == 13:
                trzynastka = 0
                co_12_13_count = 0

                prediction += 1
                p = data['predictions'][prediction]
                if len(p['bboxes']) > 6:
                    if count_step > step_extract_frame and run:
                        run = False
                        count_step = 0

            if not run:
                for box in p['bboxes']:
                    cv.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,0,255), 2)
                cv.imshow(f"Image frame", image)
                k = cv.waitKey(1)
                if k == 113:  # Q
                    run = True
                if autosave or k == 115:  # S
                    cv.imwrite(f"{save_frame_output_path}/{count}_{prediction}.jpg", clear_image)
                    if len(save_pred_frame_output_path) > 0:
                        cv.imwrite(f"{save_pred_frame_output_path}/{count}_{prediction}.jpg", image)

                    f = open(f"{save_frame_labels}/{count}_{prediction}.txt", "w")
                    for box in p['bboxes']:
                        cv.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,0,255), 2)
                        f.write(f"player {box[0]} {box[1]} {box[2] - box[0]} {box[3] - box[1]}\n")
                    f.close()
                    run = True
                if k==27:
                    exit()

            if prediction+1 >= len(data['predictions']):
                break

            if run:
                count += 1
                co_12_13_count += 1
                count_step += 1
                success,clear_image = vidcap.read()
                image = clear_image.copy()
                if not success:
                    break

        if not start:
            image = clear_image.copy()
            p = data['predictions'][prediction]
            for box in p['bboxes']:
                cv.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,0,255), 2)
            cv.imshow(f"Image frame",image)

            k=cv.waitKey(1)
            if k==27:
                break
            if k==113: # Q
                start = True
            if k==101: # E
                autosave = True
                start = True
            if k==119: # W
                success,clear_image = vidcap.read()
                if not success:
                    break
                else:
                    count += 1
                    print(count)
            if k==97: # A
                if prediction>0:
                    prediction -= 1
            if k==115: # S
                prediction += 1
# for box in boxes[0]:
#     if i==length:
#         break
#     if score[i]>0.5:
#         cv.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0,0,255), 2)
#     i += 1

