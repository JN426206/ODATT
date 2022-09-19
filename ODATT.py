import argparse
import glob
import tqdm
import time
import tempfile
# Some basic setup:

# import some common libraries
import numpy as np
import os, json, cv2, random

import detector_detectron2
import detector_yolov5
import detected_object
import homography
from enum import Enum, auto

PERSON_OBJECT_CLASS_OUTPUT_LABEL = "player"
BALL_OBJECT_CLASS_OUTPUT_LABEL = "ball"
VIDEO_FRAME_STEP_ON = False
VIDEO_FRAME_STEP = 1000

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 football detector.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        help="Path to image or directory with images. Directory can only contain images!",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument("--export-data-path", help="A file to export data for every frame or image (frame_id, bounding boxes, score). "
                                                   "If path is file then save all data in one file "
                                                   "else if path is directory then save each frame or image output data in separate file."
                                                   "Attention! directory can not contain dots!")
    parser.add_argument("--export-frames-path", help="A directory to export all frames from video as images")
    parser.add_argument("--no-gui",
                        help="Disable gui mean will not disaplay opencv window good option on server if we want only export data file without process output video or image/s")

    return parser


class ODATT:
    class DetectorType(Enum):
        DETECTRON = auto()
        YOLOv5 = auto()

    class RunOn(Enum):
        VIDEO = auto()
        IMAGE = auto()

    PERSON_OBJECT_CLASS = 0
    BALL_OBJECT_CLASS = 32
    WINDOW_NAME = "ODATT"

    def __init__(self, object_detection_model_path, object_detection_config_path, detector_type=DetectorType.DETECTRON,
                 object_detection_threshold=0.3,
                 homography_on=True,
                 homography_keypoint_path="models/FPN_efficientnetb3_0.0001_4.h5",
                 homography_deephomo_path="models/HomographyModel_0.0001_4.h5",
                 homography_threshold=0.9, homography_pretreined=True, no_gui=False,
                 pitch_2D_width=320, pitch_2D_height=180):

        # No gui mean will not disaplay opencv window. Good option on server if we want only export data file without process output video or image/s.
        self.no_gui = no_gui

        self.pitch_2D_width = pitch_2D_width
        self.pitch_2D_height = pitch_2D_height
        self.export_file_path = None
        self.export_frames_path = None

        if detector_type == ODATT.DetectorType.DETECTRON:
            self.objectDetector = detector_detectron2.DetectronDetector(model_threshold=object_detection_threshold,
                                                                        model_path=object_detection_model_path,
                                                                        model_config_path=object_detection_config_path,
                                                                        detect_classes=[ODATT.PERSON_OBJECT_CLASS,
                                                                                        ODATT.BALL_OBJECT_CLASS])

        if detector_type == ODATT.DetectorType.YOLOv5:
            self.objectDetector = detector_yolov5.YOLOv5Detector(model_threshold=object_detection_threshold,
                                                                 model_path=object_detection_model_path,
                                                                 classes=[ODATT.PERSON_OBJECT_CLASS,
                                                                          ODATT.BALL_OBJECT_CLASS])

        self.homography = None
        if homography_on:
            if not homography_pretreined:
                assert homography_keypoint_path, "Homography keypoint path not set!"
                assert homography_deephomo_path, "Homography deephomo path not set!"
            # assert os.path.isfile(homography_keypoint_path), "Homography keypoint file not exists!"
            # assert os.path.isfile(homography_deephomo_path), "Homography deephomo file not exists!"
            self.homography = homography.Homography(weights_keypoint_path=(homography_keypoint_path),
                                                    weights_deephomo_path=(homography_deephomo_path),
                                                    pretreined=homography_pretreined, threshold=homography_threshold)

    def __call__(self, run_on, file_to_process_path, export_output_path="", object_detection_threshold=0.3,
                 homography_threshold=0.9, export_data_file_path="", export_frames_path=""):
        """_summary_

        Args:
            run_on (RunOn): RunOn.VIDEO for video or RunOn.IMAGE for image
            path (String): path to video or (image or images directory) depends which function choosed
            export_output_path (str, optional): Path to file where output (proccessed video or image) will be saved. If passed directory then saves in directory with source file name or names for images directory source. Defaults to "" mean no export.
            object_detection_threshold (float, optional): _description_. Defaults to 0.3.
            homography_threshold (float, optional): _description_. Defaults to 0.9.
            export_data_file_path (str, optional): Path to export data for every frame or image. If path is file then save all data in one file else if path is directory then save each frame or image output data in separate file. Attention! directory can not contain dots! Defaults to "" mean no export.
            export_frames_path (str, optional): Path to export every frame from video. Defaults to "" mean no export.
        """
        assert os.path.exists(file_to_process_path), f"{file_to_process_path} not exists!"
        if export_output_path:
            if "." not in export_output_path:
                os.makedirs(export_output_path, exist_ok=True)

        if self.homography is not None:
            self.homography.threshold = homography_threshold

        if export_data_file_path:
            if "." not in export_data_file_path:
                os.makedirs(export_data_file_path, exist_ok=True)
                self.export_file_path = export_data_file_path[:-1] if export_data_file_path[-1] == "/" else export_data_file_path
                export_file = None
            else:
                self.export_file_path = export_data_file_path
                os.makedirs(os.path.dirname(export_data_file_path), exist_ok=True)
                export_file = open(export_data_file_path, 'w')
        else:
            export_file = None

        if export_frames_path:
            if export_frames_path[-1] == "/":
                export_frames_path = export_frames_path[:-1]
            self.export_frames_path = export_frames_path
            os.makedirs(export_frames_path, exist_ok=True)

        # -------------------------- Video preparing and processing ----------------------- #
        if run_on == ODATT.RunOn.VIDEO:
            assert not os.path.isdir(
                file_to_process_path), "For video process acceptable is only video file not directory!"
            video = cv2.VideoCapture(file_to_process_path)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(file_to_process_path)
            codec, file_ext = (
                ("x264", ".mkv") if ODATT.test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
            )
            if codec == ".mp4v":
                print("x264 codec not available, switching to mp4v")
            if export_output_path:
                if os.path.isdir(export_output_path):
                    output_fname = os.path.join(export_output_path, basename)
                    output_fname = os.path.splitext(output_fname)[0] + file_ext
                else:
                    output_fname = export_output_path
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
            assert os.path.isfile(file_to_process_path)
            for vis_frame, frame_id, raw_frame in tqdm.tqdm(
                    self.run_on_video(video, objectDetector=self.objectDetector, export_file=export_file,
                                      homography=self.homography), total=num_frames):
                if export_output_path:
                    output_file.write(vis_frame)
                elif not self.no_gui:
                    cv2.namedWindow(f"{ODATT.WINDOW_NAME} {basename}", cv2.WINDOW_NORMAL)
                    cv2.imshow(f"{ODATT.WINDOW_NAME} {basename}", vis_frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
                if self.export_frames_path:
                    cv2.imwrite(f"{self.export_frames_path}/{frame_id}.jpg", raw_frame)
            video.release()
            if export_output_path:
                output_file.release()
            else:
                cv2.destroyAllWindows()
        # -------------------------- End of video preparing and processing ----------------------- #

        # --------------------------   Image or images processing -------------------------------- #
        if run_on == ODATT.RunOn.IMAGE:
            images = []
            if os.path.isdir(file_to_process_path):
                if file_to_process_path[-1] != "/":
                    file_to_process_path += "/"
                for image_path in sorted(glob.glob(f"{file_to_process_path}*.*")):
                    images.append(image_path)
            else:
                images.append(file_to_process_path)

            for index, path in enumerate(tqdm.tqdm(images, total=len(images))):
                image = cv2.imread(path)
                image = self.run_on_image(image, os.path.basename(path), objectDetector=self.objectDetector,
                                          export_file=export_file,
                                          homography=self.homography)

                if export_output_path:
                    if os.path.isdir(export_output_path):
                        assert os.path.isdir(export_output_path), export_output_path
                        out_filename = os.path.join(export_output_path, os.path.basename(path))
                    else:
                        out_filename = export_output_path
                    cv2.imwrite(out_filename, image)
                elif not self.no_gui:
                    cv2.namedWindow(ODATT.WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(ODATT.WINDOW_NAME, image)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
        # --------------------------  End image or images processing -------------------------------- #

        if export_file is not None and not export_file.closed:
            export_file.close()

    @staticmethod
    def export_parser(detecedObject, frame_id=None, file_name=None, all_data=True, file_format="csv"):
        object_class = detecedObject.object_class
        score = detecedObject.score
        bbox = detecedObject.bbox
        separator = " "
        if file_format.lower() == "csv":
            separator = ";"
        if file_format.lower() == "tsv":
            separator = "\t"

        first_row = ""
        if frame_id:
            first_row = frame_id
        elif file_name:
            first_row = file_name

        if object_class == ODATT.BALL_OBJECT_CLASS:
            object_class = BALL_OBJECT_CLASS_OUTPUT_LABEL
        else:
            object_class = PERSON_OBJECT_CLASS_OUTPUT_LABEL
        if all_data:
            # frame_id object_class	score bb_x bb_y bb_width bb_height
            return f"{first_row} {object_class} {score:0.2} {bbox[0]} {bbox[1]} {bbox[2] - bbox[0]} {bbox[3] - bbox[1]}\n".replace(
                " ", separator)
        else:
            # object_class score bb_x bb_y bb_width bb_height
            return f"{object_class} {score:0.2} {bbox[0]} {bbox[1]} {bbox[2] - bbox[0]} {bbox[3] - bbox[1]}\n".replace(
                " ", separator)

    @staticmethod
    def test_opencv_video_format(codec, file_ext):
        with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
            filename = os.path.join(dir, "test_file" + file_ext)
            writer = cv2.VideoWriter(
                filename=filename,
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(30),
                frameSize=(10, 10),
                isColor=True,
            )
            [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
            writer.release()
            if os.path.isfile(filename):
                return True
            return False

    def export_to_file(self, deteced_objects, export_file, frame_id=None, all_data=False):
        if os.path.isdir(self.export_file_path):
            file_format = "txt"
        else:
            file_format = os.path.basename(self.export_file_path).split(".")[-1].lower()
        for index, detecedObject in enumerate(deteced_objects):
            if not detecedObject.isout:
                if frame_id:
                    export_file.write(ODATT.export_parser(detecedObject, frame_id=frame_id, all_data=all_data,
                                                          file_format=file_format))
                else:
                    export_file.write(ODATT.export_parser(detecedObject, all_data=all_data, file_format=file_format))

    def __frame_from_video(self, video):
        frame_id = 0
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame_id += 1
                if VIDEO_FRAME_STEP_ON and frame_id % VIDEO_FRAME_STEP == 0:
                    yield frame, frame_id
                else:
                    yield frame, frame_id
            else:
                break

    @staticmethod
    def draw_bboxes_on_image(deteced_objects, image):
        for index, detecedObject in enumerate(deteced_objects):
            color = (255, 0, 0)
            if detecedObject.object_class == ODATT.BALL_OBJECT_CLASS:
                color = (0, 0, 255)
            box = detecedObject.bbox
            if not detecedObject.isout:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        return image

    @staticmethod
    def merge_frame_with_pitch(frame, pitch):
        x_offset = 0
        y_offset = frame.shape[0] - pitch.shape[0]
        x_end = x_offset + pitch.shape[1]
        y_end = y_offset + pitch.shape[0]
        frame[y_offset:y_end, x_offset:x_end] = pitch
        return frame

    def run_engine(self, frame, frame_id, detected_objects, objectDetector, export_file=None, homographyDetector=None):
        """_summary_

        Args:
            frame (Mat): cv2.imread() or any array with image with compatibile format like cv2
            frame_id (str): frame_id or file name
            detected_objects (_type_): _description_
            objectDetector (Detector): _description_
            export_file (File, optional): file object. Defaults to None.
            homography (Homography, optional): _description_. Defaults to None.
        Returns:
            Mat: cv2.imread() array
        """

        if homographyDetector is not None:
            predicted_homography = homographyDetector.predict_homography(frame)

            homographyDetector.check_object_isout(frame.shape, detected_objects, predicted_homography)

        frame = ODATT.draw_bboxes_on_image(detected_objects, frame)

        if export_file is not None and not export_file.closed:
            self.export_to_file(detected_objects, export_file, frame_id, all_data=True)

        if export_file is None and isinstance(self.export_file_path, str) and os.path.isdir(self.export_file_path):
            _frame_id = frame_id
            if "." in _frame_id:
                _frame_id = ".".join(frame_id.split(".")[:-1])
            _export_file = open(f"{self.export_file_path}/{_frame_id}.txt", 'w')
            self.export_to_file(detected_objects, _export_file, None, all_data=False)
            _export_file.close()

        return frame

    def run_on_video(self, video, objectDetector, export_file=None, homography=None):
        """_summary_

        Args:
            video (VideoCapture): cv2.VideoCapture() object only 
            objectDetector (Detector): _description_
            export_file (File, optional): file object. Defaults to None.
            homography (Homography, optional): _description_. Defaults to None.

        Yields:
            Mat: cv2.imread() array frame with drawn bboxes
            Int: frame_id
            Mat: cv2.imread() array raw frame
        """
        for frame, frame_id in self.__frame_from_video(video):
            raw_frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detected_objects = objectDetector.detect_objects(frame)
            frame = self.run_engine(frame, str(frame_id), detected_objects, objectDetector, export_file=export_file,
                                    homographyDetector=homography)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield frame, frame_id, raw_frame

    def run_on_image(self, image, file_name, objectDetector, export_file=None, homography=None):
        """_summary_

        Args:
            image (Mat): cv2.imread() or any array with image with compatibile format like cv2
            objectDetector (Detector): _description_
            export_file (File, optional): file object. Defaults to None.
            homography (Homography, optional): _description_. Defaults to None.
            
        Returns:
            Mat: cv2.imread() array
        """
        start_time = time.time()

        detected_objects = objectDetector.detect_objects(image)
        # print("{}: {} in {:.2f}s".format(
        #     file_name,
        #     "detected {} instances".format(len(detected_objects))
        #     if len(detected_objects)
        #     else "finished",
        #     time.time() - start_time,
        # ))

        image = self.run_engine(image, file_name, detected_objects, objectDetector, export_file=export_file,
                                homographyDetector=homography)

        return image


if __name__ == "__main__":
    args = get_parser().parse_args()
    # Set path to the model or Detectron2 Model Zoo and Baselines models file convention name like as default. 
    ## The model with the Detectron2 convention will be downloaded automatically  (if use set detector_type=ODATT.DetectorType.DETECTRON)
    # object_detection_model_path = "data/detectron2_models/mask_rcnn_X_101_32x8d_FPN_3x_model_final_2d9806.pkl"
    ## Fast-RCNN: For use Fast-RCNN there is need to first detect RPN i.e. by "RPN R50-FPN" and pass them to the Fast-RCNN witch give as same feature as has Faster-RCNN so there is non sense to using Fast-RCNN seperatelly
    # object_detection_model_path = "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"
    # ## Faster-RCNN
    object_detection_model_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    # object_detection_model_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    # object_detection_model_path = "data/output/model_final_faster_rcnn_R_50_FPN_3x.pth"
    # object_detection_model_path = "data/output/model_final.pth"
    # ## MASK-RCNN:
    # object_detection_model_path = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    # object_detection_model_path = "data/output/model_final_mask_rcnn_X_101_32x8d_FPN_3x.pth"
    # ## Large-MASK-RCNN:
    # object_detection_model_path = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
    # ## Panoptic6
    # object_detection_model_path = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    # Same as model by for now for config. If you use model from local sotrage you can still use config 
    ## from Detectron2 Model Zoo and Baselines.
    ## Fast-RCNN:
    # object_detection_config_path = "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"
    # ## Faster-RCNN
    object_detection_config_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    # object_detection_config_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    # object_detection_config_path = "data/output/model_final_faster_rcnn_R_50_FPN_3x.yml"
    # object_detection_config_path = "data/output/model_final.yml"
    # ## MASK-RCNN:
    # object_detection_config_path = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    # object_detection_config_path = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    # object_detection_config_path = "data/output/model_final_mask_rcnn_X_101_32x8d_FPN_3x.yml"
    # ## Large-MASK-RCNN:
    # object_detection_config_path = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
    # ## Panoptic
    # object_detection_config_path = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"

    ## YOLOv5 models (if use set detector_type=ODATT.DetectorType.YOLOv5):
    ### The model with the YOLOv5 convention will be downloaded automatically
    # yolov5x6:
    # object_detection_model_path = "yolov5x6"
    # object_detection_config_path = None
    # object_detection_model_path = "models/yolov5x6.pt"

    # Homography Models:
    homography_keypoint_path = "models/FPN_efficientnetb3_0.0001_4.h5"
    homography_deephomo_path = "models/HomographyModel_0.0001_4.h5"

    export_output_path = None
    export_data_path = None
    export_frames_path = None
    no_gui = False
    if args.output:
        export_output_path = args.output

    if args.export_data_path:
        export_data_path = args.export_data_path

    if args.export_frames_path:
        export_frames_path = args.export_frames_path

    if args.no_gui:
        no_gui = True

    #############
    # ATTENTION #
    #############
    # If you use your own trained model please check if you change PERSON_OBJECT_CLASS=0 and BALL_OBJECT_CLASS=1 in ODATT class!

    if args.input:
        tv2d = ODATT(object_detection_model_path, object_detection_config_path=object_detection_config_path,
                     no_gui=no_gui, homography_on=True, homography_pretreined=False,
                     homography_deephomo_path=homography_deephomo_path,
                     homography_keypoint_path=homography_keypoint_path, detector_type=ODATT.DetectorType.DETECTRON)
        tv2d(ODATT.RunOn.IMAGE, args.input, export_output_path=export_output_path,
             export_data_file_path=export_data_path)
    elif args.video_input:
        tv2d = ODATT(object_detection_model_path, object_detection_config_path=object_detection_config_path,
                     homography_on=True, no_gui=no_gui, homography_pretreined=False,
                     homography_deephomo_path=homography_deephomo_path,
                     homography_keypoint_path=homography_keypoint_path, detector_type=ODATT.DetectorType.DETECTRON)
        tv2d(ODATT.RunOn.VIDEO, args.video_input, export_output_path=export_output_path,
             export_data_file_path=export_data_path, export_frames_path=export_frames_path)

# Used platforms:

# @misc{wu2019detectron2,
#   author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
#                   Wan-Yen Lo and Ross Girshick},
#   title =        {Detectron2},
#   howpublished = {\url{https://github.com/facebookresearch/detectron2}},
#   year =         {2019}
# }

# https://github.com/ultralytics/yolov5

# @misc{garnier2021evaluating,
#       title={Evaluating Soccer Player: from Live Camera to Deep Reinforcement Learning},
#       author={Paul Garnier and Théophane Gregoir},
#       year={2021},
#       eprint={2101.05388},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }
