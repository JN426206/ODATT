import logging

import torch

import detector
from detected_object import DetectedObject
# from yolov5.models.common import DetectMultiBackend
# https://github.com/ultralytics/yolov5/issues/36
class YOLOv5Detector(detector.Detector):

    def __init__(self, model_threshold=0.5, model_path="yolov5x6", classes=[0, 32]):
        """_summary_

        Args:
            model_threshold (float, optional): _description_. Defaults to 0.5.
            model_path (str, optional): Set YOLOv5 model type (i.e. yolov5s6, yolov5m6, yolov5l6, yolov5x6)
        """
        # logging.getLogger("yolov5").setLevel(logging.WARNING)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path)
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = model_threshold  # NMS IoU threshold
        self.model.multi_label = True  # NMS multiple labels per box
        self.model.classes = classes  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs



    def detect_objects(self, image, classes=[0, 32]) -> [DetectedObject]:
        # Detect objects and if model have panoptic or instance segmenteation detects mask too
        # https://github.com/ultralytics/yolov5/blob/ffbce3858ae3d0d1d0978a5927daa2d4f94e55b6/models/common.py#L591
        # https://github.com/ultralytics/yolov5/blob/ffbce3858ae3d0d1d0978a5927daa2d4f94e55b6/models/common.py#L655
        predictions = self.model(image, size=image.shape[1])  # custom inference size
        detected_objects = []
        for index, det_obj in predictions.pandas().xyxy[0].iterrows():
            box = [int(det_obj['xmin']), int(det_obj['ymin']), int(det_obj['xmax']), int(det_obj['ymax'])]
            score = det_obj['confidence']
            pred_class = det_obj['class']
            detected_objects.append(DetectedObject(box, pred_class, score))
        return detected_objects

