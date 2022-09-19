# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

dataset_directory = "data/dataset_for_detectron/"
cfg = None
###################
# Prepare dataset #
###################

def get_dicts(imgdir):
    json_file = imgdir+"/dataset.JSON" #Fetch the json file
    with open(json_file) as f:
        dataset_dicts = json.load(f)
        dicts = []
        for key in dataset_dicts:
            dataset_dicts[key]["file_name"] = os.path.join(imgdir, dataset_dicts[key]["file_name"])
            # annos = []
            # for anno in dataset_dicts[key]["annotations"]:
            #     anno["bbox_mode"] = BoxMode.XYWH_ABS  # Setting the required Box Mode
            #     anno["category_id"] = int(anno["category_id"])
            #     annos.append(anno)
            # dataset_dicts[key]["annotations"] = annos
            dicts.append(dataset_dicts[key])
        # print(dicts)
    return dicts

for d in ["train", "val"]:
    DatasetCatalog.register("football_" + d, lambda d=d: get_dicts(os.path.join(dataset_directory, d)))
    MetadataCatalog.get("football_" + d).set(thing_classes=["person", "ball"])
football_metadata = MetadataCatalog.get("football_train")


#import the COCO Evaluator to use the COCO Metrics
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

#register your data
register_coco_instances("my_dataset_train", {}, "/code/detectron2/detectron2/instances_train2017.json", "/code/detectron2/detectron2/train2017")
register_coco_instances("my_dataset_val", {}, "/code/detectron2/detectron2/instances_val2017.json", "/code/detectron2/detectron2/val2017")
register_coco_instances("my_dataset_test", {}, "/code/detectron2/detectron2/instances_test2017.json", "/code/detectron2/detectron2/test2017")

#load the config file, configure the threshold value, load weights
cfg = get_cfg()
cfg.merge_from_file("/code/detectron2/detectron2/output/custom_mask_rcnn_X_101_32x8d_FPN_3x_Iteration_3_dataset.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "/code/detectron2/detectron2/output/model_final.pth"

# Create predictor
predictor = DefaultPredictor(cfg)

#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")

#Use the created predicted model in the previous step
inference_on_dataset(predictor.model, val_loader, evaluator)