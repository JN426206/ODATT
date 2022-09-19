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

# https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references
# https://developpaper.com/using-detectron2-to-detect-targets-in-6-steps/
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=7unkuuiqLdqd
# https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/configs.md
# https://pallawi-ds.medium.com/detectron2-evaluation-cocoevaluator-9f1ab0236d4c

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
    MetadataCatalog.get("football_" + d).set(thing_classes=["person", "ball"]) # ATTENTION if you use your own trained model please check if you change PERSON_OBJECT_CLASS=0 and BALL_OBJECT_CLASS=1 in ODATT class!
football_metadata = MetadataCatalog.get("football_train")

dataset_dicts = get_dicts(os.path.join(dataset_directory, "train"))
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=football_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("Train preview", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

######################
# Configure training #
######################
class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) #Get the basic model configuration from the model zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")) #Get the basic model configuration from the model zoo
#Passing the Train and Validation sets
cfg.DATASETS.TRAIN = ("football_train",)
cfg.DATASETS.TEST = ("football_val",)  # Uncomment if use `trainer = CocoTrainer(cfg)` or comment if use `trainer = DefaultTrainer(cfg)`
# cfg.DATASETS.TEST = ()  # Uncomment if use trainer = DefaultTrainer(cfg) or comment if use `trainer = CocoTrainer(cfg)`
# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
# Number of images per batch across all machines.
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.0125  # pick a good LearningRate
cfg.SOLVER.MAX_ITER = 1500  #No. of iterations
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # No. of classes = [person, ball]
cfg.TEST.EVAL_PERIOD = 500 # No. of iterations after which the Validation Set is evaluated. Uncomment if use `trainer = CocoTrainer(cfg)` or comment if use `trainer = DefaultTrainer(cfg)`
cfg.MODEL.MASK_ON = False
cfg.MODEL.KEYPOINT_ON = False
cfg.MODEL.BACKBONE.FREEZE_AT = 2  # Freeze the first several stages so they are not trained. There are 5 stages in ResNet. The first is a convolution, and the following stages are each group of residual blocks.

############
# Training #
############
def training():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print(cfg.dump())
    with open(os.path.join(cfg.OUTPUT_DIR, "model_final.yml"), "w") as file:
        file.write(cfg.dump())

#############
# Inference #
#############
def inference():
    #Use the final weights generated after successful training for inference
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set the testing threshold for this model
    #Pass the validation dataset
    cfg.DATASETS.TEST = ("football_val", )

    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_dicts(os.path.join(dataset_directory, "val"))
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=football_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu")) #Passing the predictions to CPU from the GPU
        cv2.imshow("Inference preview", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)

############
# Evaluate #
############
def evaluate():
    #Use the final weights generated after successful training for inference
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.merge_from_file(os.path.join(cfg.OUTPUT_DIR, "model_final_faster_rcnn_R_50_FPN_3x.yml"))  # Get the basic model configuration from the model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_faster_rcnn_R_50_FPN_3x.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set the testing threshold for this model
    #Pass the validation dataset
    cfg.DATASETS.TEST = ("football_val", )

    predictor = DefaultPredictor(cfg)
    #Call the COCO Evaluator function and pass the Validation Dataset
    evaluator = COCOEvaluator("football_val", cfg, False, output_dir="output/")

    val_loader = build_detection_test_loader(cfg, "football_val")

    #Use the created predicted model in the previous step
    inference_on_dataset(predictor.model, val_loader, evaluator)

if __name__ == "__main__":
    training()
    inference()
    evaluate()