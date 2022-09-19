from ODATT import ODATT
import os

if __name__ == "__main__":
    # Homography Models:
    homography_keypoint_path = "models/FPN_efficientnetb3_0.0001_4.h5"
    homography_deephomo_path = "models/HomographyModel_0.0001_4.h5"

    export_output_path = None
    export_frames_path = None
    no_gui = False

    input_dir = "data/FHD/20161019 BM/images_100"
    export_data_path_main = "data/FHD/20161019 BM/images_100_preds"
    # ## Faster-RCNN
    print("Run Faster-RCNN")
    object_detection_model_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    object_detection_config_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    export_data_path = os.path.join(export_data_path_main, "preds_FasterRCNN")
    tv2d = ODATT(object_detection_model_path, object_detection_config_path=object_detection_config_path,
                 no_gui=no_gui, homography_on=True, homography_pretreined=False,
                 homography_deephomo_path=homography_deephomo_path,
                 homography_keypoint_path=homography_keypoint_path, detector_type=ODATT.DetectorType.DETECTRON)
    tv2d(ODATT.RunOn.IMAGE, input_dir, export_output_path=export_output_path,
         export_data_file_path=export_data_path)
    del tv2d
    # ## MASK-RCNN:
    print("Run MASK-RCNN")
    object_detection_model_path = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    object_detection_config_path = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    export_data_path = os.path.join(export_data_path_main, "preds_MaskRCNN")
    tv2d = ODATT(object_detection_model_path, object_detection_config_path=object_detection_config_path,
                 no_gui=no_gui, homography_on=True, homography_pretreined=False,
                 homography_deephomo_path=homography_deephomo_path,
                 homography_keypoint_path=homography_keypoint_path, detector_type=ODATT.DetectorType.DETECTRON)
    tv2d(ODATT.RunOn.IMAGE, input_dir, export_output_path=export_output_path,
         export_data_file_path=export_data_path)
    del tv2d
    # ## Large-MASK-RCNN:
    print("Run Large-MASK-RCNN")
    object_detection_model_path = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
    object_detection_config_path = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
    export_data_path = os.path.join(export_data_path_main, "preds_LargeMaskRCNN")
    tv2d = ODATT(object_detection_model_path, object_detection_config_path=object_detection_config_path,
                 no_gui=no_gui, homography_on=True, homography_pretreined=False,
                 homography_deephomo_path=homography_deephomo_path,
                 homography_keypoint_path=homography_keypoint_path, detector_type=ODATT.DetectorType.DETECTRON)
    tv2d(ODATT.RunOn.IMAGE, input_dir, export_output_path=export_output_path,
         export_data_file_path=export_data_path)
    del tv2d
    # ## Panoptic:
    print("Run Panoptic")
    object_detection_model_path = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    object_detection_config_path = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    export_data_path = os.path.join(export_data_path_main, "preds_Panoptic")
    tv2d = ODATT(object_detection_model_path, object_detection_config_path=object_detection_config_path,
                 no_gui=no_gui, homography_on=True, homography_pretreined=False,
                 homography_deephomo_path=homography_deephomo_path,
                 homography_keypoint_path=homography_keypoint_path, detector_type=ODATT.DetectorType.DETECTRON)
    tv2d(ODATT.RunOn.IMAGE, input_dir, export_output_path=export_output_path,
         export_data_file_path=export_data_path)
    del tv2d
    # ## yolov5x6:
    print("Run yolov5x6")
    object_detection_model_path = "models/yolov5x6.pt"
    object_detection_config_path = "models/yolov5x6.pt"
    export_data_path = os.path.join(export_data_path_main, "preds_YOLOv5")
    tv2d = ODATT(object_detection_model_path, object_detection_config_path=object_detection_config_path,
                 no_gui=no_gui, homography_on=True, homography_pretreined=False,
                 homography_deephomo_path=homography_deephomo_path,
                 homography_keypoint_path=homography_keypoint_path, detector_type=ODATT.DetectorType.YOLOv5)
    tv2d(ODATT.RunOn.IMAGE, input_dir, export_output_path=export_output_path,
         export_data_file_path=export_data_path)
