from auto_detect_dialog_gui import Ui_Dialog
from PyQt5.QtWidgets import QDialog

class AutoDetectDialog(QDialog, Ui_Dialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setupUi(self)
        self.setWindowTitle("Autodetect Parameters")
    
    def get_params(self):
        params = {
            "points_per_side" : self.spinBox_points_per_side.value(),
            "points_per_batch" : self.spinBox_points_per_batch.value(),
            "pred_iou_thresh" : self.spinBox_quality_threshold.value(),
            "stability_score_thresh" : self.spinBox_stability_threshold.value(),
            "box_nms_thresh" : self.spinBox_nms_threshold.value(),
            "crop_n_layers" : self.spinBox_crop_n_layers.value(),
            "crop_nms_thresh" : self.spinBox_crop_nms_threshold.value(),
            "crop_overlap_ratio" : self.spinBox_crop_overlap_ratio.value(),
            "min_mask_region_area" : self.spinBox_min_mask_region.value(),
            "max_mask_region": self.spinBox_max_mask_region.value()
        }
        return params