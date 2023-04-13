from auto_detect_dialog_gui import Ui_Dialog
from PyQt5.QtWidgets import QDialog, QErrorMessage

from model import SAMAuto, MOG2, Canny, Farneback

class AutoDetectDialog(QDialog, Ui_Dialog):
    def __init__(self, dataset, frame_idx, sam, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setupUi(self)
        self.setWindowTitle("Autodetect Parameters")
        
        self.dset = dataset
        self.sam = sam
        self.model = None
        self.model_type = "sam"
        self.frame_idx = frame_idx
        self.predict_mode = "current"
        self.params = {}
        self.output = {}

        # accept/reject button box
        self.buttonBox.accepted.connect(self.predict)
        self.buttonBox.rejected.connect(self.reject)
        
        # model selection
        self.radio_canny.clicked.connect(self.get_params_canny)
        self.radio_mog2.clicked.connect(self.get_params_mog2)
        self.radio_farneback.clicked.connect(self.get_params_franeback)
        self.radio_sam.clicked.connect(self.get_params_sam)

        # current frame vs all frames
        self.radio_all_frames.clicked.connect(self.predict_all_frames)
        self.radio_current_frame.clicked.connect(self.predict_current_frame)
    
    def predict(self):
        self.construct_model()
        if self.model is None:
            self.accept()
        if self.predict_mode == "current":
            self.output[self.frame_idx] = self.model.predict(self.frame_idx)
        else:
            for i in range(len(self.dset)):
                self.output[i] = self.model.predict(i)
        self.accept()
    
    def construct_model(self):
        if self.model_type == "sam":
            self.model = SAMAuto(self.dset, self.sam, **self.params)
        elif self.model_type == "canny":
            self.model = Canny(self.dset, **self.params)
        
        elif self.model_type == "mog2":
            self.model = MOG2(self.dset, **self.params)
        
        elif self.model_type == "farneback":
            self.model = Farneback(self.dset, **self.params)
        
        else:
            raise NotImplementedError

    def predict_all_frames(self):
        self.predict_mode = "all"

    def predict_current_frame(self):
        self.predict_mode = "current"

    def get_params_sam(self):
        self.model_type = "sam"
        self.params = {
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

    def get_params_canny(self):
        self.model_type = "canny"
        self.params = {
            "th1":self.spinBox_canny_th1.value(),
            "th2":self.spinBox_canny_th2.value(),
            "min_area":self.spinBox_canny_min_area.value(),
            "it_closing":self.spinBox_canny_closing_it.value()
        }

    def get_params_mog2(self):
        self.model_type = "mog2"
        self.params = {
            "history":self.spinBox_mog2_history.value(),
            "varThreshold":self.spinBox_mog2_th.value(),
            "it_closing":self.spinBox_mog2_closing_it.value(),
            "min_area":self.spinBox_mog2_min_area.value()
        }

    def get_params_franeback(self):
        self.model_type = "farneback"
        self.params = {
            "pyr_scale":self.spinBox_farneback_py_scale.value(),
            "levels": self.spinBox_farneback_levels.value(),
            "winsize": self.spinBox_farneback_win_size.value(),
            "iterations": self.spinBox_farneback_it.value(),
            "poly_n": self.spinBox_farneback_poly_n.value(),
            "poly_sigma": self.spinBox_farneback_poly_sigma.value()
        }