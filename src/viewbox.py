import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QProgressDialog

import numpy as np
import os
from pathlib import Path

from dataset import Dataset
from model import Model
from annotation import Annotation
from utils import draw_segmentation_masks

class ViewBox(pg.ViewBox):
    def __init__(self, parent, *args, **kargs):
        super().__init__(*args, **kargs)
        self.parent = parent
        self.idx = 0
        self.img = None
        self.label_mode = "off"
        self.show_mask_mode = "last"
        self.class_label = "unspecified"
        self.circle = None
        self.alpha = 0.5
        self.mask_scale = 1
        
        # model inputs and outputs
        self.current_points = None
        self.current_labels = None
        self.current_masks = None

        self.last_selected_id = None

        self.dset = Dataset()
        self.model = Model()
        self.annot = Annotation()

    def clear_qt_objects(self):
        self.clear()
        self.circle = None
        self.img = None

    def auto_detect(self, annot_dict, predict_mode, model_type):
        self.parent.label_mode_on()
        if predict_mode == "current":
            annot = annot_dict[self.idx]
            self.auto_detect_single(annot,model_type)
        else:
            pb = QProgressDialog("Calculating masks", "Cancel", 0, len(self.dset))
            pb.setWindowModality(Qt.WindowModal)
            for i in range(len(self.dset)):
                self.navigate_to_idx(i)
                annot = annot_dict[i]
                self.auto_detect_single(annot, model_type)
                # update progress bar
                if pb.wasCanceled():
                    break
                pb.setValue(i)

    def auto_detect_single(self, annot, model_type):
        if model_type == "sam":
            masks = annot
        else:
            self.set_image()
            masks = []
            for bbox in annot:
                mask, _, _ = self.model.predict(input_box=bbox[None, :], multimask_output=False)
                masks.append(mask[0,:,:])
            masks = np.array(masks)
        self.annot.add_auto_detect_annot(self.idx, masks)
        self.update_img_annot(masks)
        for i in range(len(masks)): self.parent.add_mask(i+1)

    def load_images(self, fname):
        if len(fname):
            fname = Path(fname)
            self.dset.set_image_folder(fname)
            self.last_dir = os.path.dirname(fname)
            self.idx = 0
            self.set_image()

    def load_video(self, fname):
        if len(fname) > 0:
            fname = Path(fname)
            self.dset.set_video_name(fname)
            self.last_dir = os.path.dirname(fname)
            self.idx = 0
            self.set_image()
    
    def navigate_to_idx(self, idx):
        if len(self.dset) > 0:
            self.idx = idx
            masks = self.annot.get_mask(self.idx)
            if masks is not None:
                for i in range(len(masks)): self.parent.add_mask(i+1)
            
            self.show_mask()

    def prev(self):
        return (self.idx - 1) % len(self.dset)

    def next(self):
        return (self.idx + 1) % len(self.dset)

    def get_img(self):
        return self.dset[self.idx]

    def set_image(self) -> None:
        self.clear_qt_objects() # clear current contents
        img = self.dset[self.idx]
        
        if self.label_mode == "on":
            # if embedding cached use that instead,
            # else, compute and cache embedding
            if self.idx in self.annot.img_embed:
                self.model.set_cached_img_embed(**self.annot.img_embed[self.idx])
            else:
                self.model.set_image(img)
                img_embed = self.model.get_img_embed()
                self.annot.add_img_embed(self.idx, img_embed)
        
        self.img = pg.ImageItem(img)
        self.addItem(self.img) # show current image
        self.parent.update_hist()
    
    def set_show_mask_mode(self, show_mode):
        self.show_mask_mode = show_mode
        self.show_mask()

    def set_label_mode(self, label_mode):
        self.label_mode = label_mode
        self.set_image()

    def set_class_label(self, class_label):
        self.class_label = class_label
    
    def current_label_changed(self, class_label):
        self.class_label = class_label

    def show_mask(self):
        if self.show_mask_mode == "all":
            self.show_mask_all()
        elif self.last_selected_id is not None:
                self.show_mask_by_id(self.last_selected_id)

    def show_mask_by_id(self, mask_id):
        masks = self.annot.get_mask(self.idx)
        if masks is not None and len(masks) > 0:
            mask = masks[mask_id,:,:]
            self.update_img_annot(mask[None,:,:])
        else:
            self.update_img_annot(None)
    
    def show_mask_all(self):
        masks = self.annot.get_mask(self.idx)
        self.update_img_annot(masks)

    def add_points(self, pos, input_label):
        if self.current_points is None:
            self.current_points = np.array([[pos.x(), pos.y()]])
        else:
            self.current_points = np.vstack((self.current_points,
                                            np.array([[pos.x(), pos.y()]])))

        if self.current_labels is None:
            self.current_labels = np.array([input_label])
        else:
            self.current_labels = np.append(self.current_labels, input_label)

        self.update_annot()

    def update_annot(self):
        masks, scores, logits = self.model.predict(input_point=self.current_points,
                                                   input_label=self.current_labels)

        mask = masks[self.mask_scale, :, :]
        self.current_masks = mask
        
        mask = mask[None,:,:]
        prev_masks = self.annot.get_mask(self.idx)
        if self.show_mask_mode=="all" and prev_masks is not None:
            mask = np.vstack((prev_masks, mask))
        self.update_img_annot(mask)

    def update_img_annot(self, mask):
        img = self.dset[self.idx]
        if mask is not None:
            img = draw_segmentation_masks(img, mask, self.alpha)
        self.clear_qt_objects()
        self.img = pg.ImageItem(img)
        self.addItem(self.img)
        self.parent.update_hist()

    def mouseClickEvent(self, event: QMouseEvent) -> None:
        try:
            pos = self.calc_pos(event.pos())
        except:
            pos = QPoint(0,0)

        if self.label_mode == "on":
            if event.button() == Qt.LeftButton:
                self.add_points(pos, input_label=1) # 1 = foreground
            elif event.button() == Qt.RightButton:
                self.add_points(pos, input_label=0) # 0 = background
    
    def hoverEvent(self, event: QMouseEvent):
        if self.img is not None and self.label_mode == "on":
            try:
                pos = self.calc_pos(event.pos())
            except:
                pos = QPoint(0, 0)
            if self.circle is not None:
                self.removeItem(self.circle)
            self.circle = pg.CircleROI(pos, radius=.1)
            self.addItem(self.circle)
    
    def calc_pos(self, pos, offset=1):
        try:
            pos = self.mapSceneToView(pos)
            pos.setX(pos.x() + offset)
            pos.setY(pos.y() + offset)
            return pos
        except:
            return QPoint(0,0)
    
    def reset_current_annot(self):
        self.current_points = None
        self.current_labels = None
        self.current_masks = None
    
    def add_to_annot_list(self):
        masks = self.annot.get_mask(self.idx)
        mask_id = len(masks)
        self.parent.add_mask(mask_id)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.annot.add_annotation(self.idx,
                                      self.current_masks,
                                      self.class_label)
            self.reset_current_annot()
            self.add_to_annot_list()
        
        elif event.key() == Qt.Key_Escape:
            self.reset_current_annot()
            self.set_image()

        event.ignore()