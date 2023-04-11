import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent

import numpy as np
import os
from pathlib import Path

from dataset import Dataset
from model import Model, Annotation
from utils import draw_segmentation_masks

class ViewBox(pg.ViewBox):
    def __init__(self, parent, *args, **kargs):
        super().__init__(*args, **kargs)
        self.parent = parent
        self.idx = 0
        self.img = None
        self.label_mode = "off"
        self.class_label = "unspecified"
        self.circle = None
        self.alpha = 0.5
        self.mask_scale = 1
        
        # model inputs and outputs
        self.current_points = None
        self.current_labels = None
        self.current_masks = None

        self.dset = Dataset()
        self.model = Model()
        self.annot = Annotation()

    def clear_qt_objects(self):
        self.clear()
        self.circle = None
        self.img = None

    def auto_detect(self):
        img = self.get_img()
        annots = self.model.generate(img)
        self.annot.add_auto_detect_annot(self.idx, annots)
        masks = self.annot.get_mask(self.idx)
        self.update_img_annot(masks)
        self.update_annot_list()

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
            self.set_image()

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
            self.model.set_image(img)
        
        img_annot = self.annot.get_image(self.idx)
        if self.label_mode == "on" and img_annot is not None:
            self.img = pg.ImageItem(img_annot)
        else:
            self.img = pg.ImageItem(img)
        self.addItem(self.img) # show current image
        self.parent.update_hist()
    
    def set_label_mode(self, label_mode):
        self.label_mode = label_mode
        if label_mode == "on":
            self.set_image()

    def set_class_label(self, class_label):
        self.class_label = class_label
    
    def current_label_changed(self, class_label):
        self.class_label = class_label

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
        if prev_masks is not None:
            mask = np.vstack((prev_masks, mask))
        self.update_img_annot(mask)

    def update_img_annot(self, mask):
        img = self.dset[self.idx]
        img = draw_segmentation_masks(img, mask, self.alpha)
        self.img_annot = img
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
        self.current_scores = None
        self.current_logits = None
        self.current_color = None
    
    def update_annot_list(self):
        masks = self.annot.get_mask(self.idx)
        mask_id = len(masks)
        self.parent.add_mask(mask_id)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.annot.add_annotation(self.idx,
                                      self.current_masks,
                                      self.class_label)
            self.annot.set_image(self.idx, self.img_annot)
            self.reset_current_annot()
            self.update_annot_list()
        
        elif event.key() == Qt.Key_Escape:
            self.reset_current_annot()
            self.set_image()

        event.ignore()