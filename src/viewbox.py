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
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        self.idx = 0
        self.img = None
        self.label_mode = "mask_on"
        self.class_label = "unspecified"
        self.class_color_dict = {}
        self.circle = None
        self.alpha = 0.5
        self.mask_scale = 1

        self.dset = Dataset()
        self.model = Model()
        self.annot = Annotation()

    def clear_qt_objects(self):
        self.clear()
        self.circle = None
        self.img = None

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

    def set_image(self) -> None:
        self.clear_qt_objects() # clear current contents
        img = self.dset[self.idx]
        self.model.set_image(img)
        self.img = pg.ImageItem(img)
        self.addItem(self.img) # show current image

    def set_class_label(self, class_label, color):
        self.class_label = class_label
        self.class_color_dict[class_label] = color
    
    def current_label_changed(self, class_label):
        self.class_label = class_label

    def mouseClickEvent(self, event: QMouseEvent) -> None:
        pos = self.calc_pos(event.pos())
        
        if self.label_mode == "mask_on":
            if event.button() == Qt.LeftButton:
                input_point = np.array([[pos.x(), pos.y()]])
                input_label = np.array([1]) # foreground
                masks, scores, logits = self.model.predict(input_point=input_point,
                                                           input_label=input_label)
                
                mask = masks[self.mask_scale,:,:]
                mask = mask[None,:,:]

                img = self.dset[self.idx]
                img = draw_segmentation_masks(img, mask, self.alpha)
                self.clear_qt_objects()
                self.img = pg.ImageItem(img)
                self.addItem(self.img)
            elif event.button() == Qt.RightButton:
                print("remove point", pos)
    
    def hoverEvent(self, event: QMouseEvent):
        if self.img is not None:
            pos = self.calc_pos(event.pos())
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