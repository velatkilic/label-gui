import sys
import os
from pathlib import Path
import json

from gui import Ui_MainWindow
from viewbox import ViewBox
from dataset import Dataset
from model import Model

from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication,
                            QMainWindow,
                            QFileDialog,
                            QListWidgetItem,
                            QColorDialog,
                            )
import pyqtgraph as pg

pg.setConfigOption('imageAxisOrder', 'row-major') # best performance

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        # set-up the GUI
        self.setupUi(self)

        # main window title
        self.setWindowTitle("Human-in-the-Loop Annotation")

        # dataset
        self.last_dir = Path(os.getcwd())
        self.idx = 0
        self.dset = Dataset()

        # model
        # self.model = Model()

        # label mode
        self.label_mode = "segmentation"
        self.radio_segmentation.clicked.connect(self.label_mode_segmentation)
        self.radio_bbox.clicked.connect(self.label_mode_bbox)
        self.radio_off.clicked.connect(self.label_mode_off)

        # view_box holds images and ROIs
        self.view_box = ViewBox(lockAspect=True, invertY=True)
        self.hist = pg.HistogramLUTItem()

        self.img_view.addItem(self.view_box, row=0, col=0, rowspan=1, colspan=1)
        self.img_view.addItem(self.hist, row=0, col=1, rowspan=1, colspan=1)
        
        # class labels
        self.button_add_class_label.clicked.connect(self.add_class)
        self.label_list.currentItemChanged.connect(self.current_label_changed)

        # action menu items: File
        self.action_load_video.triggered.connect(self.load_video)
        self.action_load_images.triggered.connect(self.load_images)
        self.action_load_annot.triggered.connect(self.load_annot)
        self.action_save.triggered.connect(self.save_annot)

        # prev/next buttons
        self.button_prev.clicked.connect(self.prev)
        self.button_next.clicked.connect(self.next)

    def load_images(self) -> None:
        fname = QFileDialog.getExistingDirectory(self, 'Select Folder', str(self.last_dir))
        if len(fname):
            fname = Path(fname)
            self.dset.set_image_folder(fname)
            self.last_dir = os.path.dirname(fname)
            self.idx = 0
            self.set_hist()
    
    def load_video(self):
        fname = QFileDialog.getOpenFileName(self, "Select Video File", str(self.last_dir), "Video Files (*.mp4; *.avi)")[0]
        if len(fname) > 0:
            fname = Path(fname)
            self.dset.set_video_name(fname)
            self.last_dir = os.path.dirname(fname)
            self.idx = 0
            self.set_hist()
    
    def load_annot(self):
        pass

    def save(self) -> Path:
        # directory and filename for saving annotation data in json format
        cwd = os.getcwd()
        fname = QFileDialog.getSaveFileName(self, "Save file", str(cwd), "JSON files (*.json)")
        return Path(fname[0])

    def label_mode_segmentation(self):
        self.label_mode = "segmentation"
        self.spinBox_mask_scale.setEnabled(True)
        self.view_box.label_mode = self.label_mode

    def label_mode_bbox(self):
        self.label_mode = "bbox"
        self.spinBox_mask_scale.setEnabled(False)
        self.view_box.label_mode = self.label_mode

    def label_mode_off(self):
        self.label_mode = "off"
        self.spinBox_mask_scale.setEnabled(False)
        self.view_box.label_mode = self.label_mode

    def prev(self):
        if len(self.dset) > 0:
            # update image and annotation data
            self.idx = (self.idx - 1) % len(self.dset)
            img = self.dset[self.idx]
            self.view_box.set_image(img)

            # update histogram
            self.update_hist()

    def next(self):
        if len(self.dset) > 0:
            # update image and annotation data
            self.idx = (self.idx + 1) % len(self.dset)
            img = self.dset[self.idx]
            self.view_box.set_image(img)

            # update histogram
            self.update_hist()
    
    def update_hist(self):
        # keep old histogram levels
        hist_levels = self.hist.getLevels()

        # tie ImageItem to hist
        self.set_hist()

        # set old histogram levels
        self.hist.setLevels(*hist_levels)

    def set_hist(self):
        img = self.dset[self.idx]
        self.view_box.set_image(img)

        self.hist.setImageItem(self.view_box.img)
        self.hist.autoHistogramRange()

    def make_class_list_item(self, label: str, color: QColor) -> None:
        # New list widget item
        item = QListWidgetItem(label)
        
        # allow text edit
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        
        # set background color
        item.setBackground(color)

        # add list widget item to the label list
        self.label_list.addItem(item)

    def add_class(self) -> None:
        # color dialog for label
        color = QColorDialog().getColor()
        
        # text box for the class label
        text = self.class_label.text()
        self.view_box.label = text

    def current_label_changed(self, item: QListWidgetItem) -> None:
        # pick the correct pen
        idx = self.label_list.indexFromItem(item).row()
        self.view_box.pen = self.view_box.pens[idx]

        # update the label text
        self.view_box.label = item.text()

    def save_annot(self) -> None:
        # get directory and filename for saving annotations
        fname = self.save()

        # convert and save annotations in json format
        self.view_box.save(fname)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev()
        elif event.key() == Qt.Key_Right:
            self.next()
        elif event.key() == Qt.Key_Space:
            print("space")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())