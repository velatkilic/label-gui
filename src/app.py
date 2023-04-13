import numpy as np
import sys
import os
from pathlib import Path
import json

from gui import Ui_MainWindow
from auto_detect_dialog import AutoDetectDialog
from viewbox import ViewBox

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication,
                            QMainWindow,
                            QFileDialog,
                            QListWidgetItem,
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

        # remember last directory for loading data
        self.last_dir = Path(os.getcwd())

        # detection
        self.button_auto_detect.clicked.connect(self.auto_detect)

        # label mode
        self.radio_annot_on.clicked.connect(self.label_mode_on)
        self.radio_annot_off.clicked.connect(self.label_mode_off)
        self.spinBox_mask_scale.valueChanged.connect(self.mask_scale)
        self.annot_list.currentItemChanged.connect(self.current_annot_changed)
        self.radio_mask_all.clicked.connect(self.show_mask_all)
        self.radio_mask_last.clicked.connect(self.show_mask_last)

        # view_box holds images
        self.view_box = ViewBox(self, lockAspect=True, invertY=True)
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

        # prev/next
        self.spinBox_frame_id.valueChanged.connect(self.navigate_to_idx)
        self.button_prev.clicked.connect(self.prev)
        self.button_next.clicked.connect(self.next)

        self.label_mode_off() # start with default off
        self.show_mask_last() # default show the last mask

    def auto_detect(self):
        dialog = AutoDetectDialog(self.view_box.dset, self.view_box.idx, self.view_box.model.sam)
        dialog.exec()
        self.view_box.auto_detect(dialog.output, dialog.predict_mode, dialog.model_type)
    
    def update_frame_id(self):
        frame_count = len(self.view_box.dset)
        self.label_frame_count.setText("/"+str(frame_count - 1))
        self.spinBox_frame_id.setMaximum(frame_count)

    def load_images(self) -> None:
        fname = QFileDialog.getExistingDirectory(self, 'Select Folder', str(self.last_dir))
        self.last_dir = os.path.dirname(fname)
        self.view_box.load_images(fname)
        self.update_frame_id()
        self.set_hist()
    
    def load_video(self):
        fname = QFileDialog.getOpenFileName(self, "Select Video File", str(self.last_dir), "Video Files (*.mp4; *.avi)")[0]
        self.last_dir = os.path.dirname(fname)
        self.view_box.load_video(fname)
        self.update_frame_id()
        self.set_hist()
    
    def load_annot(self):
        fname = QFileDialog.getOpenFileName(self, "Select Annotation File", str(self.last_dir), "JSON Files (*.json)")[0]
        # TODO

    def save_annot(self) -> Path:
        cwd = os.getcwd()
        fname = QFileDialog.getSaveFileName(self, "Save file", str(cwd), "JSON files (*.json)")
        fname =  Path(fname[0])
        # TODO

    def show_mask_all(self):
        self.view_box.set_show_mask_mode("all")

    def show_mask_last(self):
        self.view_box.set_show_mask_mode("last")

    def label_mode_on(self):
        self.spinBox_mask_scale.setEnabled(True)
        self.view_box.set_label_mode("on")

    def label_mode_off(self):
        self.spinBox_mask_scale.setEnabled(False)
        self.view_box.set_label_mode("off")

    def navigate_to_idx(self, idx):
        self.view_box.navigate_to_idx(idx)
        self.update_hist()
        self.annot_list.clear()
        self.view_box.show_mask()

    def prev(self):
        idx = self.view_box.prev()
        self.spinBox_frame_id.setValue(idx)

    def next(self):
        idx = self.view_box.next()
        self.spinBox_frame_id.setValue(idx)
    
    def update_hist(self):
        hist_levels = self.hist.getLevels()
        self.set_hist()
        self.hist.setLevels(*hist_levels)

    def set_hist(self):
        self.hist.setImageItem(self.view_box.img)
        self.hist.autoHistogramRange()

    def mask_scale(self, mask_scale):
        self.view_box.mask_scale = mask_scale

    def add_mask(self, idx):
        text = "Mask " + str(idx)
        item = QListWidgetItem(text)
        self.annot_list.addItem(item)
        self.view_box.last_selected_id = idx - 1

    def add_class(self) -> None:
        text = self.class_label.text()
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsEditable)

        # add list widget item to the label list
        self.label_list.addItem(item)

        self.view_box.set_class_label(text)

    def current_annot_changed(self, item: QListWidgetItem):
        mask_text = item.text()
        idx = int(mask_text.split()[-1]) - 1
        if self.view_box.show_mask_mode == "last":
            self.view_box.show_mask_by_id(idx)

    def current_label_changed(self, item: QListWidgetItem) -> None:
        self.view_box.current_label_changed(item.text())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev()
        elif event.key() == Qt.Key_Right:
            self.next()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())