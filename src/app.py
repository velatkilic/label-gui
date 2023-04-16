import numpy as np
import sys
import os
from pathlib import Path

from gui import Ui_MainWindow
from auto_detect_dialog import AutoDetectDialog
from viewbox import ViewBox

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication,
                            QMainWindow,
                            QFileDialog,
                            QListWidgetItem,
                            QProgressDialog,
                            QMessageBox
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
        self.button_embedding.clicked.connect(self.compute_embeddings)
        self.button_query_prev.clicked.connect(self.query_prev_frame)

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
        self.action_load_embeddings.triggered.connect(self.load_embeddings)
        self.action_save_annot.triggered.connect(self.save_annot)
        self.action_save_embeddings.triggered.connect(self.save_embeddings)

        # prev/next
        self.spinBox_frame_id.valueChanged.connect(self.navigate_to_idx)
        self.button_prev.clicked.connect(self.prev)
        self.button_next.clicked.connect(self.next)

        self.label_mode_off() # start with default off
        self.show_mask_last() # default show the last mask

        self.current_annot_idx = 0

    def auto_detect(self):
        dialog = AutoDetectDialog(self.view_box.dset, self.view_box.idx, self.view_box.model.sam)
        dialog.exec()
        self.view_box.auto_detect(dialog.output, dialog.predict_mode, dialog.model_type)
    
    def query_prev_frame(self):
        self.radio_annot_on.click()
        self.view_box.query_prev_frame()
        self.spinBox_frame_id.setValue(self.view_box.idx)

    def update_frame_id(self):
        frame_count = len(self.view_box.dset)
        self.label_frame_count.setText("/"+str(frame_count - 1))
        self.spinBox_frame_id.setMaximum(frame_count)

    def load_images(self) -> None:
        fname = QFileDialog.getExistingDirectory(self, 'Select Folder', str(self.last_dir))
        if fname is not None and len(fname) > 0:
            self.annot_list.clear()
            self.view_box.clear_data()
            self.last_dir = os.path.dirname(fname)
            self.view_box.load_images(fname)
            self.update_frame_id()
            self.set_hist()
    
    def load_video(self):
        fname = QFileDialog.getOpenFileName(self, "Select Video File", str(self.last_dir), "Video Files (*.mp4; *.avi)")[0]
        if fname is not None and len(fname) > 0:
            self.annot_list.clear()
            self.view_box.clear_data()
            self.last_dir = os.path.dirname(fname)
            self.view_box.load_video(fname)
            self.update_frame_id()
            self.set_hist()
    
    def err_msg_load_dset_first(self):
        err_dlg = QMessageBox(self)
        err_dlg.setWindowTitle("Error")
        err_dlg.setText("Load an image folder or a video file first!")
        err_dlg.exec()

    def load_annot(self):
        if len(self.view_box.dset) == 0:
            self.err_msg_load_dset_first()
            return
        fname = QFileDialog.getOpenFileName(self, "Select Annotation File", str(self.last_dir), "JSON Files (*.json)")[0]
        if fname is not None and len(fname) > 0:
            self.view_box.annot.load_annot_from_file(Path(fname))
            self.navigate_to_idx(self.view_box.idx)

            unique_labels = set()
            for frame_id, labels in self.view_box.annot.labels.items():
                for label in labels:
                    if not(label in unique_labels):
                        self.class_label.setText(label)
                        self.add_class()
                        unique_labels.add(label)
    
    def load_embeddings(self):
        if len(self.view_box.dset) == 0:
            self.err_msg_load_dset_first()
            return
        
        fname = QFileDialog.getOpenFileName(self, "Select Embedding File", str(self.last_dir), "Pytorch Files (*.pth; *.pt)")[0]
        if fname is not None and len(fname) > 0:
            self.view_box.annot.load_embed_from_torch(Path(fname))
            self.radio_annot_on.click()

    def save_annot(self) -> Path:
        if len(self.view_box.dset) == 0:
            self.err_msg_load_dset_first()
            return
        
        fname = QFileDialog.getSaveFileName(self, "Save file", str(self.last_dir), "JSON Files (*.json)")[0]
        if fname is not None and len(fname) > 0:
            self.view_box.annot.save_annotations(Path(fname))
    
    def save_embeddings(self):
        if len(self.view_box.dset) == 0:
            self.err_msg_load_dset_first()
            return
        
        fname = QFileDialog.getSaveFileName(self, "Save file", str(self.last_dir), "Pytorch Files (*.pth; *.pt)")[0]
        if fname is not None and len(fname) > 0:
            self.view_box.annot.save_embed(Path(fname))

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
        self.annot_list.clear()
        self.view_box.navigate_to_idx(idx)
        self.update_hist()
        self.view_box.show_mask()
    
    def compute_embeddings(self):
        nframes = len(self.view_box.dset)
        if nframes > 0:
            self.radio_annot_on.click()
            pb = QProgressDialog("Pre-computing image embeddings ...", "Cancel", 0, nframes)
            pb.setWindowModality(Qt.WindowModal)
            for i in range(nframes):
                self.next()

                if pb.wasCanceled():
                    break
                pb.setValue(i)

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

    def add_mask(self, idx, class_label):
        text = str(idx) + " - " + class_label
        item = QListWidgetItem(text)
        self.annot_list.addItem(item)
        self.view_box.last_selected_id = idx - 1
    
    def delete_mask(self):
        self.view_box.annot.delete_mask(self.view_box.idx, self.current_annot_idx)
        if self.current_annot_idx == 0:
            new_id = 0
        else:
            new_id = self.current_annot_idx - 1
        self.view_box.last_selected_id = new_id
        if self.view_box.show_mask_mode == "last":
            self.view_box.show_mask_by_id(new_id)
        else:
            self.view_box.show_mask_all()
        
        self.annot_list.takeItem(self.annot_list.count()-1)

    def add_class(self) -> None:
        text = self.class_label.text()
        text = "_".join(text.split()) # remove spaces
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsEditable)

        # add list widget item to the label list
        self.label_list.addItem(item)

        self.view_box.set_class_label(text)

    def current_annot_changed(self, item):
        if item is None: return

        mask_text = item.text()
        idx = int(mask_text.split()[0]) - 1
        self.current_annot_idx = idx

        if self.view_box.show_mask_mode == "last":
            self.view_box.show_mask_by_id(idx)

    def change_annot_class_label(self):
        # update listview
        annot_item = self.annot_list.currentItem()
        text = annot_item.text().split()
        label_id = int(text[0]) - 1
        text[-1] = self.view_box.class_label
        text = " ".join(text)
        annot_item.setText(text)

        # update annotation
        self.view_box.annot.set_label(self.view_box.idx, label_id, self.view_box.class_label)


    def current_label_changed(self, item: QListWidgetItem) -> None:
        self.view_box.current_label_changed(item.text())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev()
        elif event.key() == Qt.Key_Right:
            self.next()
        elif event.key() == Qt.Key_Delete:
            self.delete_mask()
        elif event.key() == Qt.Key_Shift:
            self.change_annot_class_label()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())