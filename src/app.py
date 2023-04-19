"""Main Application
"""
import sys
import os
from pathlib import Path

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import (QApplication,
                             QMainWindow,
                             QFileDialog,
                             QListWidgetItem,
                             QProgressDialog,
                             QMessageBox
                             )
import pyqtgraph as pg

from gui import Ui_MainWindow
from auto_detect_dialog import AutoDetectDialog
from viewbox import ViewBox


pg.setConfigOption('imageAxisOrder', 'row-major')  # best performance


class MainWindow(QMainWindow, Ui_MainWindow):
    """Mainwindow for the PyQt5 Application
    """

    def __init__(self) -> None:
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
        self.img_view.addItem(self.view_box, row=0,
                              col=0, rowspan=1, colspan=1)
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

        self.label_mode_off()  # start with default off
        self.show_mask_last()  # default show the last mask

    def auto_detect(self) -> None:
        """Auto-detect slot
        Presents user with an auto-detect window
        Estimates segmentation masks based on the selected model and its parameters
        """
        dialog = AutoDetectDialog(
            self.view_box.dset, self.view_box.idx, self.view_box.model.sam)
        dialog.exec()
        self.view_box.auto_detect(
            dialog.output, dialog.predict_mode, dialog.model_type)

    def query_prev_frame(self) -> None:
        """Uses previous frame detections to predict the next frame detections
        This option only makes sense for video data where frames are temporally correlated
        """
        self.radio_annot_on.click()
        self.view_box.query_prev_frame()
        self.spinBox_frame_id.setValue(self.view_box.idx)

    def update_frame_id(self) -> None:
        """Set the text for the frame id spinbox based on the loaded data size
        """
        frame_count = len(self.view_box.dset)
        self.label_frame_count.setText("/" + str(frame_count - 1))
        self.spinBox_frame_id.setMaximum(frame_count)

    def load_images(self) -> None:
        """Load image folder
        User is presented with a dialog to select an image folder (only tiff is tested)
        Annotation list, view box data are cleared and a new dataset is constructed
        Frame id spinbox and the histogram data are updated
        """
        fname = QFileDialog.getExistingDirectory(
            self, 'Select Folder', str(self.last_dir))
        if fname is not None and len(fname) > 0:
            self.annot_list.clear()
            self.view_box.clear_data()
            self.last_dir = os.path.dirname(fname)
            self.view_box.load_images(fname)
            self.update_frame_id()
            self.set_hist()

    def load_video(self) -> None:
        """Load video file
        User is presented with a dialog to select a folder
        Annotation list, view box data are cleared and a new dataset is constructed
        Frame id spinbox and the histogram data are updated
        """
        fname = QFileDialog.getOpenFileName(self, "Select Video File", str(
            self.last_dir), "Video Files (*.mp4; *.avi)")[0]
        if fname is not None and len(fname) > 0:
            self.annot_list.clear()
            self.view_box.clear_data()
            self.last_dir = os.path.dirname(fname)
            self.view_box.load_video(fname)
            self.update_frame_id()
            self.set_hist()

    def err_msg_load_dset_first(self) -> None:
        """Presents user with an error message saying image or video data
        should be opened first.
        """
        err_dlg = QMessageBox(self)
        err_dlg.setWindowTitle("Error")
        err_dlg.setText("Load an image folder or a video file first!")
        err_dlg.exec()

    def load_annot(self) -> None:
        """Load annotation data
        """
        if len(self.view_box.dset) == 0:
            self.err_msg_load_dset_first()
            return
        fname = QFileDialog.getOpenFileName(
            self, "Select Annotation File", str(self.last_dir), "JSON Files (*.json)")[0]
        if fname is not None and len(fname) > 0:
            self.view_box.annot.load_annot_from_file(Path(fname))
            self.navigate_to_idx(self.view_box.idx)

            unique_labels = set()
            for _, labels in self.view_box.annot.labels.items():
                for label in labels:
                    if not (label in unique_labels):
                        self.class_label.setText(label)
                        self.add_class()
                        unique_labels.add(label)

    def load_embeddings(self) -> None:
        """Load image embeddings for SAM
        """
        if len(self.view_box.dset) == 0:
            self.err_msg_load_dset_first()
            return

        fname = QFileDialog.getOpenFileName(self, "Select Embedding File", str(
            self.last_dir), "Pytorch Files (*.pth; *.pt)")[0]
        if fname is not None and len(fname) > 0:
            self.view_box.annot.load_embed_from_torch(Path(fname))
            self.radio_annot_on.click()

    def save_annot(self) -> None:
        """Save segmentation annotations to a JSON file
        """
        if len(self.view_box.dset) == 0:
            self.err_msg_load_dset_first()
            return

        fname = QFileDialog.getSaveFileName(
            self, "Save file", str(self.last_dir), "JSON Files (*.json)")[0]
        if fname is not None and len(fname) > 0:
            self.view_box.annot.save_annotations(Path(fname))

    def save_embeddings(self) -> None:
        """Save SAM image embeddings for later use
        """
        if len(self.view_box.dset) == 0:
            self.err_msg_load_dset_first()
            return

        fname = QFileDialog.getSaveFileName(self, "Save file", str(
            self.last_dir), "Pytorch Files (*.pth; *.pt)")[0]
        if fname is not None and len(fname) > 0:
            self.view_box.annot.save_embed(Path(fname))

    def show_mask_all(self) -> None:
        """Set mask mode of the view box to all which shows all the masks in the current frame
        """
        self.view_box.set_show_mask_mode("all")

    def show_mask_last(self) -> None:
        """Set mask mode of the view box to last selected/created mask
        """
        self.view_box.set_show_mask_mode("last")

    def label_mode_on(self) -> None:
        """Set label mode to on for the view box and enable the mask scale spin box
        """
        self.spinBox_mask_scale.setEnabled(True)
        self.view_box.set_label_mode("on")

    def label_mode_off(self) -> None:
        """Set label mode to off for the view box and disable the mask scale spin box
        """
        self.spinBox_mask_scale.setEnabled(False)
        self.view_box.set_label_mode("off")

    def navigate_to_idx(self, idx) -> None:
        """Load frame number idx

        Args:
            idx (int): Frame id to be shown
        """
        self.annot_list.clear()
        self.view_box.navigate_to_idx(idx)
        self.update_hist()
        self.view_box.show_mask()

    def compute_embeddings(self) -> None:
        """Compute image embeddings for the entire video or image folder
        """
        nframes = len(self.view_box.dset)
        if nframes > 0:
            self.radio_annot_on.click()
            pb = QProgressDialog(
                "Pre-computing image embeddings ...", "Cancel", 0, nframes)
            pb.setWindowModality(Qt.WindowModal)
            for i in range(nframes):
                self.next()

                if pb.wasCanceled():
                    break
                pb.setValue(i)

    def prev(self) -> None:
        """Navigate to the previous frame
        """
        idx = self.view_box.prev()
        self.spinBox_frame_id.setValue(idx)

    def next(self) -> None:
        """Navigate to the next frame
        """
        idx = self.view_box.next()
        self.spinBox_frame_id.setValue(idx)

    def update_hist(self) -> None:
        """Update the histogram while keeping the previos histogram levels
        """
        hist_levels = self.hist.getLevels()
        self.set_hist()
        self.hist.setLevels(*hist_levels)

    def set_hist(self) -> None:
        """Auto-set histogram range based on the current image data
        """
        self.hist.setImageItem(self.view_box.img)
        self.hist.autoHistogramRange()

    def mask_scale(self, mask_scale) -> None:
        """Set the mask scale variable for the view box

        Args:
            mask_scale (int): 0-fine masks, 1-medium masks, 2-large masks
        """
        self.view_box.mask_scale = mask_scale

    def add_mask(self, class_label) -> None:
        """Add a segmentation mask to the mask list

        Args:
            class_label (str): Class label of the segmentation mask to be created
        """
        item = QListWidgetItem(class_label)
        self.annot_list.addItem(item)
        self.view_box.last_selected_id = self.annot_list.currentRow() - 1

    def delete_mask(self) -> None:
        """Delete a segmentation mask from the list and show the next mask
        """
        annot_id = self.annot_list.currentRow()
        self.view_box.annot.delete_mask(self.view_box.idx, annot_id)
        if annot_id == self.annot_list.count() - 1 and annot_id != 0:
            self.view_box.last_selected_id = annot_id - 1
        else:
            self.view_box.last_selected_id = annot_id
        self.annot_list.takeItem(annot_id)
        self.view_box.show_mask()

    def add_class(self) -> None:
        """Add a new class item using the name in the text box
        """
        text = self.class_label.text()
        text = "_".join(text.split())  # remove spaces
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsEditable)

        # add list widget item to the label list
        self.label_list.addItem(item)

        self.view_box.set_class_label(text)

    def current_annot_changed(self, item: QListWidgetItem) -> None:
        """Show selected mask

        Args:
            item (QListWidgetItem): Selected mask list item
        """
        if item is None:
            return
        idx = self.annot_list.currentRow()
        if self.view_box.show_mask_mode == "last":
            self.view_box.show_mask_by_id(idx)

    def change_annot_class_label(self) -> None:
        """Assign a new class to a segmentation mask
        """
        # update listview
        annot_item = self.annot_list.currentItem()
        annot_item.setText(self.view_box.class_label)
        label_id = self.annot_list.currentRow()
        # update annotation
        self.view_box.annot.set_label(
            self.view_box.idx, label_id, self.view_box.class_label)

    def current_label_changed(self, item: QListWidgetItem) -> None:
        self.view_box.current_label_changed(item.text())

    def keyPressEvent(self, event: QEvent) -> None:
        """Keyboard shortcuts

        Args:
            event (QEvent): A keyboard press event
        """
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
