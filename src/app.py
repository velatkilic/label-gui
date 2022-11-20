import sys
import os
from pathlib import Path
import json

from gui import Ui_MainWindow
from viewbox import ViewBox
from roi import annot_to_label

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

        # view_box holds images and ROIs
        self.view_box = ViewBox(lockAspect=True, invertY=True)
        self.hist = pg.HistogramLUTItem()

        self.img_view.addItem(self.view_box, row=0, col=0, rowspan=1, colspan=1)
        self.img_view.addItem(self.hist, row=0, col=1, rowspan=1, colspan=1)
        
        # class labels
        self.button_add_class_label.clicked.connect(self.add_class)
        self.label_list.currentItemChanged.connect(self.current_label_changed)

        # action menu items: File
        self.action_load_images.triggered.connect(self.load_images)
        self.action_load_annot.triggered.connect(self.load_annot)
        self.action_save.triggered.connect(self.save_annot)

        # prev/next buttons
        self.button_prev.clicked.connect(self.prev)
        self.button_next.clicked.connect(self.next)

    def prev(self):
        # update image and annotation data
        self.view_box.prev()

        # update histogram
        self.update_hist()

    def next(self):
        # update image and annotation data
        self.view_box.next()

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
        # tie new ImateItem
        img = self.view_box.get_image()
        self.hist.setImageItem(img)

        # set auto range on
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
        # color dialog for picking pen color
        color = QColorDialog().getColor()
        
        # text box for the class label
        text = self.class_label.text()
        self.view_box.label = text

        # create and assign pen
        self.view_box.pen = pg.mkPen(width=1, color=color)
        self.view_box.pens.append(self.view_box.pen)

        # create list widget item using the selected color
        self.make_class_list_item(text, color)

    def current_label_changed(self, item: QListWidgetItem) -> None:
        # pick the correct pen
        idx = self.label_list.indexFromItem(item).row()
        self.view_box.pen = self.view_box.pens[idx]

        # update the label text
        self.view_box.label = item.text()

    def load(self) -> Path:
        # get directory for loading content from a folder
        fname = QFileDialog.getExistingDirectory(self, 'Select Folder')
        return Path(fname)

    def save(self) -> Path:
        # directory and filename for saving annotation data in json format
        cwd = os.getcwd()
        fname = QFileDialog.getSaveFileName(self, "Save file", cwd, "JSON files (*.json)")
        return Path(fname[0])

    def load_images(self) -> None:
        # get file directory
        fname = self.load()

        if fname is not None:
            # set image from view_box
            self.view_box.load_images(fname)

            # init histogram
            self.set_hist()
    
    def load_annot(self) -> None:
        # get filename for annotation
        cwd = os.getcwd()
        fname = QFileDialog.getOpenFileName(self, "Open file", cwd, "JSON files (*.json)")
        fname = Path(fname[0])

        # load annotation data
        with open(fname, "r") as file:
            data = json.load(file)

        # create and set rois in viewbox
        self.view_box.load_annot(data)

        # update class label list
        label_dict = annot_to_label(data)
        self.update_label_list(label_dict)

    def update_label_list(self, label_dict: dict) -> None:
        
        for label, rgb in label_dict.items():
            
            # make color object
            color = pg.mkColor(rgb)
            
            # create and assign pen
            self.view_box.pen = pg.mkPen(width=1, color=color)
            self.view_box.pens.append(self.view_box.pen)
            
            # create list widget item using the selected color
            self.make_class_list_item(label, color)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())