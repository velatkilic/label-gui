from pathlib import Path
import json

from dataset import dataset_from_filename
from roi import RectROI, rois_to_annot, annot_to_rois

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent

class ViewBox(pg.ViewBox):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        # image and annot index
        self.idx = 0
        self.length = 0

        # image data
        self.imgs = {}

        # bounding box data
        self.pen = pg.mkPen(width=1, color='r')
        self.pens = []
        self.label = None
        self.drawing = False
        self.start = None
        self.end = None
        self.prev_roi = None
        self.rois = {}
    
    def load_images(self, fname: Path) -> None:
        # get a dataset object from a video or image collection
        # TODO: handle multiple videos under the same folder
        self.reader_dataset = dataset_from_filename(fname)

        # set first image
        if self.reader_dataset is not None:
            self.length = self.reader_dataset.length()
            self.set_image()

    def load_annot(self, annot: list[dict]) -> None:
        # convert annotation data to rois
        self.rois = annot_to_rois(annot, self.remove_roi)

        # add rois to viewbox
        self.set_rois()

    def save(self, fname: Path) -> None:
        # convert rois to data dict
        data = rois_to_annot(self.rois)

        # save data dict as json
        with open(fname, "w") as file:
            json.dump(data, file)

    def prev(self) -> None:
        # calculate prev index
        self.idx = (self.idx - 1) % self.length

        # update contents (image, ROIs, masks)
        self.update_contents()

    def next(self) -> None:
        # calculate next index
        self.idx = (self.idx + 1) % self.length

        # update contents (image, ROIs, masks)
        self.update_contents()
    
    def get_image(self) -> pg.ImageItem:
        # get current image if not already buffered
        if not(self.idx in self.imgs):
            img = self.reader_dataset.get_img(self.idx)
            img = pg.ImageItem(img)
            self.imgs[self.idx] = img
        return self.imgs[self.idx]

    def set_image(self) -> None:
        # clear current contents
        self.clear()
        
        # get ImageItem object
        img = self.get_image()

        # show current image
        self.addItem(img)

    def set_rois(self) -> None:
        # add rois to the viewbox
        if self.idx in self.rois:
            for roi in self.rois[self.idx]:
                self.addItem(roi)
    
    def update_contents(self) -> None:
        # clear and then get image at current index
        self.set_image()

        # get rois from current index
        self.set_rois()

    def make_roi(self, start: QPoint, end: QPoint) -> RectROI:
        # width and height
        w = end.x() - start.x()
        h = end.y() - start.y()

        # make roi 
        roi = RectROI(self.label, start, [w, h], pen=self.pen,
                    removable=True, rotatable=False)    
        self.addItem(roi)

        # removing roi from context menu
        roi.sigRemoveRequested.connect(self.remove_roi)
        
        return roi
    
    def add_roi_to_list(self, roi: RectROI) -> None:
        # if dict entry exists, append to the list
        if self.idx in self.rois:
            self.rois[self.idx].append(roi)
        # if not, make a new list
        else:
            self.rois[self.idx] = [roi]
    
    def remove_roi(self, roi: RectROI) -> None:
        # find roi in the list
        idx = self.rois[self.idx].index(roi)

        # remove from the list
        del self.rois[self.idx][idx]

        # remove from the viewbox
        self.removeItem(roi)

    def mouseClickEvent(self, event: QMouseEvent) -> None:
        # left button for drawing bounding boxes
        if event.button() == Qt.LeftButton:

            # map coordinates
            # TODO: coordinates are off
            pos = self.mapSceneToView(event.pos())
            
            # draw roi
            if self.drawing:
                self.drawing = False
                self.end = pos
                roi = self.make_roi(self.start, self.end)
                self.add_roi_to_list(roi)
                self.removeItem(self.prev_roi)
                self.prev_roi = None
            else:
                self.drawing = True
                self.start = pos

    def hoverEvent(self, event):
        # show bounding box while drawing is on
        if self.drawing:
            # delete previous bounding box
            if self.prev_roi is not None:
                self.removeItem(self.prev_roi)
            
            # get current mouse position
            pos = self.mapSceneToView(event.pos())

            # visual feedback
            # set current as end and draw a bounding box
            roi = self.make_roi(self.start, pos)
            self.prev_roi = roi