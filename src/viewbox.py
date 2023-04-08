import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent

class ViewBox(pg.ViewBox):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        self.img = None
        self.label_mode = "segmentation"

        # bounding box data
        self.pen = pg.mkPen(width=1, color='r')
        self.drawing = False
        self.start = None
        self.end = None
        self.prev_roi = None

    def set_image(self, img) -> None:
        self.img = pg.ImageItem(img)
        self.clear() # clear current contents
        self.addItem(self.img) # show current image

    def make_roi(self, start: QPoint, end: QPoint):
        # width and height
        w = end.x() - start.x()
        h = end.y() - start.y()

        # make roi 
        roi = pg.RectROI(start, [w, h], pen=self.pen, removable=True, rotatable=False)    
        self.addItem(roi)

        # removing roi from context menu
        roi.sigRemoveRequested.connect(self.removeItem)
        
        return roi

    def mouseClickEvent(self, event: QMouseEvent) -> None:
        pos = self.mapSceneToView(event.pos())

        if event.button() == Qt.LeftButton:
            # draw roi
            if self.label_mode == "bbox":
                if self.drawing:
                    self.drawing = False
                    self.end = pos
                    roi = self.make_roi(self.start, self.end)
                    self.removeItem(self.prev_roi)
                    self.prev_roi = None
                else:
                    self.drawing = True
                    self.start = pos
            elif self.label_mode == "segmentation":
                pass

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