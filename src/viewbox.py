import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QProgressDialog, QMessageBox

from dataset import Dataset
from model import Model
from annotation import Annotation
from utils import draw_segmentation_masks, mask_to_bbox

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

    def clear_qt_objects(self) -> None:
        """Clear Qt objects before loading a new image
        """
        self.clear()
        self.circle = None
        self.img = None

    def clear_data(self) -> None:
        """Clear Qt objects and data before loading a new dataset
        """
        self.clear_qt_objects()
        self.reset_current_annot()
        self.last_selected_id = None
        self.dset = Dataset()
        self.annot = Annotation()

    def auto_detect(self, annot_dict: dict[int, list], predict_mode: str, model_type: str) -> None:
        """Auto-detect masks and refine the results using SAM

        Args:
            annot_dict (dict): Annotation dictionary output from the auto-detector model
            predict_mode (str): "current" or "all" which determines whether to run the algorithm on the current frame or all the frames
            model_type (str): Name of the model e.g "canny"
        """
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

    def auto_detect_single(self, annot: list[npt.ArrayLike], model_type: str) -> None:
        """Auto-detect masks and refine the results using SAM on a single frame

        Args:
            annot (list): List of bounding box coordinates in [x1, y1, x2, y2]
            model_type (str): Name of the model e.g "canny"
        """
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
        for _ in range(len(masks)): 
            self.parent.add_mask(self.class_label)

    def query_prev_frame(self) -> None:
        """Use previous frame predictions to estimate the next frame by fine tuning with SAM
        """
        if self.idx == 0:
            err_dlg = QMessageBox(self)
            err_dlg.setWindowTitle("Error")
            err_dlg.setText("Cannot query prev detection on frame 0!")
            err_dlg.exec()
            return
        else:
            prev_masks = self.annot.get_mask(self.idx - 1)
            if prev_masks is None:
                return

            prev_labels = self.annot.get_labels(self.idx - 1)
            for i, m in enumerate(prev_masks):
                bbox = mask_to_bbox(255*m.astype(np.uint8))[0]
                mask, _, _ = self.model.predict(input_box=bbox[None, :], multimask_output=False)
                self.annot.add_annotation(self.idx, mask[0,...], prev_labels[i])
            
            self.navigate_to_idx(self.idx)

    def load_images(self, fname: str) -> None:
        """Load images given a folder name

        Args:
            fname (str): Folder name for images
        """
        if len(fname) > 0:
            fname = Path(fname)
            self.dset.set_image_folder(fname)
            self.parent.last_dir = os.path.dirname(fname)
            self.idx = 0
            self.set_image()

    def load_video(self, fname: str) -> None:
        """Load a video file

        Args:
            fname (str): Video file name
        """
        if len(fname) > 0:
            fname = Path(fname)
            self.dset.set_video_name(fname)
            self.parent.last_dir = os.path.dirname(fname)
            self.idx = 0
            self.set_image()
    
    def navigate_to_idx(self, idx: int) -> None:
        """Navigate to a given frame

        Args:
            idx (int): Frame ID
        """
        if len(self.dset) > 0:
            self.idx = idx
            labels = self.annot.get_labels(self.idx)
            if labels is not None:
                for label in labels:
                    self.parent.add_mask(label)
            self.set_image()
            self.show_mask()

    def prev(self) -> None:
        """Update current frame ID to previous
        """
        return (self.idx - 1) % len(self.dset)

    def next(self) -> None:
        """Update current frame ID to next
        """
        return (self.idx + 1) % len(self.dset)

    def get_img(self) -> npt.ArrayLike:
        """Get current image data
        """
        return self.dset[self.idx]

    def set_image(self) -> None:
        """Create and set a pg.ImageItem for the view box.
        """
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
    
    def set_show_mask_mode(self, show_mode: str) -> None:
        """Change mask viewing mode

        Args:
            show_mode (str): Mask viewing mode
        """
        self.show_mask_mode = show_mode
        self.show_mask()

    def set_label_mode(self, label_mode) -> None:
        """Change label mode

        Args:
            label_mode (str): Label mode
        """
        self.label_mode = label_mode
        self.set_image()

    def set_class_label(self, class_label: str) -> None:
        self.class_label = class_label
    
    def current_label_changed(self, class_label: str) -> None:
        self.class_label = class_label

    def show_mask(self) -> None:
        """Show mask based on mask mode
        """
        if self.show_mask_mode == "all":
            self.show_mask_all()
        elif self.last_selected_id is not None:
            self.show_mask_by_id(self.last_selected_id)
        else:
            self.set_image()

    def show_mask_by_id(self, mask_id: int) -> None:
        """Show mask with a given ID

        Args:
            mask_id (int): Mask ID in the current frame
        """
        masks = self.annot.get_mask(self.idx)
        if masks is not None and len(masks) > mask_id:
            mask = masks[mask_id,:,:]
            self.update_img_annot(mask[None,:,:])
        else:
            self.update_img_annot(None)
    
    def show_mask_all(self) -> None:
        """Show all masks in the current frame
        """
        masks = self.annot.get_mask(self.idx)
        self.update_img_annot(masks)

    def add_points(self, pos: QPoint, input_label: int) -> None:
        """Add foreground or background points to the current SAM prompt

        Args:
            pos (QPoint): Mouse click coordinate
            input_label (int): 1 for foreground and 0 for background
        """
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

    def update_annot(self) -> None:
        """Update annotation based on newly added points
        """
        masks, scores, logits = self.model.predict(input_point=self.current_points,
                                                   input_label=self.current_labels)

        mask = masks[self.mask_scale, :, :]
        self.current_masks = mask
        
        mask = mask[None,:,:]
        prev_masks = self.annot.get_mask(self.idx)
        if self.show_mask_mode=="all" and prev_masks is not None:
            mask = np.vstack((prev_masks, mask))
        self.update_img_annot(mask)

    def update_img_annot(self, mask: npt.ArrayLike) -> None:
        """Update the viewbox image with the given masks

        Args:
            mask (npt.ArrayLike): All masks in the current frame and the mask proposal
        """
        img = self.dset[self.idx]
        if mask is not None:
            img = draw_segmentation_masks(img, mask, self.alpha)
        self.clear_qt_objects()
        self.img = pg.ImageItem(img)
        self.addItem(self.img)
        self.parent.update_hist()

    def mouseClickEvent(self, event: QMouseEvent) -> None:
        """Slot for capturing mouse click events

        Args:
            event (QMouseEvent): Mouse click event
        """
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
        """Slot for capturing hover events

        Args:
            event (QMouseEvent): Mouse hover event
        """
        if self.img is not None and self.label_mode == "on":
            try:
                pos = self.calc_pos(event.pos())
            except:
                pos = QPoint(0, 0)
            if self.circle is not None:
                self.removeItem(self.circle)
            self.circle = pg.CircleROI(pos, radius=.1)
            self.addItem(self.circle)
    
    def calc_pos(self, pos: QPoint, offset: int = 1) -> QPoint:
        """Convert scene coordinate to the image coordinate

        Args:
            pos (QPoint): Scene coordinate
            offset (int, optional): Offset from mouse pointer position. Defaults to 1.

        Returns:
            QPoint: Point in the image coordinate
        """
        try:
            pos = self.mapSceneToView(pos)
            pos.setX(pos.x() + offset)
            pos.setY(pos.y() + offset)
            return pos
        except:
            return QPoint(0,0)
    
    def reset_current_annot(self) -> None:
        """Reset current annotation fields
        """
        self.current_points = None
        self.current_labels = None
        self.current_masks = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Capture key press events

        Args:
            event (QKeyEvent): Key press event
        """
        if event.key() == Qt.Key_Space:
            self.annot.add_annotation(self.idx,
                                      self.current_masks,
                                      self.class_label)
            self.reset_current_annot()
            self.parent.add_mask(self.class_label)
        
        elif event.key() == Qt.Key_Escape:
            self.reset_current_annot()
            self.set_image()

        event.ignore()