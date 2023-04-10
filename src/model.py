import numpy as np
import os
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

class Model:
    def __init__(self, sam_checkpoint = None, model_type="vit_h", device="cuda"):
        if sam_checkpoint is None:
            sam_checkpoint = os.path.join(os.getcwd(), "models" ,"sam_vit_h_4b8939.pth")
        
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.auto_mask_generator = None

        self.img_shape = None
    
    def set_image(self, img):
        self.img_shape = img.shape
        self.predictor.set_image(img)

    def set_auto_mask_generator(self, params):
        self.auto_mask_generator = SamAutomaticMaskGenerator(model=self.sam, **params)

    def predict(self, input_point=None, input_label=None, input_mask=None, input_box=None):
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=input_mask
        )
        return masks, scores, logits

    def generate(self, img):
        return self.auto_mask_generator.generate(img)

class Annotation:
    def __init__(self) -> None:
        self.masks = {}
        self.labels = {}
        self.imgs = {}
    
    def add_annotation(self, frame_id, mask, label):
        if mask is None:
            return
        elif frame_id in self.masks:
            self.masks[frame_id].append(mask)
            self.labels[frame_id].append(label)
        else:
            self.masks[frame_id] = [mask]
            self.labels[frame_id] = [label]

    def add_auto_detect_annot(self, frame_id, annots):
        masks = []
        label = ["unspecified",] * len(annots)
        for annot in annots:
            masks.append(annot["segmentation"])
        
        self.masks[frame_id] = masks
        self.labels[frame_id] = label

    def get_mask(self, frame_id):
        if frame_id in self.masks:
            return np.array(self.masks[frame_id])
        else:
            return None

    def get_image(self, frame_id):
        if frame_id in self.imgs:
            return self.imgs[frame_id]
        else:
            return None

    def set_image(self, frame_id, img):
        self.imgs[frame_id] = img