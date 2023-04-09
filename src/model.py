import numpy as np
import os
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

class Model:
    def __init__(self, sam_checkpoint = None, model_type="vit_h", device="cuda"):
        if sam_checkpoint is None:
            sam_checkpoint = os.path.join(os.getcwd(), "models" ,"sam_vit_h_4b8939.pth")
        
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)

        self.img_shape = None
    
    def set_image(self, img):
        self.img_shape = img.shape
        self.predictor.set_image(img)

    def predict(self, input_point=None, input_label=None, input_mask=None, input_box=None):
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=input_mask
        )
        return masks, scores, logits

class Annotation:
    def __init__(self) -> None:
        self.masks = {}
        self.mask_scale = {}
        self.scores = {}
        self.logits = {}
        self.labels = {}
        self.colors = {}
        self.imgs = {}
    
    def add_annotation(self, frame_id, mask, mask_scale, score, logit, color, label):
        if frame_id in self.masks:
            self.masks[frame_id].append(mask)
            self.mask_scale[frame_id].append(mask_scale)
            self.scores[frame_id].append(score)
            self.logits[frame_id].append(logit)
            self.colors[frame_id].append(color)
            self.labels[frame_id].append(label)
        else:
            self.masks[frame_id] = [mask]
            self.mask_scale[frame_id] = [mask_scale]
            self.scores[frame_id] = [score]
            self.logits[frame_id] = [logit]
            self.colors[frame_id] = [color]
            self.labels[frame_id] = [label]

    def get_mask(self, frame_id):
        if frame_id in self.masks:
            mask_scale = self.mask_scale[frame_id]
            masks = np.array(self.masks[frame_id])
            return masks[:,mask_scale,:,:]
        else:
            return None

    def get_color(self, frame_id):
        if frame_id in self.colors:
            return self.colors[frame_id]
        else:
            return None

    def get_image(self, frame_id):
        if frame_id in self.imgs:
            return self.imgs[frame_id]
        else:
            return None

    def set_image(self, frame_id, img):
        self.imgs[frame_id] = img