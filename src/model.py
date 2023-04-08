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