import numpy as np
import torch
import cv2 as cv
import os
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

class Model:
    def __init__(self, sam_checkpoint = None, model_type=None, device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if sam_checkpoint is None:
            sam_checkpoint = os.path.join(os.getcwd(), "models" ,"sam_vit_h_4b8939.pth")
        
        if model_type is None:
            model_type = "vit_h"
        
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
    
    def set_cached_img_embed(self, original_size=None, input_size=None, features=None):
        self.predictor.reset_image()
        self.predictor.original_size = original_size
        self.predictor.input_size = input_size
        self.predictor.features = features
        self.predictor.is_image_set = True
    
    def get_img_embed(self):
        img_embed = {}
        img_embed["original_size"] = self.predictor.original_size
        img_embed["input_size"] = self.predictor.input_size
        img_embed["features"] = self.predictor.features
        return img_embed

    def set_image(self, img):
        self.predictor.set_image(img)

    def predict(self, input_point=None, input_label=None, input_mask=None, input_box=None, multimask_output=True):
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=input_mask,
            box=input_box,
            multimask_output=multimask_output
        )
        return masks, scores, logits

class Canny:
    def __init__(self, dataset, th1=50, th2=100, min_area=20, it_closing=1):
        self.dset = dataset
        self.th1 = th1
        self.th2 = th2
        self.min_area = min_area
        self.it_closing = it_closing

    def predict(self, frame_id):
        # convert to grayscale
        gray = cv.cvtColor(self.dset[frame_id], cv.COLOR_BGR2GRAY)

        # Canny edge detection
        edge = cv.Canny(gray, self.th1, self.th2)

        # Morphological transformation: closing
        kernel = np.ones((8, 8), dtype=np.uint8)
        closing = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel, iterations=self.it_closing)

        # Contours
        contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Bounding rectangle
        bbox = []
        for temp in contours:
            area = cv.contourArea(temp)
            if area > self.min_area:
                x, y, w, h = cv.boundingRect(temp)
                bbox.append(np.array([x, y, x + w, y + h]))

        return bbox


class MOG2:
    def __init__(self, dataset, history=100, varThreshold=40, it_closing=1, min_area=20):
        """

        Attributes:
            history      : int        Length of history 
            varThreshold : int        Threshold of pixel background identification in MOG2 of cv.
            it_closing   : int        Parameter of cv.morphologyEx
            minArea      : int        Minimal area of bbox to be considered as a valid particle.
        """
        self.dset = dataset
        self.it_closing = it_closing
        self.min_area = min_area
        self.mog2 = cv.createBackgroundSubtractorMOG2(history=history,
                                                     varThreshold=varThreshold)
        self.__train()

    def predict(self, idx):
        gray = cv.cvtColor(self.dset[idx], cv.COLOR_BGR2GRAY)
        mask = self.mog2.apply(gray)
        # Morphological transformation: closing
        kernel = np.ones((8, 8), dtype=np.uint8)
        closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=self.it_closing)

        # Contours
        contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Bounding rectangle
        bbox = []
        for temp in contours:
            area = cv.contourArea(temp)
            if area > self.min_area:
                x, y, w, h = cv.boundingRect(temp)
                bbox.append(np.array([x, y, x + w, y + h]))

        return bbox

    def __train(self):
        for img in self.dset:
            if img is None: break
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            self.mog2.apply(gray)