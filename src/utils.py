import numpy as np
import colorsys
from pycocotools import mask as coco_mask
import cv2 as cv

def normalize(x):
    x = x.astype(np.float32)
    mx = x.max()
    mn = x.min()
    df = mx - mn
    if df != 0:
        x = (x - mn) / df
    return x

def hsv_to_rgb(hsv):
    h, s, v = hsv
    h /= 360.
    s /= 255.
    v /= 255.
    rgb = colorsys.hsv_to_rgb(h, s, v)
    rgb = np.array(list(map(lambda x: x*255, rgb)))
    return rgb

def draw_segmentation_masks(img, masks, alpha, colors=None):
    img = 255. * normalize(img) # in case img has more than 8 bits
    n_masks = len(masks)
    if n_masks == 0: return img
    if colors is None:
        h = np.random.randint(0, 359, size=(n_masks, 1))
        s = 255.*np.ones((n_masks,1)) #np.random.randint(100, 255, size=(n_masks, 1))
        v = img.mean() * np.ones((n_masks, 1))
        colors = np.hstack((h,s,v))
    
    mask = np.zeros(img.shape)
    for i, m in enumerate(masks):
        m = normalize(m)
        m = m[:,:,None] * hsv_to_rgb(colors[i])
        mask += m
    mask = np.clip(mask, 0, 255)
    mask_bool = np.bitwise_or.reduce(masks, axis=0)
    img[mask_bool] = img[mask_bool]*alpha + mask[mask_bool]*(1-alpha)
                                                 
    return img

def make_new_color(seed_color, a=20, b=5):
    h, s, v = seed_color
    delta = np.maximum(np.random.randint(a), b)
    h += delta * np.sign(np.random.rand() - 0.5)
    h = np.clip(h, 0, 259)
    
    return np.array((h,s,v))

def rle_encode(mask_list):
    mask = np.asfortranarray(np.stack(mask_list, axis=2))
    masks_rle = coco_mask.encode(mask)
    masks_str = []
    for masks in masks_rle:
        masks_str.append({"size":masks["size"], "counts":masks["counts"].decode("ascii")})
    return masks_str

def rle_decode(rle_dict):
    masks = []
    for i in range(len(rle_dict)):
        rle = rle_dict[i]
        rle["counts"] = rle["counts"].encode("ascii")
        mask = coco_mask.decode(rle)
        masks.append(mask.astype(np.bool_))
    return masks

def mask_to_bbox(binary_img, min_area = None, max_area = None):
    # Contours
    contours, hierarchy = cv.findContours(binary_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Bounding rectangle
    bboxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        bbox = np.array([x, y, x + w, y + h])
        area = cv.contourArea(contour)
        if (min_area is None or area > min_area) and (max_area is None or area < max_area):
            bboxes.append(bbox)
    return bboxes