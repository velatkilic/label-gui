import numpy as np
import colorsys

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
    n_masks = len(masks)
    if colors is None:
        colors = np.random.randint(0, 255, size=(n_masks, 3))
    
    mask = np.zeros(img.shape)
    for i, m in enumerate(masks):
        m = normalize(m)
        m = m[:,:,None] * hsv_to_rgb(colors[i])
        mask += m
    mask /= n_masks

    return img*alpha + mask*(1-alpha)

def make_new_color(seed_color, a=20, b=5):
    h, s, v = seed_color
    delta = np.maximum(np.random.randint(a), b)
    h += delta * np.sign(np.random.rand() - 0.5)
    h = np.clip(h, 0, 259)
    
    return np.array((h,s,v))