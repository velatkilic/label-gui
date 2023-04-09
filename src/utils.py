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