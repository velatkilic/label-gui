import numpy as np

def normalize(x):
    x = x.astype(np.float32)
    mx = x.max()
    mn = x.min()
    df = mx - mn
    if df != 0:
        x = (x - mn) / df
    return x

def draw_segmentation_masks(img, masks, alpha, colors=None):
    n_masks = len(masks)
    if colors is None:
        colors = np.random.randint(0, 255, size=(n_masks, 3))
    
    mask = np.zeros(img.shape)
    for i, m in enumerate(masks):
        m = normalize(m)
        m = m[:,:,None] * colors[i]
        mask += m
    mask /= n_masks

    return img*alpha + mask*(1-alpha)

