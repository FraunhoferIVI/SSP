import numpy as np
from PIL import Image

def pred_to_mask(pred, ignore_index):
        if ignore_index > 0:
            pred[pred==ignore_index]=254
            pred = pred+1
            pred[pred==255]=0
        return pred

def mask_to_pred(mask, ignore_index):
    if ignore_index > 0:
        mask[mask==0]=ignore_index
        mask = mask-1
        mask[mask==ignore_index-1]=ignore_index
    return mask

def color_predictions(pred, colors, ignore_index, blend_img=None, is_mask=False):
    pred = pred.astype("int")
    if not is_mask:
        pred = pred_to_mask(pred, ignore_index)
    color_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=int)
    labels = np.unique(pred)
    for label in labels:
        if label<255:
            color_pred += (pred==label)[:,:,None] * np.tile(colors[label], (pred.shape[0], pred.shape[1], 1))


    if blend_img is not None:
        blend_img = Image.fromarray(blend_img.astype(np.uint8))
        color_pred = Image.fromarray(color_pred.astype(np.uint8))
        blend_result = Image.blend(blend_img, color_pred, 0.75)
        return color_pred, blend_result
    return color_pred


def inverse_normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    image = image.copy()
    for i in range(3):
        image[i] = (image[i] * std[i]) + mean[i]
    
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8).transpose(1,2,0)
    return image