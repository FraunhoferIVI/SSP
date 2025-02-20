import random
import numpy as np
import PIL
from PIL import Image
import numbers
import torchvision
import cv2
import skimage
import torch
import torchvision.transforms as transforms

import albumentations as aug
from albumentations.augmentations.crops import functional as fcrops
from albumentations.augmentations import functional as fmain


def crop_img(img, min_h, min_w, h, w):
    if isinstance(img, np.ndarray):
        cropped = img[min_h:min_h + h, min_w:min_w + w]

    elif isinstance(img, PIL.Image.Image):
        cropped = img.crop((min_w, min_h, min_w + w, min_h + h))
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got {0}'.format(type(img)))
    return cropped


def resize_img(img, size, interpolation='bilinear'):
    if isinstance(img, np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = img.shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return img
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = cv2.resize(img, size, interpolation=np_inter)
    elif isinstance(img, PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = img.size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return img
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = img.resize(size, pil_inter)
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got {0}'.format(type(img)))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def soft_to_hard_labels(soft_labels, ignore_index):
    background_pixels = soft_labels.sum(1) == 0
    labels = soft_labels.argmax(1)
    if ignore_index > 0:
        labels[background_pixels] = ignore_index
    return labels

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
class Identity(object):
    def __call__(self, x):
        return x
    
class Normalize(object):
    def __init__(self, mean, std):
        self.img_transform = aug.Normalize(mean=mean, std=std, max_pixel_value=255.)
    def __call__(self, img):
        img = self.img_transform(image=img)["image"]
        return img

class LabelToTensor(object):
    def __init__(self, n_classes, convert_labels=None):
        super(LabelToTensor, self).__init__()
        self.convert_labels = convert_labels
        self.n_classes = n_classes
    def __call__(self, label):
        if isinstance(label, Image.Image):
            label = np.array(label)
        if self.convert_labels is not None:
            label = self.convert_labels(label.copy())
        return torch.LongTensor(label)

# https://github.com/albumentations-team/albumentations/blob/bb6a6fa29f3bc259869bba6742ca8cc6087a00c1/albumentations/augmentations/crops/functional.py#L128
# See how "fcrops.get_center_crop_coords" works: image_shape: tuple[int, int], crop_shape: tuple[int, int] -> tuple[int, int, int, int]
class CenterCrop(object):
    def __init__(self, size):
        super(CenterCrop, self).__init__()
        self.size = size
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        image_shape = img.shape[:2]
        crop_coords = fcrops.get_center_crop_coords((image_shape[0], image_shape[1]), (self.size[0], self.size[1])) 
        x_min = crop_coords[0]
        y_min = crop_coords[1]
        x_max = crop_coords[2]
        y_max = crop_coords[3]
        img = fcrops.crop(img, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        return img
    
class ToTensor(object):
    def __init__(self):
        super(ToTensor, self).__init__()
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        img = torch.Tensor(img.copy())
        img = img.permute(2,0,1)
        return img
    
class SoftLabelResize(object):
    def __init__(self, n_classes, size, convert_labels=None, ignore_index=255, interpolation="bilinear"):
        super(SoftLabelResize, self).__init__()
        self.convert_labels = convert_labels
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.size = size
        self.interpolation = interpolation
    def __call__(self, label):
        if isinstance(label, Image.Image):
            label = np.array(label)
        if self.convert_labels is not None:
            label = self.convert_labels(label)
        # Soft label conversion
        if self.ignore_index > 0:
            label[label==self.ignore_index] = self.n_classes
        soft_label = torch.nn.functional.one_hot(torch.Tensor(label).long(), self.n_classes + 1)[:,:,:-1]*1.
        # Resize
        soft_label = torch.nn.functional.interpolate(soft_label.permute(2,0,1).unsqueeze(0), self.size, mode=self.interpolation, antialias=True).squeeze().permute(1,2,0)
        soft_label = np.array(soft_label)
        #soft_label = resize_img(soft_label, self.size, interpolation=self.interpolation)
        return soft_label
    
class SoftLabelResizeTorch(object):
    def __init__(self, n_classes, convert_labels=None, ignore_index=255, crop_size=(576,756), interpolation=transforms.InterpolationMode.BILINEAR):
        super(SoftLabelResizeTorch, self).__init__()
        self.convert_labels = convert_labels
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.crop_size = crop_size
        self.interpolation = interpolation
    def __call__(self, label):
        if isinstance(label, Image.Image):
            label = np.array(label)
        if self.convert_labels is not None:
            label = self.convert_labels(label)
        # Soft label conversion
        if self.ignore_index is not None:
            label[label==self.ignore_index] = self.n_classes
        soft_label = torch.nn.functional.one_hot(torch.Tensor(label).long(), self.n_classes + 1)
        soft_label = soft_label.permute(2,0,1)[:-1,:,:]*1.
        # Resize
        soft_label = transforms.functional.resize(soft_label, self.crop_size, self.interpolation, antialias=True)
        return soft_label

class VideoCompose_wLabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame, adj_frames, ref_frame, label, adj_labels=None):
        if adj_labels is None:
            adj_labels = []
        for t in self.transforms:
            frame, adj_frames, ref_frame, label, adj_labels = t(frame, adj_frames, ref_frame, label, adj_labels)
        if len(adj_labels) == 0:
            return frame, adj_frames, ref_frame, label
        else:
            return frame, adj_frames, ref_frame, label, adj_labels

class VideoCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame, adj_frames, ref_frame):
        for t in self.transforms:
            frame, adj_frames, ref_frame = t(frame, adj_frames, ref_frame)
        return frame, adj_frames, ref_frame
    
class VideoIdentity(object):
    def __call__(self, frame, adj_frames, ref_frame):
        return frame, adj_frames, ref_frame
    
class VideoIdentity_wLabel(object):
    def __call__(self, frame, adj_frames, ref_frame, label, adj_labels):
        return frame, adj_frames, ref_frame, label, adj_labels
    
class VideoResize(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.img_transform = transforms.Resize(crop_size)

    def __call__(self, frame, adj_frames, ref_frame):
        frame = self.img_transform(frame)
        adj_frames = [self.img_transform(adj_f) for adj_f in adj_frames]
        if ref_frame is not None:
            ref_frame = self.img_transform(ref_frame)
        return frame, adj_frames, ref_frame
    
class VideoToTensor(object):
    def __init__(self):
        self.img_transform = ToTensor()

    def __call__(self, frame, adj_frames, ref_frame):
        frame = self.img_transform(frame)
        adj_frames = [self.img_transform(adj_f) for adj_f in adj_frames]
        if ref_frame is not None:
            ref_frame = self.img_transform(ref_frame)
        return frame, adj_frames, ref_frame
    
class VideoNormalize(object):
    def __init__(self, mean, std):
        self.img_transform = Normalize(mean=mean, std=std)

    def __call__(self, frame, adj_frames, ref_frame):
        frame = self.img_transform(frame)
        adj_frames = [self.img_transform(adj_f) for adj_f in adj_frames]
        if ref_frame is not None:
            ref_frame = self.img_transform(ref_frame)
        return frame, adj_frames, ref_frame
    
class VideoCenterCrop(object):
    def __init__(self, crop_size):
        self.img_transform = CenterCrop(crop_size)

    def __call__(self, frame, adj_frames, ref_frame):
        frame = self.img_transform(frame)
        adj_frames = [self.img_transform(adj_f) for adj_f in adj_frames]
        if ref_frame is not None:
            ref_frame = self.img_transform(ref_frame)
        return frame, adj_frames, ref_frame
    

class VideoRandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, frame, adj_frames, ref_frame, label, adj_labels):
        if random.random() < self.prob:
            if isinstance(frame, np.ndarray):
                frame = np.fliplr(frame)
                adj_frames = [np.fliplr(f) for f in adj_frames]
                if ref_frame is not None:
                    ref_frame = np.fliplr(ref_frame)
                label = np.fliplr(label)
                adj_labels = [np.fliplr(l) for l in adj_labels]
                return frame, adj_frames, ref_frame, label, adj_labels
            elif isinstance(frame, PIL.Image.Image):
                frame = frame.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                adj_frames = [f.transpose(PIL.Image.FLIP_TOP_BOTTOM) for f in adj_frames]
                if ref_frame is not None:
                    ref_frame = ref_frame.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                label = label.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                adj_labels = [l.transpose(PIL.Image.FLIP_TOP_BOTTOM) for l in adj_labels]
                return frame, adj_frames, ref_frame, label, adj_labels
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(frame)))
        return frame, adj_frames, ref_frame, label, adj_labels
    
class VideoRandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, frame, adj_frames, ref_frame, label, adj_labels):
        if random.random() < self.prob:
            if isinstance(frame, np.ndarray):
                frame = np.flipud(frame)
                adj_frames = [np.flipud(f) for f in adj_frames]
                if ref_frame is not None:
                    ref_frame = np.flipud(ref_frame)
                label = np.flipud(label)
                adj_labels = [np.flipud(l) for l in adj_labels]

                return frame, adj_frames, ref_frame, label, adj_labels
            elif isinstance(frame, PIL.Image.Image):
                frame = frame.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                adj_frames = [f.transpose(PIL.Image.FLIP_TOP_BOTTOM) for f in adj_frames]
                if ref_frame is not None:
                    ref_frame = ref_frame.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                label = label.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                adj_labels = [l.transpose(PIL.Image.FLIP_TOP_BOTTOM) for l in adj_labels]
                return frame, adj_frames, ref_frame, label, adj_labels
            else: 
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(frame)))
        return frame, adj_frames, ref_frame, label, adj_labels
    
class VideoRandomCrop(object):
    def __init__(self, prob, size):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size
        self.prob = prob

    def __call__(self, frame, adj_frames, ref_frame, label, adj_labels):
        if random.random() < self.prob:
            if isinstance(frame, np.ndarray):
                im_h, im_w, im_c = frame.shape
            elif isinstance(frame, PIL.Image.Image):
                im_w, im_h = frame.size
            
            h, w = self.size
            if w > im_w or h > im_h:
                error_msg = (
                    'Initial image size should be larger then '
                    'cropped size but got cropped sizes : ({w}, {h}) while '
                    'scaled image is ({im_w}, {im_h})'.format(
                        im_w=im_w, im_h=im_h, w=w, h=h))
                raise ValueError(error_msg)
                
            x1 = random.randint(0, im_h - h)
            y1 = random.randint(0, im_w - w)
            frame = crop_img(frame, x1, y1, h, w)
            adj_frames = [crop_img(adj_f, x1, y1, h, w) for adj_f in adj_frames]
            if ref_frame is not None:
                ref_frame = crop_img(ref_frame, x1, y1, h, w)
            label = crop_img(label, x1, y1, h, w)
            adj_labels = [crop_img(adj_l, x1, y1, h, w) for adj_l in adj_labels]

            return frame, adj_frames, ref_frame, label, adj_labels
        return frame, adj_frames, ref_frame, label, adj_labels

class VideoRandomScaleCrop(object):
    def __init__(self, prob, size=None, scales=[0.95,1.,1.5,2.0], interpolation='nearest'):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size
        self.scales = scales
        self.interpolation = interpolation
        self.prob = prob

    def __call__(self, frame, adj_frames, ref_frame, label, adj_labels):
        if random.random() < self.prob:
            scaling_factor = random.choice(self.scales)

            if isinstance(frame, np.ndarray):
                im_h, im_w, im_c = frame.shape
            elif isinstance(frame, PIL.Image.Image):
                im_w, im_h = frame.size

            new_h = int(im_h * scaling_factor)
            new_w = int(im_w * scaling_factor)
            new_size = (new_h, new_w)
            frame = resize_img(frame, new_size, interpolation=self.interpolation)
            adj_frames = [resize_img(adj_f, new_size, interpolation=self.interpolation) for adj_f in adj_frames]
            if ref_frame is not None:
                ref_frame = resize_img(ref_frame, new_size, interpolation=self.interpolation)
            label = resize_img(label, new_size, interpolation="nearest")
            adj_labels = [resize_img(adj_l, new_size, interpolation="nearest") for adj_l in adj_labels]
            
            if self.size is not None:
                h, w = self.size
                im_h, im_w = new_size
                if w > im_w or h > im_h:
                    error_msg = (
                        'Initial image size should be larger then '
                        'cropped size but got cropped sizes : ({w}, {h}) while '
                        'scaled image is ({im_w}, {im_h})'.format(
                            im_w=im_w, im_h=im_h, w=w, h=h))
                    raise ValueError(error_msg)

                
                x1 = random.randint(0, im_h - h)
                y1 = random.randint(0, im_w - w)
                frame = crop_img(frame, x1, y1, h, w)
                adj_frames = [crop_img(adj_f, x1, y1, h, w) for adj_f in adj_frames]
                if ref_frame is not None:
                    ref_frame = crop_img(ref_frame, x1, y1, h, w)
                label = crop_img(label, x1, y1, h, w)
                adj_labels = [crop_img(adj_l, x1, y1, h, w) for adj_l in adj_labels]

            return frame, adj_frames, ref_frame, label, adj_labels
        return frame, adj_frames, ref_frame, label, adj_labels
    
class VideoRandomRotation(object):
    def __init__(self, prob, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees
        self.prob = prob

    def __call__(self, frame, adj_frames, ref_frame, label, adj_labels):
        if random.random() < self.prob:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            if isinstance(frame, np.ndarray):
                frame = skimage.transform.rotate(frame, angle)
                adj_frames = [skimage.transform.rotate(f, angle, order=0) for f in adj_frames]
                if ref_frame is not None:
                    ref_frame = skimage.transform.rotate(ref_frame, angle)
                label = skimage.transform.rotate(label, angle)
                adj_labels = [skimage.transform.rotate(l, angle, order=0) for l in adj_labels]
            elif isinstance(frame, PIL.Image.Image):
                frame = frame.rotate(angle)
                adj_frames = [f.rotate(angle) for f in adj_frames]
                if ref_frame is not None:
                    ref_frame = ref_frame.rotate(angle)
                label = label.rotate(angle, resample=Image.Resampling.NEAREST)
                adj_labels = [l.rotate(angle, resample=Image.Resampling.NEAREST) for l in adj_labels]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                'but got list of {0}'.format(type(frame)))

            return frame, adj_frames, ref_frame, label, adj_labels
        return frame, adj_frames, ref_frame, label, adj_labels
    

class VideoColorJitter(object):
    def __init__(self, prob, brightness=0.2, contrast=0.2, saturation=20, hue=30, val=20, gamma_limit=(80, 120)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.val = val
        self.prob = prob
        self.gamma_limit = gamma_limit

    def get_params(self):
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = 0.0 + random.uniform(-self.brightness, self.brightness)

        hue_shift = random.uniform(-self.hue, self.hue)
        sat_shift = random.uniform(-self.saturation, self.saturation)
        val_shift = random.uniform(-self.val, self.val)

        gamma = random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0

        return alpha, beta, hue_shift, sat_shift, val_shift, gamma

    # https://github.com/albumentations-team/albumentations/blob/main/albumentations/augmentations/crops/functional.py
    # albumentations.augmentations.functional' has no attribute 'brightness_contrast_adjust' -> use 'multiply_add'
    def __call__(self, frame, adj_frames, ref_frame):
        if random.random() < self.prob:
            alpha, beta, hue_shift, sat_shift, val_shift, gamma = self.get_params()
            frame = fmain.multiply_add(frame, alpha, beta) # 
            adj_frames = [fmain.multiply_add(f, alpha, beta) for f in adj_frames]
            if ref_frame is not None:
                ref_frame = fmain.multiply_add(ref_frame, alpha, beta)

            frame = fmain.shift_hsv(frame, hue_shift, sat_shift, val_shift)
            adj_frames = [fmain.shift_hsv(f, hue_shift, sat_shift, val_shift) for f in adj_frames]
            if ref_frame is not None:
                ref_frame = fmain.shift_hsv(ref_frame, hue_shift, sat_shift, val_shift)

            frame = fmain.gamma_transform(frame, gamma=gamma)
            adj_frames = [fmain.gamma_transform(f, gamma=gamma) for f in adj_frames]
            if ref_frame is not None:
                ref_frame = fmain.gamma_transform(ref_frame, gamma=gamma)
            
            return frame, adj_frames, ref_frame
        return frame, adj_frames, ref_frame


class VideoColorJitterTorch(object):
    def __init__(self, prob, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, frame, adj_frames, ref_frame):
        if random.random() < self.prob:
            if isinstance(frame, np.ndarray):
                raise TypeError(
                    'Color jitter not yet implemented for numpy arrays')
            elif isinstance(frame, PIL.Image.Image):
                brightness, contrast, saturation, hue = self.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

                # Create img transform function sequence
                img_transforms = []
                if brightness is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
                if saturation is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
                if hue is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
                if contrast is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
                random.shuffle(img_transforms)

                # Apply to all images
                for func in img_transforms:
                    frame = func(frame)
                adj_frames_out = []
                for f in adj_frames:
                    for func in img_transforms:
                        f = func(f)
                    adj_frames_out.append(f)
                if ref_frame is not None:
                    for func in img_transforms:
                        ref_frame = func(ref_frame)

            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                'but got list of {0}'.format(type(frame)))
            return frame, adj_frames_out, ref_frame
        return frame, adj_frames, ref_frame