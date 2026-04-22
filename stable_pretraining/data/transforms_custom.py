import random
import math
import torch
import numpy as np
from PIL import Image, ImageFilter

class GaussianBlur:
    def __init__(self, sigma_range=(0.1, 2.0), p=0.5):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        sigma = random.uniform(*self.sigma_range)
        if isinstance(img, Image.Image):
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        elif torch.is_tensor(img):
            # img: (C, H, W) or (H, W)
            from torch.nn.functional import conv2d
            if img.dim() == 2:
                img = img.unsqueeze(0)
            if img.dim() == 3:
                img = img.unsqueeze(0)  # (1, C, H, W)
            c = img.shape[1]
            kernel_size = int(2 * math.ceil(2 * sigma) + 1)
            x = torch.arange(kernel_size) - kernel_size // 2
            gauss = torch.exp(-x**2 / (2 * sigma**2))
            gauss = gauss / gauss.sum()
            kernel1d = gauss.to(img.device, dtype=img.dtype)
            kernel2d = torch.outer(kernel1d, kernel1d)
            kernel2d = kernel2d.expand(c, 1, kernel_size, kernel_size)
            padding = kernel_size // 2
            img_blur = conv2d(img, kernel2d, padding=padding, groups=c)
            return img_blur.squeeze(0)
        else:
            raise TypeError("Input should be PIL Image or torch Tensor")

class MagnitudeScaling:
    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        scale = random.uniform(*self.scale_range)
        if isinstance(img, Image.Image):
            arr = np.array(img)
            orig_dtype = arr.dtype
            # Determine if original was uint8 (common for PIL) or float
            if orig_dtype == np.uint8:
                arr = arr.astype(np.float32) * scale
                arr = np.clip(arr, 0, 255)
                arr = arr.astype(np.uint8)
            else:
                arr = arr.astype(np.float32) * scale
                arr = np.clip(arr, 0, 1)
            return Image.fromarray(arr)
        elif torch.is_tensor(img):
            orig_dtype = img.dtype
            out = img * scale
            # If uint8, clamp to [0,255], else clamp to [0,1]
            if orig_dtype == torch.uint8:
                out = out.clamp(0, 255)
            else:
                out = out.clamp(0, 1)
            return out.type(orig_dtype) if orig_dtype != torch.float32 else out
        else:
            raise TypeError("Input should be PIL Image or torch Tensor")

class GaussianNoiseInjection:
    def __init__(self, sigma=0.05, p=0.5):
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if isinstance(img, Image.Image):
            arr = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.sigma, arr.shape).astype(np.float32)
            arr = arr + noise
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr)
        elif torch.is_tensor(img):
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            noise = torch.randn_like(img) * self.sigma
            out = img + noise
            out = out.clamp(0, 1)
            return out
        else:
            raise TypeError("Input should be PIL Image or torch Tensor")

class RandomTemporalMasking:
    def __init__(self, mask_ratio_range=(0.10, 0.20), p=0.5):
        self.mask_ratio_range = mask_ratio_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if isinstance(img, Image.Image):
            arr = np.array(img)
            h, w = arr.shape[:2]
            mask_ratio = random.uniform(*self.mask_ratio_range)
            mask_width = max(1, int(w * mask_ratio))
            start = random.randint(0, w - mask_width)
            arr[..., start:start+mask_width] = 0
            return Image.fromarray(arr)
        elif torch.is_tensor(img):
            # img: (C, H, W) or (H, W)
            orig_dim = img.dim()
            if orig_dim == 2:
                img = img.unsqueeze(0)
            c, h, w = img.shape[-3:]
            mask_ratio = random.uniform(*self.mask_ratio_range)
            mask_width = max(1, int(w * mask_ratio))
            start = random.randint(0, w - mask_width)
            img_clone = img.clone()
            img_clone[..., :, start:start+mask_width] = 0
            if orig_dim == 2:
                return img_clone.squeeze(0)
            else:
                return img_clone
        else:
            raise TypeError("Input should be PIL Image or torch Tensor")
