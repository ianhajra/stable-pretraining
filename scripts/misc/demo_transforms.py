import random
def simclr_random_resized_crop(img, size=224, scale=(0.2, 1.0), ratio=(0.75, 1.33)):
    # RandomResizedCrop using PIL only
    width, height = img.size
    area = width * height
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)
        w = int(round((target_area * aspect_ratio) ** 0.5))
        h = int(round((target_area / aspect_ratio) ** 0.5))
        if w <= width and h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            img_cropped = img.crop((j, i, j + w, i + h))
            return img_cropped.resize((size, size), Image.BILINEAR)
    # fallback
    return img.resize((size, size), Image.BILINEAR)

def simclr_horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def simclr_color_jitter(img, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
    # PIL-only ColorJitter
    # Brightness
    factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    img = Image.fromarray(np.uint8(np.clip(np.array(img) * factor, 0, 255)))
    # Contrast
    mean = np.mean(np.array(img), axis=(0, 1), keepdims=True)
    factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    img = Image.fromarray(np.uint8(np.clip((np.array(img) - mean) * factor + mean, 0, 255)))
    # Saturation
    arr = np.array(img).astype(np.float32)
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])[..., None]
    factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    arr = np.clip(arr * factor + gray * (1 - factor), 0, 255)
    img = Image.fromarray(np.uint8(arr))
    # Hue
    arr = np.array(img).astype(np.uint8)
    hsv = np.array(img.convert('HSV'))
    hue_shift = int((random.uniform(-hue, hue)) * 255)
    hsv[..., 0] = (hsv[..., 0].astype(int) + hue_shift) % 256
    img = Image.fromarray(hsv, mode='HSV').convert('RGB')
    return img

def simclr_random_grayscale(img):
    return img.convert('L').convert('RGB')

import sys
import os
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import custom transforms directly from file, bypassing __init__.py and heavy deps
import importlib.util
from pathlib import Path
_custom_path = Path(__file__).parent.parent.parent / "stable_pretraining" / "data" / "transforms_custom.py"
spec = importlib.util.spec_from_file_location("transforms_custom", str(_custom_path))
transforms_custom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transforms_custom)
GaussianBlur = transforms_custom.GaussianBlur
MagnitudeScaling = transforms_custom.MagnitudeScaling
GaussianNoiseInjection = transforms_custom.GaussianNoiseInjection
RandomTemporalMasking = transforms_custom.RandomTemporalMasking

# Minimal versions of the transforms needed for this script
class ToTensor:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            arr = np.array(img)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            arr = arr.transpose((2, 0, 1))  # HWC to CHW
            return torch.from_numpy(arr).float() / 255.0
        elif torch.is_tensor(img):
            return img.float() / 255.0 if img.dtype == torch.uint8 else img
        else:
            raise TypeError("Input should be PIL Image or torch Tensor")

class ToImage:
    def __call__(self, tensor):
        if torch.is_tensor(tensor):
            arr = tensor.detach().cpu().numpy()
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
            arr = arr.transpose((1, 2, 0))  # CHW to HWC
            return Image.fromarray(arr.squeeze())
        elif isinstance(tensor, Image.Image):
            return tensor
        else:
            raise TypeError("Input should be torch Tensor or PIL Image")

class Resize:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        if isinstance(img, Image.Image):
            return img.resize((self.size, self.size), Image.BILINEAR)
        elif torch.is_tensor(img):
            from torchvision.transforms.functional import resize
            return resize(img, [self.size, self.size])
        else:
            raise TypeError("Input should be PIL Image or torch Tensor")

# Compose utility
class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

# Main script
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python demo_transforms.py path/to/image.png /absolute/output/dir")
        sys.exit(1)
    img_path = sys.argv[1]
    out_dir = sys.argv[2]
    if not os.path.isabs(out_dir):
        print("Output directory must be an absolute path.")
        sys.exit(1)
    img = Image.open(img_path).convert("RGB")

    # Always resize to 224x224 for fair comparison
    resize = Resize(224)
    to_tensor = ToTensor()
    to_image = ToImage()

    # List of transforms to test
    transforms_to_apply = [
                ("simclr_random_resized_crop", Compose(
                    lambda x: simclr_random_resized_crop(x, size=224), to_image)),
                ("simclr_horizontal_flip", Compose(
                    resize, lambda x: simclr_horizontal_flip(x), to_image)),
                ("simclr_color_jitter", Compose(
                    resize, lambda x: simclr_color_jitter(x, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1), to_image)),
                ("simclr_random_grayscale", Compose(
                    resize, lambda x: simclr_random_grayscale(x), to_image)),
                ("simclr_pipeline_combined", Compose(
                    lambda x: simclr_random_resized_crop(x, size=224),
                    simclr_horizontal_flip,
                    lambda x: simclr_color_jitter(x, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                    simclr_random_grayscale,
                    to_image
                )),
        ("original", lambda x: x),
        ("gaussian_blur", Compose(resize, to_tensor, GaussianBlur(sigma_range=(4.0,6.0), p=1.0), to_image)),
        ("magnitude_scaling", Compose(resize, to_tensor, MagnitudeScaling(scale_range=(1.6,2.0), p=1.0), to_image)),
        ("gaussian_noise", Compose(resize, to_tensor, GaussianNoiseInjection(sigma=0.25, p=1.0), to_image)),
        ("temporal_masking", Compose(resize, to_tensor, RandomTemporalMasking(mask_ratio_range=(0.30,0.40), p=1.0), to_image)),
        ("all_except_masking", Compose(
            resize, to_tensor,
            GaussianBlur(sigma_range=(4.0,6.0), p=1.0),
            MagnitudeScaling(scale_range=(1.6,2.0), p=1.0),
            GaussianNoiseInjection(sigma=0.25, p=1.0),
            to_image
        )),
        ("all_transforms_combined", Compose(
            resize, to_tensor,
            GaussianBlur(sigma_range=(4.0,6.0), p=1.0),
            MagnitudeScaling(scale_range=(1.6,2.0), p=1.0),
            GaussianNoiseInjection(sigma=0.25, p=1.0),
            RandomTemporalMasking(mask_ratio_range=(0.30,0.40), p=1.0),
            to_image
        )),
    ]

    # Apply and save each transform
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(out_dir, base_name + "_transforms")
    os.makedirs(out_dir, exist_ok=True)

    images = []
    names = []
    for name, transform in transforms_to_apply:
        out_img = transform(img)
        out_path = os.path.join(out_dir, f"{name}.png")
        out_img.save(out_path)
        print(f"Saved: {out_path}")
        images.append(np.array(out_img.convert("RGB")))
        names.append(name)

    # Plot all images in a grid
    n = len(images)
    ncols = 4 if n > 4 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for idx, (im, title) in enumerate(zip(images, names)):
        ax = axes[idx // ncols, idx % ncols]
        ax.imshow(im)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    # Hide any unused axes
    for idx in range(len(images), nrows*ncols):
        axes[idx // ncols, idx % ncols].axis('off')
    plt.tight_layout()
    comp_path = os.path.join(out_dir, f"{base_name}_comparison.png")
    plt.savefig(comp_path)
    print(f"Saved comparison grid: {comp_path}")
    plt.close(fig)

    print(f"All transformed images saved to {out_dir}/")
