import random
import torch
from PIL import Image
import torchvision.transforms.functional as vF

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


class ToTensor(object):
    """Convert a (H x W x C) PIL Image / uint8 ndarray in [0, 255] to a
    (C x H x W) float tensor in [0.0, 1.0]; other dtypes are left unscaled."""

    def __call__(self, image, color=True):
        if image.ndim == 2:
            image = image[:, :, None]
        image = torch.from_numpy(
            ((image / 255).transpose([2, 0, 1]).astype("float32")).copy()
        )  # convert numpy data to tensor
        return image

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Rotate(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        # img must be tensor; random rotation by a multiple of self.degrees
        if self.degrees == 0:
            return img
        max_k = 360 // self.degrees
        k = random.randint(0, max_k - 1)
        deg = k * self.degrees
        return vF.rotate(img, deg)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs):
        imgs = vF.normalize(imgs, mean=self.mean, std=self.std)
        return imgs


def make_transform(degrees=0):
    normalize_value = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    selected_norm = normalize_value
    normalize = Compose([
        ToTensor(), 
        Rotate(degrees),
        Normalize(selected_norm[0], selected_norm[1])
    ])

    return Compose(
        [
            normalize,
        ]
    )
