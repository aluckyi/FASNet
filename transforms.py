import torch
import random
import sys
import collections
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision.transforms import ToPILImage, ToTensor, ColorJitter
from torchvision.transforms.functional import hflip, crop, resize

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class PILToLongTensor(object):
    """Converts a ``PIL Image`` to a ``torch.LongTensor``.

    Code adapted from: http://pytorch.org/docs/master/torchvision/transforms.html?highlight=totensor

    """

    def __call__(self, pic):
        """Performs the conversion from a ``PIL Image`` to a ``torch.LongTensor``.

        Keyword arguments:
        - pic (``PIL.Image``): the image to convert to ``torch.LongTensor``

        Returns:
        A ``torch.LongTensor``.

        """
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))

        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()

        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0,
                                             2).contiguous().long().squeeze_()


class LongTensorToRGBPIL(object):
    """Converts a ``torch.LongTensor`` to a ``PIL image``.

    The input is a ``torch.LongTensor`` where each pixel's value identifies the
    class.

    Keyword arguments:
    - rgb_encoding (``OrderedDict``): An ``OrderedDict`` that relates pixel
    values, class names, and class colors.

    """
    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
        """Performs the conversion from ``torch.LongTensor`` to a ``PIL image``

        Keyword arguments:
        - tensor (``torch.LongTensor``): the tensor to convert

        Returns:
        A ``PIL.Image``.

        """
        # Check if label_tensor is a LongTensor
        if not isinstance(tensor, torch.LongTensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}"
                            .format(type(tensor)))
        # Check if encoding is a ordered dictionary
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(
                type(self.rgb_encoding)))

        # label_tensor might be an image without a channel dimension, in this
        # case unsqueeze it
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)

        color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2))

        for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
            # Get a mask of elements equal to index
            mask = torch.eq(tensor, index).squeeze_()
            # Fill color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)

        return ToPILImage()(color_tensor)


class ToTensors(object):
    """Convert img (``PIL Image``) to ``torch.tensor`` and
               label (``PIL Image``) to ``torch.LongTensor``.

    In the other cases, tensors are returned without scaling.
    """
    def __init__(self):
        self.img_to_tensor = ToTensor()
        self.label_to_tensor = PILToLongTensor()

    def __call__(self, sample):
        """
        Args:
            sample (tuple): Tuple of img(PIL image) and label(PIL image).
                            Image to be converted to Tensor.
                            Label to be converted to LongTensor.
        Returns:
            Tensor: Converted image.
        """
        assert (isinstance(sample, tuple) and len(sample) == 2)
        img, label = sample
        img = self.img_to_tensor(img)
        label = self.label_to_tensor(label)
        return img, label


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        img_interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
        label_interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.NEAREST``
    """

    def __init__(self, size, img_interpolation=Image.BILINEAR, label_interpolation=Image.NEAREST):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    def __call__(self, sample):
        """
        Args:
            sample (tuple): Tuple of img(PIL image) and label(PIL image).
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        assert (isinstance(sample, tuple) and len(sample) == 2)
        img, label = sample
        img = resize(img, self.size, self.img_interpolation)
        label = resize(label, self.size, self.label_interpolation)
        return img, label


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """
        Args:
            sample (tuple): Tuple of img(PIL image) and label(PIL image).

        Returns:
            PIL image: Randomly cropped image.
            PIL Label: Randomly cropped label.
        """
        assert (isinstance(sample, tuple) and len(sample) == 2)
        img, label = sample
        w, h = img.size
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = crop(img, top, left, new_h, new_w)
        label = crop(label, top, left, new_h, new_w)

        return img, label


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            sample (tuple): Tuple of img(PIL image) and label(PIL image).

        Returns:
            PIL Image: Randomly flipped image.
            PIL Label: Randomly flipped label.
        """
        assert (isinstance(sample, tuple) and len(sample) == 2)
        img, label = sample
        if random.random() < self.p:
            return hflip(img), hflip(label)
        return img, label


class RandomColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        assert (isinstance(sample, tuple) and len(sample) == 2)
        img, label = sample
        img = self.color_jitter(img)
        return img, label

