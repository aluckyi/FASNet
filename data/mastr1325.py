import os
import os.path as osp
from collections import OrderedDict
import torch.utils.data as data
from . import utils


class MaSTr1325(data.Dataset):

    # The values associated with the 3 classes
    classes = (0, 1, 2)  # obstacle, water, sky

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('obstacle', (255, 255, 0)),    # yellow (255, 255, 0)  (255, 246, 132)
        ('water', (65, 105, 255)),      # RoyalBlue (deep blue)  (65, 105, 255)  (142, 216, 248)
        ('sky', (135, 206, 235)),       # SkyBlue   (light blue)  (135, 206, 235) (118, 112, 179)
    ])

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 loader=utils.pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.loader = loader
        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            cur_dir = osp.split(osp.abspath(__file__))[0]
            file_list = osp.join(cur_dir, 'datalist', self.mode.lower() + '.txt')
            file_list = tuple(open(file_list, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.train_data = [osp.join(self.root_dir, 'images', id_ + '.jpg') for id_ in file_list]
            self.train_labels = [osp.join(self.root_dir, 'masks', id_ + 'm.png') for id_ in file_list]

        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            cur_dir = osp.split(osp.abspath(__file__))[0]
            file_list = osp.join(cur_dir, 'datalist', self.mode.lower() + '.txt')
            file_list = tuple(open(file_list, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.val_data = [osp.join(self.root_dir, 'images', id_ + '.jpg') for id_ in file_list]
            self.val_labels = [osp.join(self.root_dir, 'masks', id_ + 'm.png') for id_ in file_list]

        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            # Get the validation data and labels filepaths
            cur_dir = osp.split(osp.abspath(__file__))[0]
            file_list = osp.join(cur_dir, 'datalist', self.mode.lower() + '.txt')
            file_list = tuple(open(file_list, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.test_data = [osp.join(self.root_dir, 'images', id_ + '.jpg') for id_ in file_list]
            self.test_labels = [osp.join(self.root_dir, 'masks', id_ + 'm.png') for id_ in file_list]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
                Args:
                - index (``int``): index of the item in the dataset

                Returns:
                A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
                of the image.

                """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        if self.transform is not None:
            img, label = self.transform((img, label))

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")



