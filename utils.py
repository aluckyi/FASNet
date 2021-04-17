import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os


def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.

    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``

    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``

    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)

    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def save_checkpoint(model, optimizer, epoch, miou, args):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - args (``ArgumentParser``): An instance of ArgumentParser which contains
    the arguments used to train ``model``. The arguments are written to a text
    file in ``args.save_dir`` named "``args.name``_args.txt".

    """
    name = args.name
    save_dir = args.save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name + '.pth')
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)

    # =======================================
    tmp_path = os.path.join(save_dir, name + str(epoch) + '.pth')
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, tmp_path)

    tmp_summary_filename = os.path.join(save_dir, name + '_summary_' + str(epoch) + '.txt')
    with open(tmp_summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n".format(epoch))
        summary_file.write("Mean IoU: {0}\n".format(miou))
    # =======================================

    # Save arguments
    summary_filename = os.path.join(save_dir, name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(miou))


def load_checkpoint(model, optimizer, folder_dir, filename):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.

    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.

    """
    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename + '.pth')
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path, map_location='cuda:1')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    return model, optimizer, epoch, miou


class CustomLoss(nn.Module):
    def __init__(self, class_weights, class_encoding, w1=0.6):
        super(CustomLoss, self).__init__()
        self.w1 = 1.0
        self.w2 = 1.0
        print('w1: {}, w2: {}'.format(self.w1, self.w2))

        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.fs_criterion = FeatureSeparationLoss(class_encoding, class_weights)

    def forward(self, target, predicted, features):
        ce_loss = self.ce_criterion(predicted, target)
        fs_loss = self.fs_criterion(features, target)
        loss = ce_loss * self.w1 + fs_loss * self.w2

        return loss


class FeatureSeparationLoss(nn.Module):
    eps = 1e-6

    def __init__(self, class_encoding, class_weights=None):
        super(FeatureSeparationLoss, self).__init__()

        self.class_weights = class_weights
        self.class_encoding = class_encoding
        self.num_classes = len(self.class_encoding)

    def forward(self, feature, target):
        assert feature.dim() == 4, \
            "features must be of dimension (B, F, H, W)"
        assert target.dim() == 3, \
            "targets must be of dimension (B, H, W)"

        b, f, h, w = feature.size()

        with torch.no_grad():
            target_new = self.one_hot(target)  # (B, C, H, W)
            target_new = F.interpolate(target_new.float(), size=(h, w), mode='nearest')
            target_new = target_new.long()

        # calculate means of water and obstacle regions
        classes = list(self.class_encoding.keys())
        # ow_labels = [idn for idn, cls in enumerate(classes) if cls != 'sky']
        ow_labels = [idn for idn, cls in enumerate(classes)]

        mu_dict = []
        for idn in ow_labels:
            mask = target_new[:, idn].unsqueeze(1).float()  # (B, 1, H, W)
            masked_features = feature * mask  # (B, F, H, W)
            num = torch.sum(mask)
            mu = torch.sum(masked_features, dim=(0, 2, 3)) / (self.eps + num)  # (F,)
            mu_dict.append(mu)

        mu = torch.stack(mu_dict).unsqueeze(0).permute(0, 2, 1)  # (1, F, K)
        mu = mu.repeat(b, 1, 1)  # (B, F, K)
        mu = self._l2norm(mu, dim=1)

        x = feature.view(b, f, h * w)  # (B, F, N)
        x_t = x.permute(0, 2, 1)  # (B, N, F)
        z = torch.bmm(x_t, mu)  # (B, N, K)
        z = F.softmax(z, dim=2)  # (B, N, K)
        z = z.permute(0, 2, 1).view(b, -1, h, w)  # (B, K, H, W)

        # =============================================
        # import numpy as np
        # import matplotlib.pyplot as plt
        # aa = z.cpu().detach().numpy()
        # bb = []
        # for i in range(z.size()[0]):
        #     bb.append(aa[i, 0])
        # bb = np.hstack(bb)
        # plt.imshow(bb)
        # plt.show()

        # import numpy as np
        # import matplotlib.pyplot as plt
        # feat = feature.cpu().detach().numpy()
        # savename = os.path.join('./heatmap', 'heatmap.png')
        # draw_features(4, 4, feat, savename)
        # =============================================

        z_log = torch.log(z)   # (B, K, H, W)

        loss = 0
        num = 0
        for i, idn in enumerate(ow_labels):
            mask = target_new[:, idn].unsqueeze(1).float()  # (B, 1, H, W)
            z_ = z_log[:, i].unsqueeze(1)   # (B, 1, H, W)
            if self.class_weights is None:
                loss += torch.sum(-1.0 * mask * z_)
            else:
                loss += self.class_weights[idn] * torch.sum(-1.0 * mask * z_)
            num += torch.sum(mask)
        loss /= num

        return loss

    def one_hot(self, input):
        """Convert class index tensor to one hot encoding tensor.
            Args:
                 input: A tensor of shape (B, H, W)
            Returns:
                A tensor of shape (B, C, H, W)
            """
        assert input.dim() == 3, \
            "input must be of dimension (B, H, W)"

        input = input.unsqueeze(1)
        shape = np.array(input.shape)
        shape[1] = self.num_classes
        shape = tuple(shape)
        result = torch.zeros(shape, dtype=input.dtype, device=input.device)
        result = result.scatter_(1, input, 1)

        return result

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (self.eps + inp.norm(dim=dim, keepdim=True))


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.device != logpt.device:
                self.alpha = self.alpha.to(logpt.device)

            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [bs, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [bs, num_classes, *]
    """
    input = input.unsqueeze(1)
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape, dtype=input.dtype, device=input.device)
    result = result.scatter_(1, input, 1)

    return result


import time
import cv2


def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255
        img=img.astype(np.uint8)
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite('heatmap/ht_{}.png'.format(i), img)
        img = img[:, :, ::-1]
        plt.imshow(img)
        # plt.imsave('heatmap/ht_{}.png'.format(i), img)
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))


