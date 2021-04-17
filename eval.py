import os

import torch
import torch.nn as nn
import torch.optim as optim

import utils
from metric.stream_metrics import StreamSegMetrics
from main import load_dataset
import time

# from networks.enet import ENet_own1 as Net
# from networks.enet import FASNet as Net
# from networks.fssnet import FSSNet_own as Net
# from networks.fssnet import FASNet as Net
# from networks.erfnet import ERFNet_own as Net
# from networks.erfnet import FASNet as Net
# from networks.esnet import ESNet_own as Net
from networks.esnet import FASNet as Net

from args import get_arguments
# Get the arguments
args = get_arguments()

device = torch.device(args.device)

import torch


class Test:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        self.model.eval()
        epoch_loss = 0.0
        self.metric.reset()

        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            with torch.no_grad():
                # ========================================== #
                # # Forward propagation
                # outputs = self.model(inputs)
                #
                # # Loss computation
                # loss = self.criterion(outputs, labels)

                # Forward propagation

                results = self.model(inputs)
                if isinstance(results, tuple):
                    if len(results) == 2:
                        outputs, feats = results
                        # Loss computation
                        loss = self.criterion(labels, outputs, feats)
                    else:
                        raise RuntimeError("Unexpected outputs of the network.")
                else:
                    outputs = results
                    # Loss computation
                    loss = self.criterion(outputs, labels)
                # ========================================== #

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of evaluation the metric
            # self.metric.add(outputs.detach(), labels.detach())
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            self.metric.update(targets, preds)

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        score = self.metric.get_results()

        return epoch_loss / len(self.data_loader), score

    def run_epoch_time(self):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        self.model.eval()

        data_iter = iter(self.data_loader)
        inputs = data_iter.next()[0]
        inputs = inputs.to(self.device)

        frame_num = len(self.data_loader)
        print('frame_num: {}'.format(frame_num))
        torch.cuda.synchronize()
        time_start = time.time()

        for step in range(len(self.data_loader)):
            with torch.no_grad():
                # ========================================== #
                # Forward propagation
                _ = self.model(inputs)
                # ========================================== #

        torch.cuda.synchronize()
        time_end = time.time()
        total_time = (time_end - time_start) * 1000
        print('time cost', (total_time / frame_num), 'ms')


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    # =============================================== #
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = utils.CustomLoss(class_weights, class_encoding, w1=args.loss_w1)
    # =============================================== #

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    # metric = IoU(num_classes, ignore_index=ignore_index)
    metric = StreamSegMetrics(num_classes)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")

    # =============================================== #
    loss, score = test.run_epoch(args.print_step)
    print(metric.to_str(score))
    # test.run_epoch_time()
    # =============================================== #


if __name__ == '__main1__':
    # Fail fast if the dataset directory doesn't exist
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
        args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
        args.save_dir)

    # Import the requested dataset
    if args.dataset.lower() == 'camvid':
        from data import CamVid as dataset
    elif args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    elif args.dataset.lower() == 'mastr1325':
        from data import MaSTr1325 as dataset
    else:
        # Should never happen...but just in case it does
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(
            args.dataset))

    loaders, w_class, class_encoding = load_dataset(dataset)
    train_loader, val_loader, test_loader = loaders

    # Intialize a new Network
    num_classes = len(class_encoding)
    # =============================================== #
    model = Net(num_classes).to(device)
    # =============================================== #

    # Initialize a optimizer just so we can retrieve the model from the
    # checkpoint
    optimizer = optim.Adam(model.parameters())

    # Load the previoulsy saved model state to the FSSNet model
    model = utils.load_checkpoint(model, optimizer, args.save_dir,
                                  args.name)[0]
    print(model)
    test(model, test_loader, w_class, class_encoding)


if __name__ == '__main2__':
    from thop import profile
    num_classes = 3
    # =============================================== #
    # model = Net(num_classes).to(device)
    model = Net(num_classes)
    # =============================================== #
    # # Initialize a optimizer just so we can retrieve the model from the
    # # checkpoint
    # optimizer = optim.Adam(model.parameters())
    #
    # # Load the previoulsy saved model state to the FSSNet model
    # model = utils.load_checkpoint(model, optimizer, args.save_dir,
    #                               args.name)[0]
    input = torch.randn(1, 3, 384, 512)
    flop, para = profile(model, inputs=(input, ))
    print('%.2fG' % (flop/1e9), '%.2fM' % (para/1e6))


