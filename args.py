from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    # ===================================== #
    # (1) Execution mode
    # ===================================== #
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full'],
        default='train',    # !!!!!!******
        help=("train: performs training and validation; test: tests the model "
              "found in \"--save_dir\" with name \"--name\" on \"--dataset\"; "
              "full: combines train and test modes. Default: train"))
    parser.add_argument(
        "--resume",
        action='store_true',
        help=("The model found in \"--checkpoint_dir/--name/\" and filename "
              "\"--name.h5\" is loaded."))

    # ===================================== #
    # (2) Hyper-parameters of training
    # ===================================== #
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,  # !!!!!!******
        help="The batch size. Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs. Default: 150")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-3,
        help="The learning rate. Default: 1e-3")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.2,
        help="The learning rate decay factor. Default: 0.2")
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=30,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 500")
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=4e-4,
        help="L2 regularization factor. Default: 4e-4")
    parser.add_argument(
        "--loss-w1",
        type=float,
        default=0.6,
        help="The weight of loss. Default: 0.6")

    # ===================================== #
    # (3) Dataset settings
    # ===================================== #
    parser.add_argument(
        "--dataset",
        choices=['camvid', 'cityscapes', 'mastr1325'],
        default='mastr1325',
        help="Dataset to use. Default: camvid")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/MaSTr1325",  # !!!!!!******
        help="Path to the root directory of the selected dataset. "
        "Default: data/MaSTr1325")
    parser.add_argument(
        "--height",
        type=int,
        default=384,
        help="The image height. Default: 360")
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="The image width. Default: 480")
    parser.add_argument(
        "--crop-height",
        type=int,
        default=310,    # ****** no cropping ****** #
        help="The crop height. Default: 360")
    parser.add_argument(
        "--crop-width",
        type=int,
        default=410,    # ****** no cropping ****** #
        help="The crop width. Default: 720")
    parser.add_argument(
        "--color-jitter",
        type=list,
        default=[0.2, 0.2, 0.2, 0],
        help="The parameters of color jitter. Default: [0.2, 0.2, 0.2, 0]")
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.5,
        help="The probability of the image being flipped. Default: 0.5")
    parser.add_argument(
        "--weighing",
        choices=['enet', 'mfb', 'none'],
        default='enet',  # !!!!!!******
        help="The class weighing technique to apply to the dataset. "
        "Default: none")
    parser.add_argument(
        "--with-unlabeled",
        dest='ignore_unlabeled',
        action='store_true',    # !!!!!!******
        help="The unlabeled class is not ignored. Default: store_false")

    # ===================================== #
    # (4) Settings
    # ===================================== #
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. Default: 4")
    parser.add_argument(
        "--print-step",
        action='store_true',
        help="Print loss every step. Default: store_true")
    parser.add_argument(
        "--imshow-batch",
        action='store_true',  # !!!!!!******
        help=("Displays batch images when loading the dataset and making "
              "predictions. Default: store_true"))
    parser.add_argument(
        "--device",
        default='cuda:1',   # !!!!!!******
        help="Device on which the network will be trained. Default: cuda")

    # ===================================== #
    # (5) Storage settings
    # ===================================== #
    parser.add_argument(
        "--name",
        type=str,
        default='FASNet',   # !!!!!!******
        help="Name given to the model when saving. Default: FASNet")
    parser.add_argument(
        "--save-dir",
        type=str,
        default='save/FASNet',  # !!!!!!******
        help="The directory where models are saved. Default: save")

    return parser.parse_args()
