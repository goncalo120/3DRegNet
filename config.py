# config.py ---
#
# Description:
# Author: Goncalo Pais
# Date: 28 Jun 2019
# https://arxiv.org/abs/1904.01701
# 
# Instituto Superior Técnico (IST)

# Code:

import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ("true", "1")


# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--data_pre", type=str, default="data", help=""
    "prefix for the dump folder locations")
data_arg.add_argument(
    "--data_tr", type=str, default="sun3d", help=""
    "name of the dataset for train")
data_arg.add_argument(
    "--data_va", type=str, default="sun3d", help=""
    "name of the dataset for valid")
data_arg.add_argument(
    "--data_te", type=str, default="sun3d", help=""
    "name of the dataset for test")

# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--net_depth", type=int, default=12, help=""
    "number of layers")
net_arg.add_argument(
    "--net_nchannel", type=int, default=128, help=""
    "number of channels in a layer")
net_arg.add_argument(
    "--net_act_pos", type=str, default="post",
    choices=["pre", "mid", "post"], help=""
    "where the activation should be in case of resnet")
net_arg.add_argument(
    "--net_gcnorm", type=str2bool, default=True, help=""
    "whether to use context normalization for each layer")
net_arg.add_argument(
    "--net_batchnorm", type=str2bool, default=True, help=""
    "whether to use batch normalization")
net_arg.add_argument(
    "--net_bn_test_is_training", type=str2bool, default=False, help=""
    "is_training value for testing")
net_arg.add_argument(
    "--net_concat_post", type=str2bool, default=False, help=""
    "retrieve top k values or concat from different layers")
net_arg.add_argument(
        "--gpu_options", type=str, default='gpu', choices=['gpu', 'cpu'],
    help="choose which gpu or cpu")
net_arg.add_argument(
    "--gpu_number", type=str, default='0',
    help="choose which gpu number")

# -----------------------------------------------------------------------------
# Loss
loss_arg = add_argument_group("loss")
loss_arg.add_argument(
    "--loss_decay", type=float, default=0.0, help=""
    "l2 decay")
loss_arg.add_argument(
    "--loss_classif", type=float, default=0.5, help=""
    "weight of the classification loss")
loss_arg.add_argument(
    "--loss_reconstruction", type=float, default=0.01, help=""
    "weight of the essential loss")
loss_arg.add_argument(
    "--loss_reconstruction_init_iter", type=int, default=20000, help=""
    "initial iterations to run only the classification loss")


# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument(
    "--run_mode", type=str, default="train", help=""
    "run_mode")
train_arg.add_argument(
    "--train_batch_size", type=int, default=16, help=""
    "batch size")
train_arg.add_argument(
    "--train_max_tr_sample", type=int, default=10000, help=""
    "number of max training samples")
train_arg.add_argument(
    "--train_max_va_sample", type=int, default=1000, help=""
    "number of max validation samples")
train_arg.add_argument(
    "--train_max_te_sample", type=int, default=1000, help=""
    "number of max test samples")
train_arg.add_argument(
    "--train_lr", type=float, default=1e-5, help=""
    "learning rate")
train_arg.add_argument(
    "--train_epoch", type=int, default=3750, help=""
    "training iterations to perform")
train_arg.add_argument(
    "--train_step", type=int, default=200, help=""
    "training iterations to perform")

train_arg.add_argument(
    "--res_dir", type=str, default="./logs", help=""
    "base directory for results")
train_arg.add_argument(
    "--log_dir", type=str, default="logs_lie", help=""
    "save directory name inside results")
train_arg.add_argument(
    "--test_log_dir", type=str, default="", help=""
    "which directory to test inside results")
train_arg.add_argument(
    "--val_intv", type=int, default=5, help=""
    "validation interval")
train_arg.add_argument(
    "--report_intv", type=int, default=100, help=""
    "summary interval")
net_arg.add_argument(
    "--loss_function", type=str, default='l1', choices=['l1', 'l2', 'wls', 'gm', 'l05'],
    help="choose which loss function")

# -----------------------------------------------------------------------------
# Data Augmentation
d_aug = add_argument_group('Augmentation')

d_aug.add_argument("--data_aug", type=str2bool, default=False, help="Perform data Augmentation")
d_aug.add_argument("--aug_cl", type=str2bool, default=True, help="Perform Curriculum Learning")
d_aug.add_argument("--aug_dir", type=str, default="augmentented", help="save directory name inside results")

# -----------------------------------------------------------------------------
# Visualization
vis_arg = add_argument_group('Visualization')
vis_arg.add_argument(
    "--tqdm_width", type=int, default=79, help=""
    "width of the tqdm bar")
vis_arg.add_argument(
    "--reg_flag", type=str2bool, default=False, help="Refine transformation")

test_arg = add_argument_group('Test')
test_arg.add_argument(
        "--reg_function", type=str, default='fast', choices=['fast', 'global'], help="Registration function: global or fast")

test_arg.add_argument(
    "--representation", type=str, default='lie', choices=['lie', 'quat', 'linear'], help="Type of Representation")


def setup_dataset(dataset_name):

    dataset_name = dataset_name.split(".")
    data_dir = []

    for name in dataset_name:

        if 'sun3d' == name:
            data_dir.append(name)

    assert data_dir

    return data_dir


def get_config():

    config, unparsed = parser.parse_known_args()

    # Setup the dataset related things
    for _mode in ["tr", "va", "te"]:
        data_dir = setup_dataset(
           getattr(config, "data_" + _mode))
        setattr(config, "data_dir_" + _mode, data_dir)
        # setattr(config, "data_geom_type_" + _mode, geom_type)
        # setattr(config, "data_vis_th_" + _mode, vis_th)

    return config, unparsed


def print_usage():
    parser.print_usage()
