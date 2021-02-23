import torch
import torch.nn as nn
import os
import os.path as osp
import sys
sys.path.insert(1, 'data_processing/')
sys.path.insert(1, 'lgn/')
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import args
from args import setup_argparse

from data_processing.make_pytorch_data import initialize_datasets, data_to_loader
import make_plots
from lgn.models.lgn_toptag import LGNTopTag

import json
import pickle

from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import math

from make_plots import * # Roc curves + CONFUSION MATRIX
from training import *

# Get a unique directory name for each trained model
def get_model_fname(args, dataset, model):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}_epochs_{}_batch_{}_ntrain_{}'.format(
        model_name,
        dataset.split("/")[-1],
        args.num_epoch,
        args.batch_size,
        args.num_train)
    return model_fname

# Create the directory to store the weights/epoch for the trained models
def create_model_folder(args, model):
    if not osp.isdir(args.outpath):
        os.makedirs(args.outpath)

    model_fname = get_model_fname(args, 'model#', model)
    outpath = osp.join(args.outpath, model_fname)

    if osp.isdir(outpath):
        print("model output {} already exists, please delete it".format(outpath))
        sys.exit(0)
    else:
        os.makedirs(outpath)

    model_kwargs = {'model_name': model_fname, 'learning_rate': args.lr_init}

    with open('{}/model_kwargs.pkl'.format(outpath), 'wb') as f:
        pickle.dump(model_kwargs, f,  protocol=pickle.HIGHEST_PROTOCOL)

    return outpath

#---------------------------------------------------------------------------------------------

if __name__ == "__main__":

    args = setup_argparse()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({"num_train": 200, "num_valid": 200, "num_test": 200, "task": "train", "num_epoch": 1, "batch_size": 2, "batch_group_size": 1, "weight_decay": 0, "cutoff_decay": 0, "lr_init": 0.001, "lr_final": 1e-05, "lr_decay": 9999, "lr_decay_type": "cos", "lr_minibatch": True, "sgd_restart": -1, "optim": "amsgrad", "parallel": False, "shuffle": True, "seed": 1, "alpha": 50, "save": True, "test": True, "log_level": "info", "textlog": True, "predict": True, "quiet": True, "prefix": "nosave", "loadfile": "", "checkfile": "", "bestfile": "", "logfile": "", "predictfile": "", "workdir": "./", "logdir": "log/", "modeldir": "model/", "predictdir": "predict/", "datadir": "data/", "dataset": "jet", "target": "is_signal", "add_beams": False, "beam_mass": 1, "force_download": False, "cuda": True, "dtype": "float", "num_workers": 0, "pmu_in": False, "num_cg_levels": 3, "mlp_depth": 3, "mlp_width": 2, "maxdim": [3], "max_zf": [1], "num_channels": [2, 3, 4, 3], "level_gain": [1.0], "cutoff_type": ["learn"], "num_basis_fn": 10, "scale": 0.005, "full_scalars": False, "mlp": True, "activation": "leakyrelu", "weight_init": "randn", "input": "linear", "num_mpnn_levels": 1, "top": "linear", "gaussian_mask": False,
    # 'patience': 100, 'outpath': 'trained_models/', 'train': True, 'load': True, 'test': True,
    # 'load_model': 'LGNTopTag_model#four_epochs_batch32', 'load_epoch': 0})

    with open("args_cache.json", "w") as f:
        json.dump(vars(args), f)

    # check if gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Num of GPUs:", torch.cuda.device_count())

    if device.type == 'cuda':
        print("GPU tagger is:", torch.cuda.current_device())
        print("GPU model:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    print('Working on:', device)

    if args.dtype == 'double':
        dtype = torch.double
    elif args.dtype == 'float':
        dtype = torch.float

    # load the data and cast it as a Pytorch dataloader
    args, torch_datasets = initialize_datasets(args, datadir='../data', num_pts=None)

    train_loader, test_loader, valid_loader = data_to_loader(args, torch_datasets)

    # Initialize model
    model = LGNTopTag(maxdim=args.maxdim, max_zf=args.max_zf, num_cg_levels=args.num_cg_levels, num_channels=args.num_channels, weight_init=args.weight_init, level_gain=args.level_gain, num_basis_fn=args.num_basis_fn,
                      top=args.top, input=args.input, num_mpnn_layers=args.num_mpnn_levels, activation=args.activation, pmu_in=args.pmu_in, add_beams=args.add_beams,
                      scale=1., full_scalars=args.full_scalars, mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                      device=device, dtype=dtype)

    if (next(model.parameters()).is_cuda):
        print('model is indeed training on gpu..')
    else:
        print('model is not training on gpu..')

    if args.train:
        # get directory name for the model to train
        outpath = create_model_folder(args, model)

        # define an optimizer
        optimizer = torch.optim.Adam(model.parameters(), args.lr_init)

        # start the training loop
        train_loop(args, model, optimizer, outpath, train_loader, valid_loader, device)

        # test over all the epochs trained
        if args.test_over_all_epoch:
            for epoch in range(args.num_epoch):
                # load the model per epoch to be tested
                PATH = outpath + '/epoch_' + str(epoch+1) + '_weights.pth'
                model.load_state_dict(torch.load(PATH, map_location=device))

                # evaluate the model
                make_plots.Evaluate(args, model, epoch, test_loader, outpath)

        # test the model over the last epoch only
        elif args.test:
            PATH = outpath + '/epoch_' + str(args.num_epoch) + '_weights.pth'
            model.load_state_dict(torch.load(PATH, map_location=device))

            # evaluate the model
            make_plots.Evaluate(args, model, args.num_epoch, test_loader, outpath)


    if args.load:
        if args.load_epoch == 0:
            for epoch in range(args.num_epoch):
                # load the correct model per epoch
                PATH = outpath + '/epoch_' + str(epoch+1) + '_weights.pth'
                model.load_state_dict(torch.load(PATH, map_location=device))

                # evaluate the model
                make_plots.Evaluate(args, model, epoch, test_loader, outpath)

        else:
            # test only the chosen epoch
            make_plots.Evaluate(args, model, args.load_epoch, test_loader, outpath)


# with open('trained_models/LGNTopTag_model#four_epochs_batch32/fractional_loss_train.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     f = pickle.load(f)
