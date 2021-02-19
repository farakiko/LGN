import torch
import torch.nn as nn
import os
import os.path as osp
import sys
sys.path.insert(1, 'data_processing/')
sys.path.insert(1, 'lgn/')

import args
from args import setup_argparse

from data_processing.make_pytorch_data import initialize_datasets, data_to_loader
from lgn.models.lgn_toptag import LGNTopTag

import json
import pickle

from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# import mplhep as hep
# plt.style.use(hep.style.ROOT)


def Evaluate(args, model, test_loader, outpath):
    model.eval()

    preds = []
    targets = []
    c=0
    acc=0

    for i, batch in enumerate(test_loader):
        pred = model(batch)

        preds.append(pred.detach().numpy())
        targets.append(batch['is_signal'].detach().numpy())

        c = c + (preds[i].argmax(axis=1) == targets[i]).sum().item()
        acc = 100*c/(args.batch_size*len(test_loader))

    with open(outpath + '/test_acc.pkl', 'wb') as f:
        pickle.dump(acc, f)

    # create ROC curves
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    fpr_gnn, tpr_gnn, threshold_gnn = roc_curve(targets, preds[:,1])

    # plot ROC curves
    #plt.figure()
    fig, ax = plt.subplots()
    ax.plot(tpr_gnn, fpr_gnn, lw=2.5, label="GNN, AUC = {:.1f}%".format(auc(fpr_gnn,tpr_gnn)*100))
    ax.set_xlabel(r'True positive rate')
    ax.set_ylabel(r'False positive rate')
    #ax.grid(True)
    ax.legend(loc='upper left')
    plt.savefig(outpath + '/Roc_curves.png')

    fig, ax = plt.subplots()
    ax.plot(tpr_gnn, fpr_gnn, lw=2.5, label="GNN, AUC = {:.1f}%".format(auc(fpr_gnn,tpr_gnn)*100))
    ax.set_xlabel(r'True positive rate')
    ax.set_ylabel(r'False positive rate')
    ax.semilogy()
    #ax.grid(True)
    ax.legend(loc='upper left')
    plt.savefig(outpath + '/Roc_curves_log.png')

    return acc


#---------------------------------------------------------------------------------

# # the next part initializes some args values (to run the script not from terminal)
# class objectview(object):
#     def __init__(self, d):
#         self.__dict__ = d
#
# args = objectview({"num_train": -1, "num_valid": -1, "num_test": -1, "task": "train", "num_epoch": 1, "batch_size": 2, "batch_group_size": 1, "weight_decay": 0, "cutoff_decay": 0, "lr_init": 0.001, "lr_final": 1e-05, "lr_decay": 9999, "lr_decay_type": "cos", "lr_minibatch": True, "sgd_restart": -1, "optim": "amsgrad", "parallel": False, "shuffle": True, "seed": 1, "alpha": 50, "save": True, "test": True, "log_level": "info", "textlog": True, "predict": True, "quiet": True, "prefix": "nosave", "loadfile": "", "checkfile": "", "bestfile": "", "logfile": "", "predictfile": "", "workdir": "./", "logdir": "log/", "modeldir": "model/", "predictdir": "predict/", "datadir": "data/", "dataset": "jet", "target": "is_signal", "add_beams": False, "beam_mass": 1, "force_download": False, "cuda": True, "dtype": "float", "num_workers": 0, "pmu_in": False, "num_cg_levels": 3, "mlp_depth": 3, "mlp_width": 2, "maxdim": [3], "max_zf": [1], "num_channels": [2, 3, 4, 3], "level_gain": [1.0], "cutoff_type": ["learn"], "num_basis_fn": 10, "scale": 0.005, "full_scalars": False, "mlp": True, "activation": "leakyrelu", "weight_init": "randn", "input": "linear", "num_mpnn_levels": 1, "top": "linear", "gaussian_mask": False,
# 'patience': 100, 'outpath': 'trained_models/', 'train': True, 'load':False, 'test': True})
#
# # load the data and cast it as a Pytorch dataloader
# args, torch_datasets = initialize_datasets(args, datadir='../data', num_pts=None)
#
# train_loader, test_loader, valid_loader = data_to_loader(args, torch_datasets)
#
# # Initialize model
# model = LGNTopTag(maxdim=args.maxdim, max_zf=args.max_zf, num_cg_levels=args.num_cg_levels, num_channels=args.num_channels, weight_init=args.weight_init, level_gain=args.level_gain, num_basis_fn=args.num_basis_fn,
#                   top=args.top, input=args.input, num_mpnn_layers=args.num_mpnn_levels, activation=args.activation, pmu_in=args.pmu_in, add_beams=args.add_beams,
#                   scale=1., full_scalars=args.full_scalars, mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
#                   device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), dtype=torch.float)
#
# Evaluate(args, model, test_loader, '.')
