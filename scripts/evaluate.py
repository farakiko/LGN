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
from lgn.models.lgn_toptag import LGNTopTag

import json
import pickle

from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import math

import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
# import mplhep as hep
# plt.style.use(hep.style.ROOT)


# given a model, evaluate it on test data: (1) plot roc curves (2) plot confusion matrix
def Evaluate(args, model, epoch, test_loader, outpath):

    t0 = time.time()

    model.eval()

    preds = []
    targets = []
    c=0
    acc=0

    for i, batch in enumerate(test_loader):
        pred = model(batch)

        preds.append(pred.detach().cpu().numpy())
        targets.append(batch['is_signal'].detach().cpu().numpy())

        c = c + (preds[i].argmax(axis=1) == targets[i]).sum().item()
        acc = 100*c/(args.batch_size_test*len(test_loader))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # store the accuracy
    with open(outpath + '/test_acc_epoch_' + str(epoch+1) + '.pkl', 'wb') as f:
        pickle.dump(acc, f)

    # # make confusion matrix plots
    # confusion_matrix = generate_confusion_matrix(torch.as_tensor(preds).argmax(axis=1), torch.as_tensor(targets), num_classes=2)
    # plot_confusion_matrix(confusion_matrix, epoch, savepath=outpath, format='png')
    #
    # with open(outpath + '/confusion_matrix_epoch_' + str(epoch+1) + '.pkl', 'wb') as f:
    #     pickle.dump(confusion_matrix, f)

    # make confusion matrix plots
    confusion_matrix = sklearn.metrics.confusion_matrix(targets, preds.argmax(axis=1), normalize='true')
    confusion_matrix[[0, 1],:] = confusion_matrix[[1, 0],:]  # swap rows for better visualization of confusion matrix

    plot_confusion_matrix(confusion_matrix, epoch, outpath)

    with open(outpath + '/confusion_matrix_epoch_' + str(epoch+1) + '.pkl', 'wb') as f:
        pickle.dump(confusion_matrix, f)

    # make ROC curves
    fpr_gnn, tpr_gnn, threshold_gnn = roc_curve(targets, preds[:,1])
    with open(outpath + '/Roc_curves_epoch_' + str(epoch+1) + '.pkl', 'wb') as f:
        pickle.dump(confusion_matrix, f)

    fig, ax = plt.subplots()
    ax.plot(tpr_gnn, 1/fpr_gnn, lw=2.5, label="GNN, AUC = {:.1f}%".format(auc(fpr_gnn,tpr_gnn)*100))
    ax.set_title("Roc curves at Epoch" + str(epoch+1))
    ax.set_xlabel(r'Signal efficiency "TPR"')
    ax.set_ylabel(r'Background rejection "1/FPR"')
    ax.legend(loc='upper left')
    plt.savefig(outpath + '/Roc_curves_epoch_' + str(epoch+1) + '.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(tpr_gnn, 1/fpr_gnn, lw=2.5, label="GNN, AUC = {:.1f}%".format(auc(fpr_gnn,tpr_gnn)*100))
    ax.set_title("Roc curves at Epoch" + str(epoch+1))
    ax.set_xlabel(r'Signal efficiency "TPR"')
    ax.set_ylabel(r'Background rejection "log(1/FPR)"')
    ax.semilogy()
    ax.legend(loc='upper left')
    plt.savefig(outpath + '/Roc_curves_log_epoch_' + str(epoch+1) + '.png')
    plt.close(fig)

    t1 = time.time()
    print("Time it took testing epoch", epoch+1, "is:", round((t1-t0)/60,2), "min")

    return acc


def plot_confusion_matrix(confusion_matrix, epoch, outpath):
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, ax = ax) #annot=True to annotate cells
    ax.set_title('Confusion Matrix at Epoch' + str(epoch+1))
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['bkg', 'sig'])
    ax.yaxis.set_ticklabels(['sig', 'bkg'])
    plt.savefig(outpath + '/confusion_matrix_epoch_' + str(epoch+1) + '.png')
    plt.close(fig)


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
# PATH = args.outpath + '/LGNTopTag_model#one_epoch_batch32/epoch_0_weights.pth'
# model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
#
# outpath = args.outpath + '/LGNTopTag_model#one_epoch_batch32'
#
# Evaluate(args, model, 0, test_loader, outpath)
#
#
# # with open('trained_models/LGNTopTag_model#one_epoch_batch32/confusion_matrix_epoch_1.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
# #     f = pickle.load(f)
# #
# # plot_confusion_matrix(f, 0, outpath)
