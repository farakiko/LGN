import torch
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

# Get a unique directory name for each trained model
def get_model_fname(dataset, model, n_train, lr):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}__npar_{}__cfg_{}__user_{}__ntrain_{}__lr_{}__{}'.format(
        model_name,
        dataset.split("/")[-1],
        model_params,
        model_cfghash,
        model_user,
        n_train,
        lr, int(time.time()))
    return model_fname

# Create the directory to store the weights/epoch for the trained models
def create_model_folder(args, model):
    if not osp.isdir(args.outpath):
        os.makedirs(args.outpath)

    model_fname = get_model_fname('model#', model, args.num_train, args.lr_init)
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

# create the training loop
@torch.no_grad()
def test(model, loader):
    with torch.no_grad():
        test_pred = train(model, loader, None, None)
    return test_pred

def train(model, loader, optimizer, lr):

    is_train = (optimizer is not None)

    if is_train:
        model.train()
    else:
        model.eval()

    avg_loss_per_epoch = []

    for i, batch in enumerate(loader):
        t0 = time.time()

        # for better reading of the code
        X = batch
        Y = batch['is_signal']

        # forwardprop
        preds = model(X)

        # backprop
        batch_loss = torch.nn.CrossEntropyLoss().cuda()(preds, Y.long())
        if is_train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        t1 = time.time()

        if is_train:
            print('{}/{} train_loss={:.2f} dt={:.1f}s'.format(i+1, len(loader), batch_loss.item(), t1-t0), end='\r', flush=True)
        else:
            print('{}/{} valid_loss={:.2f} dt={:.1f}s'.format(i+1, len(loader), batch_loss.item(), t1-t0), end='\r', flush=True)

        avg_loss_per_epoch.append(batch_loss.item())

        i += 1

    avg_loss_per_epoch = sum(avg_loss_per_epoch)/len(avg_loss_per_epoch)

    return avg_loss_per_epoch


def train_loop(args, model, optimizer, outpath):
    t0_initial = time.time()

    losses_train = []
    losses_valid = []

    best_valid_loss = 99999.9
    stale_epochs = 0

    print("Training over {} epochs".format(args.num_epoch))

    for epoch in range(args.num_epoch):
        t0 = time.time()

        if stale_epochs > args.patience:
            print("breaking due to stale epochs")
            break

        model.train()
        train_loss = train(model, train_loader, optimizer, args.lr_init)
        losses_train.append(train_loss)

        # test generalization of the model
        model.eval()
        valid_loss = test(model, valid_loader)
        losses_valid.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_val_loss = valid_loss
            stale_epochs = 0
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = args.num_epoch - epoch
        time_per_epoch = (t1 - t0_initial)/(epoch + 1)

        eta = epochs_remaining*time_per_epoch/60

        torch.save(model.state_dict(), "{0}/epoch_{1}_weights.pth".format(outpath, epoch))

        print("epoch={}/{} dt={:.2f}s train_loss={:.5f} valid_loss={:.5f} stale={} eta={:.1f}m".format(
            epoch, args.num_epoch,
            t1 - t0, train_loss, valid_loss,
            stale_epochs, eta))

    fig, ax = plt.subplots()
    ax.plot(range(len(losses_train)), losses_train, label='train loss')
    ax.plot(range(len(losses_valid)), losses_valid, label='test loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    plt.show()

#---------------------------------------------------------------------------------------------

if __name__ == "__main__":

    args = setup_argparse()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({"num_train": 4, "num_valid": 2, "num_test": 2, "task": "train", "num_epoch": 4, "batch_size": 2, "batch_group_size": 1, "weight_decay": 0, "cutoff_decay": 0, "lr_init": 0.001, "lr_final": 1e-05, "lr_decay": 9999, "lr_decay_type": "cos", "lr_minibatch": True, "sgd_restart": -1, "optim": "amsgrad", "parallel": False, "shuffle": True, "seed": 1, "alpha": 50, "save": True, "load": False, "test": True, "log_level": "info", "textlog": True, "predict": True, "quiet": True, "prefix": "nosave", "loadfile": "", "checkfile": "", "bestfile": "", "logfile": "", "predictfile": "", "workdir": "./", "logdir": "log/", "modeldir": "model/", "predictdir": "predict/", "datadir": "data/", "dataset": "jet", "target": "is_signal", "add_beams": False, "beam_mass": 1, "force_download": False, "cuda": True, "dtype": "float", "num_workers": 0, "pmu_in": False, "num_cg_levels": 3, "mlp_depth": 3, "mlp_width": 2, "maxdim": [3], "max_zf": [1], "num_channels": [2, 3, 4, 3], "level_gain": [1.0], "cutoff_type": ["learn"], "num_basis_fn": 10, "scale": 0.005, "full_scalars": False, "mlp": True, "activation": "leakyrelu", "weight_init": "randn", "input": "linear", "num_mpnn_levels": 1, "top": "linear", "gaussian_mask": False, 'patience': 100, 'outpath': 'trained_models/'})

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

    # get directory name for the model to train
    outpath = create_model_folder(args, model)

    # define an optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr_init)

    # start the training loop
    train_loop(args, model, optimizer, outpath)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


device





#print("GPU tagger is:", torch.cuda.current_device())
#print("GPU model:", torch.cuda.get_device_name(0))


print(device)
