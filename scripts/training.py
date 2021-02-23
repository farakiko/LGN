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

@torch.no_grad()
def test(args, model, loader, epoch, device, outpath):
    with torch.no_grad():
        test_pred, acc = train(args, model, loader, None, None, epoch, device, outpath) # CONFUSION MATRIX
    return test_pred, acc

def train(args, model, loader, optimizer, lr, epoch, device, outpath):

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    avg_loss_per_epoch = []
    fractional_loss = []
    c = 0
    acc = 0

    for i, batch in enumerate(loader):
        t0 = time.time()

        # for better reading of the code
        X = batch
        Y = batch['is_signal'].to(device)

        # forwardprop
        preds = model(X)

        # backprop
        loss = nn.CrossEntropyLoss()
        batch_loss = loss(preds, Y.long())

        if is_train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        t1 = time.time()

        # to get some accuracy measure
        c = c + (preds.argmax(axis=1) == Y).sum().item()
        acc = 100*c/(args.batch_size*len(loader))

        if is_train:
            print('batch={}/{} train_loss={:.2f} train_acc={:.1f} dt={:.1f}s'.format(i+1, len(loader), batch_loss.item(), acc, t1-t0), end='\r', flush=True)
        else:
            print('batch={}/{} valid_loss={:.2f} valid_acc={:.1f} dt={:.1f}s'.format(i+1, len(loader), batch_loss.item(), acc, t1-t0), end='\r', flush=True)

        avg_loss_per_epoch.append(batch_loss.item())

        # added to attempt plotting over a fraction of an epoch
        if (i % math.floor(0.1*len(loader)))==0 :
            fractional_loss.append(sum(avg_loss_per_epoch)/len(avg_loss_per_epoch))

    avg_loss_per_epoch = sum(avg_loss_per_epoch)/len(avg_loss_per_epoch)

    if is_train:
        fig, ax = plt.subplots()
        ax.plot(range(len(fractional_loss)), fractional_loss, label='fractional loss train')
        ax.set_xlabel('Fraction of Epoch' + str(epoch+1) + ' completed (% epoch)')
        ax.set_ylabel('Loss')
        ax.legend(loc='best')
        plt.savefig(outpath + '/fractional_loss_train_epoch_' + str(epoch+1) + '.png')
        plt.close(fig)

        with open(outpath + '/fractional_loss_train_epoch_' + str(epoch+1) + '.pkl', 'wb') as f:
            pickle.dump(fractional_loss, f)
        with open(outpath + '/train_acc_epoch_' + str(epoch+1) + '.pkl', 'wb') as f:
            pickle.dump(acc, f)

    else:
        fig, ax = plt.subplots()
        ax.plot(range(len(fractional_loss)), fractional_loss, label='fractional loss test')
        ax.set_xlabel('Fraction of Epoch' + str(epoch+1) + ' completed (% epoch)')
        ax.set_ylabel('Loss')
        ax.legend(loc='best')
        plt.savefig(outpath + '/fractional_loss_valid_epoch_' + str(epoch+1) + '.png')
        plt.close(fig)

        with open(outpath + '/fractional_loss_valid_epoch_' + str(epoch+1) + '.pkl', 'wb') as f:
            pickle.dump(fractional_loss, f)
        with open(outpath + '/valid_acc_epoch_' + str(epoch+1) + '.pkl', 'wb') as f:
            pickle.dump(acc, f)

    return avg_loss_per_epoch, acc


def train_loop(args, model, optimizer, outpath, train_loader, valid_loader, device):
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
        train_loss, train_acc = train(args, model, train_loader, optimizer, args.lr_init, epoch, device, outpath)
        losses_train.append(train_loss)

        # test generalization of the model
        model.eval()
        valid_loss, valid_acc = test(args, model, valid_loader, epoch, device, outpath)
        losses_valid.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            stale_epochs = 0
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = args.num_epoch - (epoch+1)
        time_per_epoch = (t1 - t0_initial)/(epoch + 1)

        eta = epochs_remaining*time_per_epoch/60

        torch.save(model.state_dict(), "{0}/epoch_{1}_weights.pth".format(outpath, epoch+1))

        print("epoch={}/{} dt={:.2f}s train_loss={:.5f} valid_loss={:.5f} train_acc={:.5f} valid_acc={:.5f} stale={} eta={:.1f}m".format(
            epoch+1, args.num_epoch,
            t1 - t0, train_loss, valid_loss, train_acc, valid_acc,
            stale_epochs, eta))

    fig, ax = plt.subplots()
    ax.plot(range(len(losses_train)), losses_train, label='train loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    plt.savefig(outpath + '/loss_train.png')
    plt.close(fig)

    with open(outpath + '/loss_train.pkl', 'wb') as f:
        pickle.dump(losses_train, f)

    fig, ax = plt.subplots()
    ax.plot(range(len(losses_valid)), losses_valid, label='test loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    plt.savefig(outpath + '/loss_valid.png')
    plt.close(fig)

    with open(outpath + '/loss_valid.pkl', 'wb') as f:
        pickle.dump(losses_valid, f)
