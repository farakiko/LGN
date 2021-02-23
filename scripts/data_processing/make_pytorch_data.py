import torch
import numpy as np
from math import sqrt
import os, h5py, glob
import sys
sys.path.insert(1, '../')

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from collate import collate_fn
from jetdatasets import JetDataset

# this script makes use of the other two scripts in data_processing: collate & jetdatasets; to do the following:
# (1) load the data from .h5 files and cast them as Pytorch Datasets
# (2) generate Pytorch loaders from the Pytorch Datasets

def initialize_datasets(args, datadir='../../data', num_pts=None):
    """
    Initialize datasets.
    """

    ### ------ 1: Get the file names ------ ###
    # datadir should be the directory in which the HDF5 files (e.g. out_test.h5, out_train.h5, out_valid.h5) reside.
    # There may be many data files, in some cases the test/train/validate sets may themselves be split across files.
    # We will look for the keywords defined in splits to be be in the filenames, and will thus determine what
    # set each file belongs to.
    splits = ['train', 'test', 'valid'] # We will consider all HDF5 files in datadir with one of these keywords in the filename
    files = glob.glob(datadir + '/*.h5')
    datafiles = {split:[] for split in splits}
    for file in files:
        for split in splits:
            if split in file: datafiles[split].append(file)
    nfiles = {split:len(datafiles[split]) for split in splits}

    ### ------ 2: Set the number of data points ------ ###
    # There will be a JetDataset for each file, so we divide number of data points by number of files,
    # to get data points per file. (Integer division -> must be careful!) #TODO: nfiles > npoints might cause issues down the line, but it's an absurd use case
    if num_pts is None:
        num_pts={'train':args.num_train,'test':args.num_test,'valid':args.num_valid}

    num_pts_per_file = {}
    for split in splits:
        num_pts_per_file[split] = []

        if num_pts[split] == -1:
            for n in range(nfiles[split]): num_pts_per_file[split].append(-1)
        else:
            for n in range(nfiles[split]): num_pts_per_file[split].append(int(np.ceil(num_pts[split]/nfiles[split])))
            num_pts_per_file[split][-1] = int(np.maximum(num_pts[split] - np.sum(np.array(num_pts_per_file[split])[0:-1]),0))

    ### ------ 3: Load the data ------ ###
    datasets = {}
    for split in splits:
        datasets[split] = []
        for file in datafiles[split]:
            with h5py.File(file,'r') as f:
                datasets[split].append({key: torch.from_numpy(val[:]) for key, val in f.items()})

    ### ------ 4: Error checking ------ ###
    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = []
    for split in splits:
        for dataset in datasets[split]:
            keys.append(dataset.keys())
    assert all([key == keys[0] for key in keys]), 'Datasets must have same set of keys!'

    ### ------ 5: Initialize datasets ------ ###
    # Now initialize datasets based upon loaded data
    torch_datasets = {split: ConcatDataset([JetDataset(data, num_pts=num_pts_per_file[split][idx]) for idx, data in enumerate(datasets[split])]) for split in splits}

    # Now, update the number of training/test/validation sets in args
    args.num_train = torch_datasets['train'].cumulative_sizes[-1]
    args.num_test = torch_datasets['test'].cumulative_sizes[-1]
    args.num_valid = torch_datasets['valid'].cumulative_sizes[-1]

    return args, torch_datasets


def data_to_loader(args, torch_datasets):

    # use collate_fn to construct some: atom_mask and edge_mask
    train_loader = DataLoader(torch_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_loader.collate_fn = collate_fn

    test_loader = DataLoader(torch_datasets['test'], batch_size=args.batch_size_test, shuffle=True, num_workers=args.num_workers)
    test_loader.collate_fn = collate_fn

    valid_loader = DataLoader(torch_datasets['valid'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader.collate_fn = collate_fn

    return train_loader, test_loader, valid_loader


# #-----------------------------------------------------------------------------------------------------
# # test the data loader to get familiar with the data that we feed to the model:
# class objectview(object):
#     def __init__(self, d):
#         self.__dict__ = d
#
# args = objectview({'num_epoch': 6, 'batch_size': 2, 'num_train': 4, 'num_test': 1, 'num_valid': 1, 'scale':1, 'nobj':None,
#                     'shuffle':False, 'add_beams':False, 'beam_mass':1, 'num_workers': 0})
#
# args, torch_datasets = initialize_datasets(args, datadir='../../data', num_pts=None)
#
# train_loader, test_loader, valid_loader = data_to_loader(args, torch_datasets)
#
# for batch in train_loader:
#     break
#
# # Each jet is essentially a dictionary .. each batch contains a set of jets
# print(batch.keys())
# print(batch['Nobj'].shape)
# print(batch['Pmu'].shape)
# print(batch['is_signal'])
#
# # Nobj: length=2 signifies 2 jets because batch_size=2.. the 2 elements stored in Nobj are the actual # of tracks in each jet (ranges from 1 to 200)
# # Pmu: p4 of each track (capped at 200 because there by construction there are at most 200 tracks in a jet )
# # jet_pt: total jet momentum
# # mass: mass of each track
# # truth_Pmu: p4 of jet
# # is_signal: binary (0 or 1)
# # label: 1 for actual track, 0 for zero-padded track
