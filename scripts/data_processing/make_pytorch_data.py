import torch
import numpy as np
from math import sqrt
import os, h5py, glob
import sys
sys.path.insert(1, '../')

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

from collate import collate_fn
from jetdatasets import JetDataset

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
    data_train = DataLoader(torch_datasets['train'], args.batch_size, pin_memory=True, shuffle=True)
    data_train.collate_fn = collate_fn
    data_test = DataListLoader(torch_datasets['test'], args.batch_size, pin_memory=True, shuffle=True)
    data_test.collate_fn = collate_fn
    data_valid = DataListLoader(torch_datasets['valid'], args.batch_size, pin_memory=True, shuffle=True)
    data_valid.collate_fn = collate_fn

    # initialize some values to start constructing the Data objects
    d=[]
    batch_data_train = []
    batch_data_test = []
    batch_data_valid = []
    Nobj=[]
    Pmu=[]
    is_signal=[]
    jet_pt=[]
    label=[]
    mass=[]
    truth_Pmu=[]
    atom_mask=[]
    edge_mask=[]

    # Casting the train_dataset as a list of pytorch Data objects
    for i,data in enumerate(data_train):
        Nobj.append(data['Nobj'][0].clone().detach())
        Pmu.append(data['Pmu'][0].clone().detach())
        is_signal.append(data['is_signal'][0].clone().detach())
        jet_pt.append(data['jet_pt'][0].clone().detach())
        label.append(data['label'][0].clone().detach())
        mass.append(data['mass'][0].clone().detach())
        truth_Pmu.append(data['truth_Pmu'][0].clone().detach())
        atom_mask.append(data['atom_mask'][0].clone().detach())
        edge_mask.append(data['edge_mask'][0].clone().detach())

        d = Data(
            Nobj=Nobj[i],
            Pmu=Pmu[i],
            is_signal=is_signal[i],
            jet_pt=jet_pt[i],
            label=label[i],
            mass=mass[i],
            truth_Pmu=truth_Pmu[i],
            atom_mask=atom_mask[i],
            edge_mask=edge_mask[i]
        )

        batch_data_train.append(d)

    # Casting the test_dataset as a list of pytorch Data objects
    for i,data in enumerate(data_test):
        Nobj.append(data['Nobj'][0].clone().detach())
        Pmu.append(data['Pmu'][0].clone().detach())
        is_signal.append(data['is_signal'][0].clone().detach())
        jet_pt.append(data['jet_pt'][0].clone().detach())
        label.append(data['label'][0].clone().detach())
        mass.append(data['mass'][0].clone().detach())
        truth_Pmu.append(data['truth_Pmu'][0].clone().detach())
        atom_mask.append(data['atom_mask'][0].clone().detach())
        edge_mask.append(data['edge_mask'][0].clone().detach())

        d = Data(
            Nobj=Nobj[i],
            Pmu=Pmu[i],
            is_signal=is_signal[i],
            jet_pt=jet_pt[i],
            label=label[i],
            mass=mass[i],
            truth_Pmu=truth_Pmu[i],
            atom_mask=atom_mask[i],
            edge_mask=edge_mask[i]
        )

        batch_data_test.append(d)

    # Casting the valid_dataset as a list of pytorch Data objects
    for i,data in enumerate(data_valid):
        Nobj.append(data['Nobj'][0].clone().detach())
        Pmu.append(data['Pmu'][0].clone().detach())
        is_signal.append(data['is_signal'][0].clone().detach())
        jet_pt.append(data['jet_pt'][0].clone().detach())
        label.append(data['label'][0].clone().detach())
        mass.append(data['mass'][0].clone().detach())
        truth_Pmu.append(data['truth_Pmu'][0].clone().detach())
        atom_mask.append(data['atom_mask'][0].clone().detach())
        edge_mask.append(data['edge_mask'][0].clone().detach())

        d = Data(
            Nobj=Nobj[i],
            Pmu=Pmu[i],
            is_signal=is_signal[i],
            jet_pt=jet_pt[i],
            label=label[i],
            mass=mass[i],
            truth_Pmu=truth_Pmu[i],
            atom_mask=atom_mask[i],
            edge_mask=edge_mask[i]
        )

        batch_data_valid.append(d)

    # defining a DataListLoader to iterate over the data during training
    train_dataset = torch.utils.data.Subset([batch_data_train], np.arange(start=0, stop=args.num_train))
    test_dataset = torch.utils.data.Subset([batch_data_test], np.arange(start=0, stop=args.num_test))
    valid_dataset = torch.utils.data.Subset([batch_data_valid], np.arange(start=0, stop=args.num_valid))

    def collate(items):
        l = sum(items, [])
        return Batch.from_data_list(l)

    train_loader = DataListLoader(train_dataset, args.batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate
    test_loader = DataListLoader(test_dataset, args.batch_size, pin_memory=True, shuffle=True)
    test_loader.collate_fn = collate
    valid_loader = DataListLoader(valid_dataset, args.batch_size, pin_memory=True, shuffle=True)
    valid_loader.collate_fn = collate

    return train_loader, test_loader, valid_loader


#-----------------------------------------------------------------------------------------------------
# # test the data loader to get familiar with the data that we feed to the model:
# class objectview(object):
#     def __init__(self, d):
#         self.__dict__ = d
#
# args = objectview({'num_epoch': 6, 'batch_size': 2, 'num_train': 1, 'num_test': 1, 'num_valid': 1, 'scale':1, 'nobj':None,
#                     'shuffle':False, 'add_beams':False, 'beam_mass':1, 'num_wrokers': 0})
#
# args, torch_datasets = initialize_datasets(args, datadir='../../data', num_pts=None)
#
# train_loader, test_loader, valid_loader = data_to_loader(args, torch_datasets)
#
# next(iter(train_loader))
## Batch(Nobj=[1], Pmu=[200, 4], atom_mask=[200], edge_mask=[200, 200], is_signal=[1], jet_pt=[1], label=[200], mass=[200], truth_Pmu=[4])
## Nobj: # of actual tracks (ranges from 1 to 200)
## Pmu: p4 of each track
## jet_pt: total jet momentum
## mass: mass of each track
## truth_Pmu: p4 of jet
## is_signal: binary (0 or 1)
## label: 1 for actual track, 0 for zero-padded track
