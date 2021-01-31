import sys
sys.path.insert(1, 'data_processing/')

import args
from args import setup_argparse

from data_processing import make_pytorch_data
from make_pytorch_data import initialize_datasets
from make_pytorch_data import data_to_loader


def main(args):

    args, torch_datasets = initialize_datasets(args, datadir='../data', num_pts=None)

    train_loader, test_loader, valid_loader = data_to_loader(args, torch_datasets)

    return train_loader


if __name__ == "__main__":

    #args = setup_argparse()

    # the next part initializes some args values (to run the script not from terminal)
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    args = objectview({'num_epoch': 6, 'batch_size': 2, 'num_train': 1, 'num_test': 1, 'num_valid': 1, 'scale':1, 'nobj':None,
                        'shuffle':False, 'add_beams':False, 'beam_mass':1, 'num_wrokers': 0})

    train_loader = main(args)

    print('finished')
