import torch
import sys
sys.path.insert(1, 'data_processing/')
sys.path.insert(1, 'lgn/')

import args
from args import setup_argparse

from data_processing import make_pytorch_data
from make_pytorch_data import initialize_datasets
from make_pytorch_data import data_to_loader
from lgn.models.lgn_toptag import LGNTopTag

def main(args):

    args, torch_datasets = initialize_datasets(args, datadir='../data', num_pts=None)

    train_loader, test_loader, valid_loader = data_to_loader(args, torch_datasets)

    device = torch.device('cpu')

    if args.dtype == 'double':
        dtype = torch.double
    elif args.dtype == 'float':
        dtype = torch.float

    # # Initialize model
    # model = LGNTopTag(args.maxdim, args.max_zf, args.num_cg_levels, args.num_channels,
    #                   args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
    #                   args.weight_init, args.level_gain, args.num_basis_fn,
    #                   args.top, args.input, args.num_mpnn_levels, activation=args.activation, pmu_in=args.pmu_in, add_beams=args.add_beams,
    #                   mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
    #                   scale=1., full_scalars=args.full_scalars,
    #                   device=device, dtype=dtype)

    return train_loader


if __name__ == "__main__":

    args = setup_argparse()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'num_epoch': 6, 'batch_size': 2, 'num_train': 1, 'num_test': 1, 'num_valid': 1, 'scale':1, 'nobj': None,
    #                    'shuffle': False, 'add_beams': False, 'beam_mass': 1, 'num_wrokers': 0,
    #                    'maxdim': [3], 'max_zf': [1], 'num_cg_levels': 3, 'num_channels': [2, 3, 4, 3],
    #                    'weight_init': 'randn', 'level_gain':[1.], 'num_basis_fn':10,
    #                    'top': 'linear', 'input': 'linear', 'num_mpnn_levels': 1,
    #                    'activation': 'leakyrelu', 'pmu_in': False, 'add_beams': True,
    #                    'mlp': True, 'mlp_depth': 3, 'mlp_width': 2, 'full_scalars': False})

    train_loader = main(args)

    print('finished')
