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


if __name__ == "__main__":

    args = setup_argparse()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'num_epoch': 6, 'batch_size': 3, 'num_train': 4, 'num_test': 1, 'num_valid': 1, 'scale':1, 'nobj': None,
    #                    'shuffle': False, 'add_beams': False, 'beam_mass': 1, 'num_workers': 0,
    #                    'maxdim': [3], 'max_zf': [1], 'num_cg_levels': 3, 'num_channels': [2, 3, 4, 3],
    #                    'weight_init': 'randn', 'level_gain':[1.], 'num_basis_fn':10,
    #                    'top': 'linear', 'input': 'linear', 'num_mpnn_levels': 1,
    #                    'activation': 'leakyrelu', 'pmu_in': False,
    #                    'mlp': True, 'mlp_depth': 3, 'mlp_width': 2, 'full_scalars': False, 'dtype': 'float'})

    device = torch.device('cpu')

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

    # make a forward pass
    for batch in train_loader:
        pred = model(batch)
        break

    # print model prediction (which is a one-hot encoded 2D vector corresponding to the 2 classes)
    # note: batch_size=3, that's why pred.shape=(3,2)
    print('A quick prediction of the class of 1 batch containing 3 jets is:')
    print(pred)
