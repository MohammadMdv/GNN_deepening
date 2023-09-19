import gc

import numpy as np
import torch
import os

from options.base_options import BaseOptions
from trainer import trainer
from utils import set_seed, print_args, overwrite_with_yaml
# importing run_GNN.py from graph-neural-networks directory
import sys
import subprocess

sys.path.append('./graph-neural-pde/src/')


def main(args):
    list_test_acc = []
    list_valid_acc = []
    list_train_loss = []
    if args.compare_model:
        args = overwrite_with_yaml(args, args.type_model, args.dataset)
    print_args(args)

    test_accs = None

    for seed in range(args.N_exp):
        print(f'seed (which_run) = <{seed}>')
        args.random_seed = seed
        set_seed(args)
        torch.cuda.empty_cache()
        trnr = trainer(args, seed)
        train_loss, valid_acc, test_acc, test_accs = trnr.train_and_test()
        list_test_acc.append(test_acc)
        list_valid_acc.append(valid_acc)
        list_train_loss.append(train_loss)

        del trnr
        torch.cuda.empty_cache()
        gc.collect()

        # record training data
        print('mean and std of test acc: {:.4f}±{:.4f}'.format(
            np.mean(list_test_acc), np.std(list_test_acc)))

    print('final mean and std of test acc with <{}> runs: {:.4f}±{:.4f}'.format(
        args.N_exp, np.mean(list_test_acc), np.std(list_test_acc)))

    # Saving test_accs into a file with the models name
    if not os.path.exists('./results'):
        os.makedirs('./results')
    with open(f'./results/{args.type_model}_{args.dataset}_test_accs.txt', 'w') as f:
        for item in test_accs:
            f.write("%s\n" % item)


if __name__ == "__main__":
    args = BaseOptions().initialize()

    # Running the main function for each model (GCN, GAT, SGC, GCNII, DAGNN, GPRGNN, APPNP, JKNet, DeeperGCN) separately
    # for model in ['GCN', 'GAT', 'SGC', 'GCNII', 'DAGNN', 'GPRGNN', 'APPNP', 'JKNet', 'DeeperGCN']:
    #     args.type_model = model
    #     args.epoch = 1000
    #     print(f'Using cuda: {args.cuda}')
    #     print("Using model: ", args.type_model)
    #     main(args)

    # args.dataset = "TEXAS"
    # main(args)
    # args.model = "GCNII"
    # args.type_model = "GCNII"
    # args.dataset = "ACTOR"
    # main(args)

    model_name = input("Type the model name: ")
    dataset_name = input("Type the dataset name: ")
    if model_name == "GRAND":
        subprocess.call(" python graph-neural-pde/src/run_GNN.py --dataset " + dataset_name, shell=True)
    else:
        args.type_model = model_name
        args.dataset = dataset_name
        main(args)
