#!/usr/bin/env python3
"""
Author : linhtran <linhtran@localhost>
Date   : 2021-10-23
Purpose: Use GridsearchCV to find optimal MLPR hyperparameters
on dadi-generated data
"""

import argparse
import sys
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


def tuple_of_pos_int(input_str):
    """
    Custom type check for hidden layer sizes input from command line.
    Convert comma-separated string input to tuples of pos int."""

    single_tup_int = map(int, input_str.split(','))
    if any(layer_size < 1 for layer_size in single_tup_int):
        raise argparse.ArgumentTypeError(
            f"invalid tuple_of_pos_int value: '{input_str}'")
    return tuple(single_tup_int)


# --------------------------------------------------
def int_kfold(input_int):
    """
    Check if k-fold input is an integer > 2."""
    if int(input_int) < 2:
        raise argparse.ArgumentTypeError("k-fold input must be > 2")
    return int(input_int)


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Perform GridsearchCV to search for MLP hyperparameter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data',
                        metavar='input pickle file',
                        help='Data dictionary to use for hyperparam search',
                        type=argparse.FileType('rt'))

    parser.add_argument('-hls',
                        '--hidden_layer_sizes',
                        nargs='*',
                        metavar='TUPLE(S) OF POSITIVE INT',
                        type=tuple_of_pos_int,
                        action='store',
                        dest='hidden_layer_sizes',
                        help='use commas to separate layers',
                        default=[(100,)])

    parser.add_argument('-a',
                        '--activation',
                        nargs='*',
                        metavar='NAME',
                        action='store',
                        dest='activation',
                        choices=['identity', 'logistic', 'tanh', 'relu'],
                        help='options: identity, logistic, tanh, relu',
                        default=['relu'])

    parser.add_argument('-s',
                        '--solver',
                        nargs='*',
                        metavar='NAME',
                        action='store',
                        dest='solver',
                        choices=['lbfgs', 'sgd', 'adam'],
                        help='options: lbfgs, sgd, adam',
                        default=['adam'])

    parser.add_argument('-lr',
                        '--learning_rate',
                        nargs='*',
                        metavar='NAME',
                        action='store',
                        dest='learning_rate',
                        choices=['constant', 'invscaling', 'adaptive'],
                        help='options: constant, invscaling, adaptive')

    parser.add_argument('-mi',
                        '--max_iter',
                        nargs='*',
                        type=int,
                        action='store',
                        dest='max_iter',
                        help='Maximum number of iterations',
                        default=[500])

    parser.add_argument('-l2',
                        '--alpha',
                        nargs='*',
                        type=float,
                        action='store',
                        dest='alpha',
                        help='L2 penalty regularization param')

    parser.add_argument('-es',
                        '--early_stopping',
                        metavar='BOOLEAN',
                        type=bool,
                        action='store',
                        dest='early_stopping',
                        nargs='*',
                        help='Whether to use early stopping')

    parser.add_argument('-t',
                        '--tol',
                        type=float,
                        action='store',
                        dest='tol',
                        help='tolerance for optimization with early stopping')

    parser.add_argument('-n',
                        '--n_iter_no_change',
                        type=int,
                        action='store',
                        dest='n_iter_no_change',
                        help='Max n epochs to not meet tol improvement')

    parser.add_argument('-v',
                        '--verbose',
                        type=int,
                        help='Level of GridsearchCV Verbose',
                        choices=[0, 1, 2, 3, 4],
                        default=0)

    parser.add_argument('-cv',
                        '--cross_val',
                        type=int_kfold,
                        help='k-fold cross validation, default None=5')

    parser.add_argument('-o',
                        '--outfile',
                        help='Output filename',
                        metavar='FILE',
                        type=argparse.FileType('wt'),
                        default=sys.stdout)

    return parser.parse_args()


# --------------------------------------------------
def main():
    """Main program"""

    args = get_args()

    # Load training data
    with open(args.data.name, 'rb') as train_dict_fh:
        train_dict = pickle.load(train_dict_fh)
    print(f'\nData file used is {args.data.name}',  file=args.outfile)

    # nice to have: check if pickle file is a dictionary
    # with float numpy arrays for regressions

    # process training data into features and labels
    train_features = [np.array(train_dict[params]).flatten()
                      for params in train_dict]
    train_labels = list(train_dict)

    # process input from command line into a dictionary of params
    param_dict = {}
    for arg in vars(args):
        if arg not in ['data', 'outfile', 'cross_val',
                       'verbose'] and getattr(args, arg) is not None:
            param_dict[arg] = getattr(args, arg)
            print(arg, ':', getattr(args, arg), file=args.outfile)

    # get the total number of models being tested from param_dict
    n_models = 1
    for hyperparam in param_dict.values():
        n_models *= len(hyperparam)
    print(f"\nNumber of models tested: {n_models}\n", file=args.outfile)

    # Specify the ML models to be optimized
    mlpr = MLPRegressor()

    # perform grid search using selected ML model, data, and params
    cv = GridSearchCV(mlpr, param_dict, n_jobs=-1,
                      verbose=args.verbose, cv=args.cross_val)
    cv.fit(train_features, train_labels)
    results = cv.cv_results_

    # process results for printing ranked models
    for i in range(1, n_models + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print(f'Model with rank: {i}', file=args.outfile)
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]),
                  file=args.outfile)
            print(f"Parameters: {results['params'][candidate]}",
                  file=args.outfile)


# --------------------------------------------------
if __name__ == '__main__':
    main()
