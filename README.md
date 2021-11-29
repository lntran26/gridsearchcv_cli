# BE534 Fall 2021 Final Project

## Scikit-learn GridSearchCV Command Line Interface (CLI) for Multi-layer Perceptron Regressor and dadi

This program provides a command line interface for exhaustive search over specified hyperparameter values for the [scikit-learn](https://scikit-learn.org/stable/index.html) [Multi-layer Perceptron Regressor (MLPR)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlpr) training on [dadi](https://dadi.readthedocs.io/en/latest/) demographic model's site frequency spectrum (SFS) data. It implements the scikit-learn method [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV) and allow users to input a wide variety of hyperparameter options from the command line and output the rankings of all models' performance from best to worst.

## Requirements
1. scikit-learn 0.24.1
2. dadi
3. numpy

To install the dependencies, run `make install`.

## Usage
The program accepts as inputs:
1. a pickle data file containing dadi-simulated SFS data as training features and the demographic parameters used to generate the SFS as training labels.
2. different hyperparameter values for the Multi-layer Perceptron Regressor, including choices of hidden layer size, activation function, optimizer algorithm, learning rate, number of max iteration, L2 regularization value, whether or not to implement early stopping, tolerance value (if implementing early stopping), number of iterations allowed with no improvement (if implementing early stopping), number of folds for cross-validation.
3. level of verbose for printing out output as the cross-validation runs.
4. specified path for output file (print to stdout by default).

When run with no arguments, the program will produce a brief usage:

```
$ ./gridsearch.py
usage: gridsearch.py [-h] [-hls [TUPLES OF POSITIVE INT ...]] [-a [NAME ...]] [-s [NAME ...]]
                     [-lr [NAME ...]] [-mi [MAX_ITER ...]] [-l2 [ALPHA ...]] [-es [BOOLEAN ...]] [-t TOL]
                     [-n N_ITER_NO_CHANGE] [-v {0,1,2,3,4}] [-cv CROSS_VAL] [-o FILE]
                     input pickle file
gridsearch.py: error: the following arguments are required: input pickle file
```

When run with the `-h|--help` flag, it will produce a longer help document:

```
$ ./gridsearch.py -h
usage: gridsearch.py [-h] [-hls [TUPLES OF POSITIVE INT ...]] [-a [NAME ...]] [-s [NAME ...]]
                     [-lr [NAME ...]] [-mi [MAX_ITER ...]] [-l2 [ALPHA ...]] [-es [BOOLEAN ...]] [-t TOL]
                     [-n N_ITER_NO_CHANGE] [-v {0,1,2,3,4}] [-cv CROSS_VAL] [-o FILE]
                     input pickle file

Perform GridsearchCV to search for MLP hyperparameter

positional arguments:
  input pickle file     Data dictionary to use for hyperparam search

optional arguments:
  -h, --help            show this help message and exit
  -hls [TUPLE(S) OF POSITIVE INT ...], --hidden_layer_sizes [TUPLE(S) OF POSITIVE INT ...]
                        use commas to separate layers (default: [(100,)])
  -a [NAME ...], --activation [NAME ...]
                        options: identity, logistic, tanh, relu (default: ['relu'])
  -s [NAME ...], --solver [NAME ...]
                        options: lbfgs, sgd, adam (default: ['adam'])
  -lr [NAME ...], --learning_rate [NAME ...]
                        options: constant, invscaling, adaptive (default: None)
  -mi [MAX_ITER ...], --max_iter [MAX_ITER ...]
                        Maximum number of iterations (default: [500])
  -l2 [ALPHA ...], --alpha [ALPHA ...]
                        L2 penalty regularization param (default: None)
  -es [BOOLEAN ...], --early_stopping [BOOLEAN ...]
                        Whether to use early stopping (default: None)
  -t TOL, --tol TOL     tolerance for optimization with early stopping (default: None)
  -n N_ITER_NO_CHANGE, --n_iter_no_change N_ITER_NO_CHANGE
                        Max n epochs to not meet tol improvement (default: None)
  -v {0,1,2,3,4}, --verbose {0,1,2,3,4}
                        Level of GridsearchCV Verbose (default: 0)
  -cv CROSS_VAL, --cross_val CROSS_VAL
                        k-fold cross validation, default None=5 (default: None)
  -o FILE, --outfile FILE
                        Output filename (default: <_io.TextIOWrapper name='<stdout>' mode='w'
                        encoding='utf-8'>)
```

The output from the program provides the rankings of MLPR models created from all combinations of input hyperparameter values.
For instance, the result for running with the default model on a toy data set with 100 examples (train_data_100) is:
```
$ ./gridsearch.py inputs/train_data_100

Data file used is inputs/train_data_100
hidden_layer_sizes : [(100,)]
activation : ['relu']
solver : ['adam']
max_iter : [500]

Number of models tested: 1

Model with rank: 1
Mean validation score: 0.441 (std: 0.033)
Parameters: {'activation': 'relu', 'hidden_layer_sizes': (100,), 'max_iter': 500, 'solver': 'adam'}
```

The program will print out the data used and the specification of the model tested.

We can test different model configurations to see if we can get a better score with different hidden layer size architectures with the `-hls|--hidden_layer_sizes` flag:

```
$ ./gridsearch.py inputs/train_data_100 -hls 100 50,50 25,25,25,25

Data file used is inputs/train_data_100
hidden_layer_sizes : [(100,), (50, 50), (25, 25, 25, 25)]
activation : ['relu']
solver : ['adam']
max_iter : [500]

Number of models tested: 3

Model with rank: 1
Mean validation score: 0.446 (std: 0.022)
Parameters: {'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'max_iter': 500, 'solver': 'adam'}

Model with rank: 2
Mean validation score: 0.437 (std: 0.041)
Parameters: {'activation': 'relu', 'hidden_layer_sizes': (100,), 'max_iter': 500, 'solver': 'adam'}

Model with rank: 3
Mean validation score: 0.429 (std: 0.023)
Parameters: {'activation': 'relu', 'hidden_layer_sizes': (25, 25, 25, 25), 'max_iter': 500, 'solver': 'adam'}
```

We can also test different activation function and optimizer (solver) options with the `-a|--activation` and `-s|--solver` flags. Note that in this case we may have to increase the max iteration parameter with the `-mi|--max_iter` flag to avoid getting non-convergence warning.

```
$ ./gridsearch.py inputs/train_data_100 -hls 50,50 -a relu tanh -s lbfgs adam -mi 10000

Data file used is inputs/train_data_100
hidden_layer_sizes : [(50, 50)]
activation : ['relu', 'tanh']
solver : ['lbfgs', 'adam']
max_iter : [10000]

Number of models tested: 4

Model with rank: 1
Mean validation score: 0.635 (std: 0.353)
Parameters: {'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'max_iter': 10000, 'solver': 'lbfgs'}

Model with rank: 2
Mean validation score: 0.449 (std: 0.022)
Parameters: {'activation': 'tanh', 'hidden_layer_sizes': (50, 50), 'max_iter': 10000, 'solver': 'adam'}

Model with rank: 3
Mean validation score: 0.445 (std: 0.034)
Parameters: {'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'max_iter': 10000, 'solver': 'adam'}

Model with rank: 4
Mean validation score: 0.062 (std: 1.353)
Parameters: {'activation': 'tanh', 'hidden_layer_sizes': (50, 50), 'max_iter': 10000, 'solver': 'lbfgs'}
```

Verbose mode (levels: 0-4) with the `-v|--verbose` flag can be used to tune the amount of information printed out for each run:

```
$ ./gridsearch.py inputs/train_data_1000 -hls 50,50 -s lbfgs  -mi 10000 -v 4

Data file used is inputs/train_data_1000
hidden_layer_sizes : [(50, 50)]
activation : ['relu']
solver : ['lbfgs']
max_iter : [10000]

Number of models tested: 1

Fitting 5 folds for each of 1 candidates, totalling 5 fits
[CV 1/5] END activation=relu, hidden_layer_sizes=(50, 50), max_iter=10000, solver=lbfgs; total time=   4.5s
[CV 5/5] END activation=relu, hidden_layer_sizes=(50, 50), max_iter=10000, solver=lbfgs; total time=   5.2s
[CV 4/5] END activation=relu, hidden_layer_sizes=(50, 50), max_iter=10000, solver=lbfgs; total time=   6.8s
[CV 3/5] END activation=relu, hidden_layer_sizes=(50, 50), max_iter=10000, solver=lbfgs; total time=   7.6s
[CV 2/5] END activation=relu, hidden_layer_sizes=(50, 50), max_iter=10000, solver=lbfgs; total time=   9.5s
Model with rank: 1
Mean validation score: 0.944 (std: 0.009)
Parameters: {'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'max_iter': 10000, 'solver': 'lbfgs'}
```

The default folds for cross-validation is 5, but can be customized as desired with the `-cv|--cross_val` flag:
```
$ ./gridsearch.py inputs/train_data_1000 -s lbfgs -cv 10 -v 2 -mi 10000

Data file used is inputs/train_data_1000
hidden_layer_sizes : [(100,)]
activation : ['relu']
solver : ['lbfgs']
max_iter : [10000]

Number of models tested: 1

Fitting 10 folds for each of 1 candidates, totalling 10 fits
[CV] END activation=relu, hidden_layer_sizes=(100,), max_iter=10000, solver=lbfgs; total time=   5.2s
[CV] END activation=relu, hidden_layer_sizes=(100,), max_iter=10000, solver=lbfgs; total time=   5.3s
[CV] END activation=relu, hidden_layer_sizes=(100,), max_iter=10000, solver=lbfgs; total time=   5.6s
[CV] END activation=relu, hidden_layer_sizes=(100,), max_iter=10000, solver=lbfgs; total time=   5.9s
[CV] END activation=relu, hidden_layer_sizes=(100,), max_iter=10000, solver=lbfgs; total time=   6.6s
[CV] END activation=relu, hidden_layer_sizes=(100,), max_iter=10000, solver=lbfgs; total time=   1.9s
[CV] END activation=relu, hidden_layer_sizes=(100,), max_iter=10000, solver=lbfgs; total time=   7.6s
[CV] END activation=relu, hidden_layer_sizes=(100,), max_iter=10000, solver=lbfgs; total time=   7.6s
[CV] END activation=relu, hidden_layer_sizes=(100,), max_iter=10000, solver=lbfgs; total time=   7.9s
[CV] END activation=relu, hidden_layer_sizes=(100,), max_iter=10000, solver=lbfgs; total time=   4.5s
Model with rank: 1
Mean validation score: 0.923 (std: 0.028)
Parameters: {'activation': 'relu', 'hidden_layer_sizes': (100,), 'max_iter': 10000, 'solver': 'lbfgs'}
```

Changes to the code should be run through the test suites using command `make test` to ensure passing of all tests:

```
$ make test
pytest -xv --pylint --flake8 test.py gridsearch.py
======================================== test session starts ========================================
...
collected 21 items                                                                                        
--------------------------------------------------------------------------------
Linting files
..
--------------------------------------------------------------------------------

test.py::PYLINT PASSED                                                                              [  4%]
test.py::FLAKE8 PASSED                                                                              [  9%]
test.py::test_exists PASSED                                                                         [ 14%]
test.py::test_usage PASSED                                                                          [ 19%]
test.py::test_bad_file PASSED                                                                       [ 23%]
test.py::test_bad_hls_comma PASSED                                                                  [ 28%]
test.py::test_bad_hls_negative_1 PASSED                                                             [ 33%]
test.py::test_bad_hls_negative_2 PASSED                                                             [ 38%]
test.py::test_bad_hls_negative_3 PASSED                                                             [ 42%]
test.py::test_bad_activation PASSED                                                                 [ 47%]
test.py::test_bad_solver PASSED                                                                     [ 52%]
test.py::test_bad_cv_1 PASSED                                                                       [ 57%]
test.py::test_bad_cv_2 PASSED                                                                       [ 61%]
test.py::test1 PASSED                                                                               [ 66%]
test.py::test2 PASSED                                                                               [ 71%]
test.py::test3 PASSED                                                                               [ 76%]
test.py::test1_outfile PASSED                                                                       [ 80%]
test.py::test2_outfile PASSED                                                                       [ 85%]
test.py::test3_outfile PASSED                                                                       [ 90%]
gridsearch.py::PYLINT PASSED                                                                        [ 95%]
gridsearch.py::FLAKE8 PASSED                                                                        [100%]

=========================================== 21 passed in 35.73s ===========================================                                                                       [100%]
```

## Author

Linh Tran <lnt@email.arizona.edu>
