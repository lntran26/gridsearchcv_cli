""" Tests for gridsearch.py """

import os
import re
import random
import string
from subprocess import getstatusoutput, getoutput

PRG = './gridsearch.py'
DATA1 = './inputs/train_data_100'
DATA2 = './inputs/train_data_diabetes'


# --------------------------------------------------
def random_string():
    """ generate a random string """

    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))


# --------------------------------------------------
def test_exists():
    """ Program exists """

    assert os.path.isfile(PRG)


# --------------------------------------------------
def test_usage():
    """ Prints usage """

    for flag in ['', '-h', '--help']:
        out = getoutput(f'{PRG} {flag}')
        assert re.match("usage", out, re.IGNORECASE)


# --------------------------------------------------
def test_bad_file():
    """ fails on bad input """

    bad = random_string()
    rv, out = getstatusoutput(f'{PRG} {bad}')
    assert rv != 0
    assert re.search(f"No such file or directory: '{bad}'", out)


# --------------------------------------------------
def test_bad_hls_comma():
    """ fails on bad hidden layer sizes specification """

    char = ['-', '.', '/']  # cannot use ;
    rv, out = getstatusoutput(
        f'{PRG} {DATA1} -hls 10{random.choice(char)}10')
    assert rv != 0
    assert re.search("invalid tuple_of_pos_int value:", out)


# --------------------------------------------------
def test_bad_hls_negative_1():
    """ fails on bad hidden layer sizes specification """

    rv, out = getstatusoutput(
        f'{PRG} {DATA1} -hls -5')
    assert rv != 0
    assert re.search("invalid tuple_of_pos_int value: '-5'", out)


# --------------------------------------------------
def test_bad_hls_negative_2():
    """ fails on bad hidden layer sizes specification """

    rv, out = getstatusoutput(
        f'{PRG} {DATA1} -hls 100,0')
    assert rv != 0
    assert re.search("invalid tuple_of_pos_int value: '100,0'", out)


# --------------------------------------------------
def test_bad_hls_negative_3():
    """ fails on bad hidden layer sizes specification """

    rv, out = getstatusoutput(
        f'{PRG} {DATA1} -hls 100 -1')
    assert rv != 0
    assert re.search("invalid tuple_of_pos_int value: '-1'", out)


# --------------------------------------------------
def test_bad_activation():
    """ fails on bad activation function name """

    bad = random_string()
    rv, out = getstatusoutput(f'{PRG} {DATA1} -a {bad}')
    assert rv != 0
    assert re.search(f"-a/--activation: invalid choice: '{bad}'", out)


# --------------------------------------------------
def test_bad_solver():
    """ fails on bad optimizer name """

    bad = random_string()
    rv, out = getstatusoutput(f'{PRG} {DATA2} -s {bad}')
    assert rv != 0
    assert re.search(f"-s/--solver: invalid choice: '{bad}'", out)


# --------------------------------------------------
def test_bad_cv_1():
    """ fails on bad k-fold value """

    rv, out = getstatusoutput(f'{PRG} {DATA1} -cv 4.5')
    assert rv != 0
    assert re.search("invalid int_kfold value: '4.5'", out)


# --------------------------------------------------
def test_bad_cv_2():
    """ fails on bad k-fold value """

    rv, out = getstatusoutput(f'{PRG} {DATA1} -cv 1')
    assert rv != 0
    assert re.search("-cv/--cross_val: k-fold input must be > 2", out)


# --------------------------------------------------
def run(args, n):
    """ Run test """

    rv, out = getstatusoutput(f'{PRG} {" ".join(args)}')
    assert rv == 0
    assert re.search(f"Data file used is {args[0]}", out)
    assert re.search(f"Number of models tested: {n}", out)


# --------------------------------------------------
def run_outfile(args, n):
    """ Run test with outfile """

    outfile = random_string()
    try:
        rv, out = getstatusoutput(f'{PRG} {" ".join(args)} -o {outfile}')
        assert rv == 0
        assert os.path.isfile(outfile)
        assert re.search(f"Data file used is {args[0]}", out) is None
        assert re.search(f"Number of models tested: {n}", out) is None
    finally:
        if os.path.isfile(outfile):
            os.remove(outfile)


# --------------------------------------------------
def test1():
    """ test """

    run([DATA1], 1)


# --------------------------------------------------
def test2():
    """ test """

    run([DATA2, '-hls 100 50,50 25,25,25,25'], 3)


# --------------------------------------------------
def test3():
    """ test """

    run([DATA1, '-a relu tanh', '-s lbfgs adam'], 4)


# --------------------------------------------------
def test1_outfile():
    """ test """

    run([DATA1], 1)


# --------------------------------------------------
def test2_outfile():
    """ test """

    run([DATA2, '-hls 100 50,50 25,25,25,25'], 3)


# --------------------------------------------------
def test3_outfile():
    """ test """

    run([DATA1, '-a relu tanh', '-s lbfgs adam'], 4)
