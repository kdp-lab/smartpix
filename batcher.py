'''
batch manager for handling a list of files on input. loads them asynchronously to be ready when called on. 
'''

# python imports
import torch
import numpy as np
import pandas as pd
import argparse

def loadDataFromCSV(xInFileName, yInFileName, nrows=None):

    # pick up the data
    x = pd.read_csv(xInFileName, nrows=nrows).to_numpy().reshape(-1, 20, 21, 13) # np.all(r3d[:,-1] == r2d[:,0]) -> True comparing the 2d and r3d datasets. The r2d is the last time slice of 3d
    # just do the 2d for now
    x = x[:,-1].reshape(x.shape[0], -1) # only last time slice for now
    # pick up the labels
    y = pd.read_csv(yInFileName, nrows=nrows) # to get column names list(y.columns): ['x-entry', 'y-entry', 'z-entry', 'n_x', 'n_y', 'n_z', 'number_eh_pairs', 'y-local', 'pt', 'cotAlpha', 'cotBeta', 'y-midplane', 'x-midplane']
    y = y['y-midplane'].values # ['x-midplane','y-midplane']

    # convert to tensors
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    return x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-x",  "--xInFileName", default=None, help="Input file")
    parser.add_argument("-y",  "--yInFileName", default=None, help="Input file")
    ops = parser.parse_args()

    x, y = loadDataFromCSV(ops.xInFileName, ops.yInFileName, 500)
    print(x.shape, y.shape)