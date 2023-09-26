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
    x = x[:,-1].sum(1) # .reshape(x.shape[0], -1) # only last time slice for now
    # add dummy for the y0 for now
    x = np.concatenate([x,np.zeros((x.shape[0],1))], -1)
    # pick up the labels
    y = pd.read_csv(yInFileName, nrows=nrows) # to get column names list(y.columns): ['x-entry', 'y-entry', 'z-entry', 'n_x', 'n_y', 'n_z', 'number_eh_pairs', 'y-local', 'pt', 'cotAlpha', 'cotBeta', 'y-midplane', 'x-midplane']
    #y = y['y-midplane'].values # ['x-midplane','y-midplane']

    # The models are trained with three output categories:
    pt = y['pt'].values
    # 0. positively charged and pT < 200 MeV (0.2 GeV)
    c0 = (pt > 0) * (abs(pt) < 0.2)
    # 1. negatively charged and pT < 200 MeV
    c1 = (pt < 0) * (abs(pt) < 0.2)
    # 2. pT > 200MeV, both positively and negatively charged.
    c2 = (abs(pt) > 0.2)
    # make final label
    y = np.stack([c0,c1,c2],-1)
    y = np.where(y==1)[1]
    # count occurances and print
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    # convert to tensors
    x = torch.Tensor(x)
    y = torch.Tensor(y).long()

    return x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-x",  "--xInFileName", default=None, help="Input file")
    parser.add_argument("-y",  "--yInFileName", default=None, help="Input file")
    ops = parser.parse_args()

    x, y = loadDataFromCSV(ops.xInFileName, ops.yInFileName, 1000)
    print(x.shape, y.shape)