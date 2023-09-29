'''
batch manager for handling a list of files on input. loads them asynchronously to be ready when called on. 
'''

# python imports
import torch
import numpy as np
import argparse
import h5py

def loadDataFromH5(inFileName):
    
    # load in the data
    with h5py.File(inFileName) as f:
        x = np.array(f["data"])
        y = np.array(f["labels"])
    
    # the labels are x-entry, y-entry, z-entry, n_x, n_y, n_z, number_eh_pairs, y-local, pt
    # taken from https://zenodo.org/record/7331128
    # other relevant parameters to compute are https://github.com/kdipetri/semiparametric/blob/master/processing/datagen.py#L110C9-L115C95
    # cotAlpha = y[:,3]/y[:,5] # n_x/n_z
    cotBeta = y[:,4]/y[:,5] # n_y/n_z
    # sensor_thickness = 100 #um                                                          
    # x_midplane = y[:,0] + cotBeta*(sensor_thickness/2 - y[:,2]) # x-entry + cotAlpha*(sensor_thickness/2 - z-entry)
    # y_midplane = y[:,1] + cotBeta*(sensor_thickness/2 - y[:,2]) # y-entry + cotBeta*(sensor_thickness/2 - z-entry)

    # for 1D only the last time slice
    x = x[:,-1].reshape(x.shape[0], -1)
    
    # convert to tensor
    x = torch.Tensor(x)
    y = torch.Tensor(cotBeta)

    return x, y 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i",  "--inFileName", default=None, help="Input file")
    ops = parser.parse_args()

    x, y = loadDataFromH5(ops.inFileName)
    print(x.shape, y.shape)
