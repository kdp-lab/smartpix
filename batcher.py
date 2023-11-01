'''
batch manager for handling a list of files on input. loads them asynchronously to be ready when called on. 
'''

# python imports
import torch
import numpy as np
import argparse
import h5py
import pandas as pd

def loadDataFromPd():
    datadir='/data/ckumar/'
    #train_df = pd.read_csv(datadir+'train_1Dy_23-09-29-c.csv')
    #test_df = pd.read_csv(datadir+'test_1Dy_23-09-29-c.csv')
    train_df = pd.read_csv(datadir+'train_1Dy_d17201.csv')
    test_df = pd.read_csv(datadir+'test_1Dy_d17201.csv')
    #train_df = pd.read_csv(datadir+'train_1Dy_23-10-04.csv')
    #test_df = pd.read_csv(datadir+'train_1Dy_23-10-04.csv')

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    x_train = train_df.drop(columns=['y-midplane','cotBeta','pt', 'cotAlpha', 'y-local']).values
    x_test = test_df.drop(columns=['y-midplane','cotBeta','pt', 'cotAlpha', 'y-local']).values

    y_train = (train_df['cotBeta'].values)
    y_test = (test_df['cotBeta'].values)

    x = np.concatenate([x_train,x_test])
    y = np.concatenate([y_train,y_test])

    # convert to tensor
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    return x, y


def loadDataFromH5(inFileName):
    
    # load in the data
    with h5py.File(inFileName) as f:
        x = np.array(f["data"])
        y = np.array(f["labels"])
    
    # the labels are x-entry, y-entry, z-entry, n_x, n_y, n_z, number_eh_pairs, y-local, pt
    # taken from https://zenodo.org/record/7331128
    # other relevant parameters to compute are https://github.com/kdipetri/semiparametric/blob/master/processing/datagen.py#L110C9-L115C95
    # cotAlpha = y[:,3]/y[:,5] # n_x/n_z
    # cotBeta = y[:,4]/y[:,5] # n_y/n_z
    # cotBeta = abs(cotBeta)
    # sensor_thickness = 100 #um                                                          
    # x_midplane = y[:,0] + cotBeta*(sensor_thickness/2 - y[:,2]) # x-entry + cotAlpha*(sensor_thickness/2 - z-entry)
    # y_midplane = y[:,1] + cotBeta*(sensor_thickness/2 - y[:,2]) # y-entry + cotBeta*(sensor_thickness/2 - z-entry)
    y_local = torch.Tensor(y[:,7])
    y_local = torch.unsqueeze(y_local, dim=1)

    #new angles
    nx = torch.Tensor(y[:,3])
    ny = torch.Tensor(y[:,4])
    nz = torch.Tensor(y[:,5])
    eta = -torch.log(abs(torch.tan((1/2)*(torch.arctan2(nz,nx))))) #all negative values go to NaN w/np.log; abs value prevents this
    phi = (torch.arctan2(nz,ny))*(180/torch.pi)

    pT = torch.Tensor(y[:,8])
    
    # for 1D only the last time slice
    x = x[:,-1].reshape(x.shape[0], -1)
    # x = x[:,-1].sum(2)
    # x = (x-x.mean())/x.std()
                
    # form final input
    # x = np.concatenate([x, y_local.reshape(-1,1)],-1)

    # event selection
    # mask = abs(y[:,8]) >= 0.3 # pt>0.3 GeV
    # print(mask.sum()/mask.shape[0])

    # convert to tensor
    xcluster = torch.Tensor(x)#[mask])
    print(xcluster.shape)
    print(y_local.shape)
    x = torch.concatenate([xcluster, y_local], dim=1)
    e = torch.Tensor(eta) #[mask])
    p = torch.Tensor(phi)
    y = torch.Tensor(pT)

    return x, y 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i",  "--inFileName", default=None, help="Input file")
    ops = parser.parse_args()

    x, y = loadDataFromH5(ops.inFileName)
    print(x.shape, y.shape)
