#import packages
import torch
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse

#inFileName = "/data/karri/smartPix/dataset-lowpt/h5FromRaw/pixel_clusters_d17201.h5"
#inFileName = "/home/ckumar/smartpix/pixel_clusters_d17201_Model.h5"

def genhist(truthFileName, predictionFileName, plotname):
    with h5py.File(truthFileName) as f:
        y_t = np.array(f["labels"])
        cotBeta = y_t[:,4]/y_t[:,5] # n_y/n_z
        y_true = torch.Tensor(cotBeta) #[mask])
    
    with h5py.File(predictionFileName) as f:
        y_p = np.array(f["y_hat"])
        y_pred = torch.Tensor(y_p) #[mask])
    
    #make plot
    plt.hist(y_true,bins=np.linspace(-1,1,50),histtype='step',label='truth')
    plt.hist(y_pred,bins=np.linspace(-1,1,50),histtype='step',label='prediction')
    plt.hist(y_true-y_pred, bins=np.linspace(-1,1,50), histtype='step', label='difference')
    plt.yscale('log')
    plt.xlabel(r'$\cot\beta$')
    plt.legend()
    plt.show()
    plt.savefig(f"{plotname}.png", format="png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#    parser.add_argument("-h", default=None, help="help")
    parser.add_argument("-t",  "--truthFileName", default=None, help="Input file for truths")
    parser.add_argument("-p",  "--predictionFileName", default=None, help="Input file for predictions")
    parser.add_argument("-n", "--plotname", default="plot", help="Name of resulting plot")
    ops = parser.parse_args()

    genhist(ops.truthFileName, ops.predictionFileName, ops.plotname)