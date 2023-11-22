#import packages
import torch
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse

#import data
def genhist(truthFileName, predictionFileName, plotname):
    with h5py.File(truthFileName) as f:
        y_t = np.array(f["labels"])
        nx = torch.Tensor(y_t[:,3])
        ny = torch.Tensor(y_t[:,4])
        nz = torch.Tensor(y_t[:,5])
        pTTrue = torch.Tensor(y_t[:,8])
        eta = -torch.log(abs(torch.tan((1/2)*(torch.arctan2(nz,nx)))))
        etaTrue = torch.Tensor(eta)
        phi = (torch.arctan2(nz,ny))
        phiTrue = torch.Tensor(phi)
        # y_true = torch.Tensor()
        # y_true = torch.stack((etatensor, phitensor), dim=1)
    
    with h5py.File(predictionFileName) as f:
        y_p = np.array(f["outputs"])
        eta = y_p[:,0] 
        etaPred = torch.Tensor(eta) #[mask])
        phi = (y_p[:,1])
        phiPred = torch.Tensor(phi) #[mask])
        pTp = y_p[:,2]
        pTPred = torch.Tensor(pTp)
        # y_pred = torch.Tensor()
        # y_pred = torch.stack((etatensor, phitensor), dim=1)

    #make resolution plot
    plt.hist(etaTrue,bins=np.linspace(-4,4,200),histtype='step',label='eta') #, color="blue")
    plt.hist(phiTrue,bins=np.linspace(-4,4,200),histtype='step',label='phi')
    plt.hist(pTTrue,bins=np.linspace(-4,4,200),histtype='step',label='pT')
    #plt.hist(pTPred,bins=np.linspace(-1,2,200),histtype='step',label='pT prediction') #, color="green")
    #plt.hist(etaTrue-etaPred,bins=np.linspace(-5,5,100),histtype='step',label='pT difference')
    #plt.hist(((abs(etaTrue-etaPred))/etaTrue), bins=np.linspace(-5,5,50), histtype='step', label='resolution eta absvalue')#, color="red")
    #plt.hist(((pTTrue-pTPred)/pTTrue), bins=np.linspace(-1,2,200), histtype='step', label='pT resolution') #, color="purple")

    plt.yscale('log')
    plt.xlabel('truth value eta, phi (rad), pT (GeV)')
    plt.title ('labels (truth distribution)')
    plt.legend()
    plt.show()
    plt.savefig(f"{plotname}-7layers-0.01lr-32b-onlypT-pT-ref.png", format="png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("-h", default=None, help="help")
    parser.add_argument("-t",  "--truthFileName", default=None, help="Input file for truths")
    parser.add_argument("-p",  "--predictionFileName", default=None, help="Input file for predictions")
    parser.add_argument("-n", "--plotname", default="plot", help="Name of resulting plot")
    ops = parser.parse_args()

    genhist(ops.truthFileName, ops.predictionFileName, ops.plotname)