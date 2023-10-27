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
        nx = torch.Tensor(y_t[:,3])
        ny = torch.Tensor(y_t[:,4])
        nz = torch.Tensor(y_t[:,5])
        eta = -torch.log(abs(torch.tan((1/2)*(torch.arctan2(nz,nx)))))
        etaTrue = torch.Tensor(eta)
        phi = (torch.arctan2(nz,ny))*(180/torch.pi)
        phiTrue = torch.Tensor(phi)
        # y_true = torch.Tensor()
        # y_true = torch.stack((etatensor, phitensor), dim=1)
    
    with h5py.File(predictionFileName) as f:
        y_p = np.array(f["angles"])
        eta = y_p[:,0] 
        etaPred = torch.Tensor(eta) #[mask])
        phi = y_p[:,1] 
        phiPred = torch.Tensor(phi) #[mask])
        # y_pred = torch.Tensor()
        # y_pred = torch.stack((etatensor, phitensor), dim=1)
    #tensortrue = f"{var}t"
    #tensorpred = f"{var}pred"
    
    #make eta plot
    plt.hist(phiTrue,bins=np.linspace(-200,50,50),histtype='step',label='phi truth')
    plt.hist(phiPred,bins=np.linspace(-200,50,50),histtype='step',label='phi prediction')
    #plt.hist(etaTrue-etaPred, bins=np.linspace(-1,1,50), histtype='step', label='difference')
    plt.yscale('log')
    plt.xlabel(r'$\phi$ (deg)')
    plt.legend()
    plt.show()
    plt.savefig(f"{plotname}-ylocal-phi-01-log.png", format="png")
    
    #plt.hist(tensortrue,bins=np.linspace(-1,1,50),histtype='step',label='f"{var}" truth')
    #plt.hist(tensorpred,bins=np.linspace(-1,1,50),histtype='step',label='f"{var}" prediction')
    #plt.hist(etaTrue-etaPred, bins=np.linspace(-1,1,50), histtype='step', label='difference')
    #plt.yscale('log')
    #plt.xlabel(f"r'$\{var}$'")
    #plt.legend()
    #plt.show()
    #plt.savefig(f"{plotname}-{var}-corrected-4.png", format="png")

    #make phi plot
    #plt.hist(phiTrue,bins=np.linspace(-1,1,50),histtype='step',label='phi truth')
    #plt.hist(phiPred,bins=np.linspace(-1,1,50),histtype='step',label='phi prediction')
    #plt.hist(phiTrue-phiPred, bins=np.linspace(-1,1,50), histtype='step', label='difference')
    #plt.yscale('log')
    #plt.xlabel(r'$\phi$')
    #plt.legend()
    #plt.show()
    #plt.savefig(f"{plotname}-phi-corrected-3.png", format="png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("-h", default=None, help="help")
    parser.add_argument("-t",  "--truthFileName", default=None, help="Input file for truths")
    parser.add_argument("-p",  "--predictionFileName", default=None, help="Input file for predictions")
    parser.add_argument("-n", "--plotname", default="plot", help="Name of resulting plot")
    #parser.add_argument("-v", "--var", default="eta", help="eta or phi variable?")
    ops = parser.parse_args()

    genhist(ops.truthFileName, ops.predictionFileName, ops.plotname)