#import packages
import torch
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse

def genhist(truthFileName, predictionFileName, withoutlabelFileName, plotname):
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
        #pTp = y_p[:,2]
        #pTPred = torch.Tensor(pTp)
        # y_pred = torch.Tensor()
        # y_pred = torch.stack((etatensor, phitensor), dim=1)

    with h5py.File(withoutlabelFileName) as f:
        y_wo = np.array(f["outputs"])
        phi_wo = (y_wo[:,1])
        phiWOPred = torch.Tensor(phi_wo)

    #make resolution plot
    #plt.hist(phiTrue,bins=np.linspace(-0.5,0.5,100),histtype='step',label='pT truth') #, color="blue")
    #plt.hist(phiPred,bins=np.linspace(-0.5,0.5,100),histtype='step',label='pT prediction w/ label') #, color="green")
    #plt.hist(phiWOPred, bins=np.linspace(-0.5,0.5, 100), histtype='step', label='pT prediction w/o label')
    #plt.hist(pTTrue-pTPred,bins=np.linspace(-2,2,200),histtype='step',label='pT difference')
    #plt.hist(((abs(etaTrue-etaPred))/etaTrue), bins=np.linspace(-5,5,50), histtype='step', label='resolution eta absvalue')#, color="red")
    plt.hist(((phiTrue-phiPred)/phiTrue), bins=np.linspace(-0.5,0.5,200), histtype='step', label='with pT label') #, color="purple")
    plt.hist(((phiTrue-phiWOPred)/phiTrue), bins=np.linspace(-0.5,0.5,200), histtype='step', label='without pT label')
    plt.yscale('log')
    plt.xlabel('phi (rad)')
    plt.title('phi resolution')
    plt.legend()
    plt.show()
    #plt.savefig("11-13-flatplot-7layers--0.01lr-32b-pT-res-comparison-01.png", format="png")
    plt.savefig(f"{plotname}-7layers--0.01lr-32b-phi-res-comparison-12(log).png", format="png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("-h", default=None, help="help")
    parser.add_argument("-t",  "--truthFileName", default=None, help="Input file for truths")
    parser.add_argument("-p",  "--predictionFileName", default=None, help="Input file for predictions")
    parser.add_argument("-wo", "--withoutlabelFileName", default=None, help="Input file for predictions without pT labels")
    parser.add_argument("-n", "--plotname", default="plot", help="Name of resulting plot")
    #parser.add_argument("-v", "--var", default="eta", help="eta or phi variable?")
    ops = parser.parse_args()

    genhist(ops.truthFileName, ops.predictionFileName, ops.withoutlabelFileName, ops.plotname)