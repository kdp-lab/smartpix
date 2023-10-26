#! /usr/bin/env python

'''
Author: Anthony Badea
'''

# python packages
import torch
import argparse
import numpy as np
import os
import h5py
import json
import glob
from tqdm import tqdm 

# multiprocessing
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# custom code
from batcher import loadDataFromH5, loadDataFromPd
from model import ModelLightning

def evaluate(config):
    ''' perform the full model evaluation '''
    
    ops = options()
    config["model"]["weights"] = ops.weights

    print(f"evaluating on {config['inFileName']}")

    # load model
    model = ModelLightning(**config["model"])
    model.to(config["device"])
    model.eval()

    # load data
    x, y = loadDataFromH5(config["inFileName"])
    # x, y = loadDataFromPd()
    
    # evaluate
    outData = {}
    with torch.no_grad():
        
        angles = model(x) # .to(config["device"]) # uncomment to use the gpu memory
                
        # make output
        outData = {
            "angles": angles #.numpy().flatten(), eliminated the flatten in the model as well # raw prediction
        }

    # save final file
    print(f"Saving to {config['outFileName']}")
    with h5py.File(config['outFileName'], 'w') as hf:
        for key, val in outData.items():
            print(f"{key} {val.shape}")
            hf.create_dataset(key, data=val)
    print("Done!")

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config_file", help="Configuration file.", default="./config_files/default_config.json")
    parser.add_argument("-i",  "--inFile", help="Data file to evaluate on.", default=None, required=True)
    parser.add_argument("-o",  "--outDir", help="Directory to save evaluation output to.", default="./")
    parser.add_argument("-j",  "--ncpu", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    parser.add_argument("-w",  "--weights", help="Pretrained weights to evaluate with.", default=None, required=True)
    parser.add_argument("-b", "--batch_size", help="Batch size", default=10**5, type=int)
    parser.add_argument('--doOverwrite', action="store_true", help="Overwrite already existing files.")
    parser.add_argument('--gpu', action="store_true", help="Run evaluation on gpu.")
    return parser.parse_args()
 
if __name__ == "__main__":

    # user options
    ops = options()

    # load input data
    data = ops.inFile
    if os.path.isfile(data) and ".h5" in os.path.basename(data):
        data = [data]
    elif os.path.isfile(data) and ".root" in os.path.basename(data):
        data = [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        data = sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        data = sorted(os.listdir(data))
    elif "*" in data:
        data = sorted(glob.glob(data))

    # make output dir
    outDir = ops.outDir if ops.outDir != "same" else os.path.dirname(ops.weights)
    if not os.path.isdir(outDir):
        os.makedirs(outDir)

    # pick up model configurations
    print(f"Using configuration file: {ops.config_file}")
    with open(ops.config_file, 'r') as fp:
        model_config = json.load(fp)

    # understand device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if ops.gpu else "cpu"

    # create evaluation job dictionaries
    config  = []
    for inFileName in data:

        # make out file name and check if already exists
        outFileName = os.path.join(outDir, os.path.basename(inFileName)).replace(".h5","_Model.h5")
        if os.path.isfile(outFileName) and not ops.doOverwrite:
            print(f"File already exists not evaluating on: {outFileName}")
            continue

        # append configuration
        config.append({
            "inFileName" : inFileName,
            "outFileName" : outFileName,
            "device" : device,
            **model_config
        })

    # launch jobs
    if ops.ncpu == 1:
        for conf in config:
            evaluate(conf)
    else:
        results = mp.Pool(ops.ncpu).map(evaluate, config)
