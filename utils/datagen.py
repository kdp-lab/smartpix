'''
Author: Anthony Badea
Date: September 28, 2023
'''

import numpy as np
from datetime import datetime as dt
import multiprocessing as mp
import h5py
import argparse
import glob
import os

def process(config):
    
    print(f"Processing {config['inFileName']}")
    # notes for self about d17201 that the first two lines are other information. other files do not have this information

    with open(config["inFileName"],"r") as f:
        lines = [line.rstrip() for line in f]
        x = [[float(j) for j in i.split(" ")] for i in lines if "<" not in i and ">" not in i and len(i.split(" ")) == 21]
        x = np.array(x).reshape(int(len(x)/(13*20)), 20, 13, 21)
        y = [[float(j) for j in i.split(" ")] for i in lines if "<" not in i and ">" not in i and len(i.split(" ")) == 9]
        y = np.array(y)
    
    # save to file
    print(f"Creating {config['outFileName']}")
    with h5py.File(config['outFileName'], 'w') as hf:
        hf.create_dataset("labels", data=y)
        hf.create_dataset("data", data=x)

if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inFileDir", default=None, help="Input file")
    parser.add_argument("-o", "--outDir", default="./", help="Output directory")
    parser.add_argument("-j", "--ncpu", default=4, help="Number of cores to use for multiprocessing.", type=int)
    ops = parser.parse_args()

    # get file list
    fileList = glob.glob(ops.inFileDir) #"/data/karri/smartPix/dataset-lowpt/raw/pixel_clusters_*.out")
    
    # make configurations
    confs = []
    for f in fileList:
        runNumber = os.path.basename(fileList[0]).split("_")[-1].strip(".out")
        confs.append({
            "inFileName" : f,
            "outFileName" : os.path.join(ops.outDir, os.path.basename(f).replace(".out",".h5"))
        })

    # launch jobs
    if ops.ncpu == 1:
        for conf in confs:
            process(conf)
    else:
        results = mp.Pool(ops.ncpu).map(process, confs)
