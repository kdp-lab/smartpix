"concat data"

import numpy as np
import h5py

file1 = "/data/karri/smartPix/dataset-lowpt/h5FromRaw/pixel_clusters_d17201.h5"
file2 = "/data/karri/smartPix/dataset-lowpt/h5FromRaw/pixel_clusters_d17202.h5"

# load in the data
with h5py.File(file1) as f:
    x1 = np.array(f["data"])
    y1 = np.array(f["labels"])

with h5py.File(file2) as f:
    x2 = np.array(f["data"])
    y2 = np.array(f["labels"])
    
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    print("datalength")
    print(len(x))
    print(len(y))
    print(x)
    print(y)

#save to file from datagen.py
print("Creating file")
with h5py.File('/data/ckumar/11-15_twodatasets.h5', 'w') as hf:
    hf.create_dataset("labels", data=y)
    hf.create_dataset("data", data=x)
    print("done")