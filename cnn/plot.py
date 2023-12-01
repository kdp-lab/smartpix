import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

# Load data from the HDF5 file
with h5py.File(sys.argv[1], 'r') as f:
    labels = np.array(f['labels'])
    outputs = np.array(f['outputs'])

bins = np.linspace(-2,2,101)

# Create histograms
plt.figure(figsize=(10, 6))
plt.hist(labels[:,0], bins=bins, color='blue', label='Labels 0', histtype="step") 
plt.hist(outputs[:,0], bins=bins, color='orange',  label='Outputs 0', histtype="step")
plt.hist(labels[:,1], bins=bins, color='green', label='Labels 1', histtype="step") 
plt.hist(outputs[:,1], bins=bins, color='gold',  label='Outputs 1', histtype="step")
plt.hist(labels[:,2], bins=bins, color='purple', label='Labels 2', histtype="step") 
plt.hist(outputs[:,2], bins=bins, color='red',  label='Outputs 2', histtype="step")
plt.xlabel('Output Values')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig("/home/ckumar/smartpix/cnn/plot.png", format="png")