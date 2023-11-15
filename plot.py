import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load data from the HDF5 file
with h5py.File('evaluation_results.h5', 'r') as f:
    labels = np.array(f['labels'])
    outputs = np.array(f['outputs'])

bins = np.linspace(-2,2,101)

# Create histograms
plt.figure(figsize=(10, 6))
plt.hist(labels, bins=bins, color='blue', label='Labels', histtype="step")
plt.hist(outputs, bins=bins, color='orange', label='Outputs', histtype="step")
plt.xlabel('Output Values')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
