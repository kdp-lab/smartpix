import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt, numpy as np, h5py
import os
import argparse

parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i",  "--inFileName", default=None, help="Input file")
parser.add_argument("-n",  "--nEvents", default=10, help="Number of events to plot", type=int)
ops = parser.parse_args()

inIndex = os.path.basename(ops.inFileName).split("_")[-1].strip(".h5").strip("d")
f = h5py.File(ops.inFileName,"r")
x = np.array(f['data'])
y = np.array(f['labels'])
f.close()

# x-entry, y-entry, z-entry, n_x, n_y, n_z, number_eh_pairs, y-local, pt
x_entry = y[:,0]
y_entry = y[:,1]
z_entry = y[:,2]
n_x = y[:,3]
n_y = y[:,4]
n_z = y[:,5]
number_eh_pairs = y[:,6]
y_local = y[:,7]
pT = y[:,8]

# calculated
cotAlpha = n_x/n_z
cotBeta = n_y/n_z
times = np.linspace(200, 4000, 20)

# bins
xbins = np.linspace(1,21,21).astype(int)
ybins = np.linspace(1,13,13).astype(int)

# make pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
for i in range(ops.nEvents):
    fig = plt.figure()
    plt.imshow(x[i][-1], cmap=plt.cm.Reds, origin='lower')
    plt.xticks(xbins-1, xbins)
    plt.yticks(ybins-1, ybins)
    plt.colorbar()
    plt.grid()
    plt.xlabel("x plane")
    plt.ylabel("y plane")
    # plt.title(rf"Event={i} @ {times[-1]:.0f} ps, $p_{{T}}$={pT[i]:.3f} GeV, cot$\beta$={cotBeta[i]:.3f}, cot$\alpha$={cotAlpha[i]:.3}", fontsize=8)
    
    stamps = [
        rf"File: {inIndex}, Event: {i}, t: {times[-1]:.0f} ps",
        rf"x-entry: {x_entry[i]:.3f}, y-entry: {y_entry[i]:.3f}, z-entry: {z_entry[i]:.3f} $\mu$m",
        rf"n_x: {n_x[i]:.3f}, n_y: {n_y[i]:.3f}, n_z: {n_z[i]:.3f}",
        rf"number_eh_pairs: {number_eh_pairs[i]:.3f}, y_local: {y_local[i]:.3f} mm, $p_{{T}}$: {pT[i]:.3f} GeV",
        rf"cot$\beta$={cotBeta[i]:.3f}, cot$\alpha$={cotAlpha[i]:.3}"
    ]
    xtop, ytop, width = 0.02, 1.23, 0.05
    for iS, i in enumerate(stamps):
        plt.text(xtop, ytop-iS*width, i, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=7)

    pdf.savefig(fig)

pdf.close()
