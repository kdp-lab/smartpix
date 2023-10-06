import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt, numpy as np, h5py

f = h5py.File("dataPixelClusters/pixel_clusters_d17201.h5","r")
x = np.array(f['data'])
y = np.array(f['labels'])

cotAlpha = y[:,3]/y[:,5] # n_x/n_z
cotBeta = y[:,4]/y[:,5] # n_y/n_z

N=10
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
for i in range(N):
    fig = plt.figure()
    plt.imshow(x[i][-1], cmap=plt.cm.Reds, origin='lower')
    plt.colorbar()
    plt.xlabel("x plane")
    plt.ylabel("y plane")
    plt.title(rf"$p_{{T}}$={y[i][-1]:.3f} GeV, cot$\beta$={cotBeta[i]:.3f}, cot$\alpha$={cotAlpha[i]:.3}, last time step", fontsize=10)
    pdf.savefig(fig)

pdf.close()
