'''
Author: Anthony Badea
Date: 02/05/24
Purpose: Extract track parameters from delphes output root files and save to track list input for PixelAV
'''

import argparse
import uproot
import glob
import awkward as ak

# track list
tracks = [] # cota cotb p flp localx localy pT

# load the root files
files = "/local/d1/badea/tracker/smartpix/simulation/outdir/cmsMatch/10/*.root"
tree = "Delphes"
delphes_track_pt = []
delphes_particle_pt = []
branches = ["Track.PID", "Track.PT", "Track.Eta", "Track.Phi"]
pionPID = 211 # plus/minus
for array in uproot.iterate(f"{files}:{tree}", branches):
    
    # track pt
    pt = np.array(ak.flatten(array[branches[0]]))
    pid = np.array(ak.flatten(array[branches[1]]))
    delphes_track_pt.append(pt[(abs(pid)==pionPID)])

    # particle pt
    p_pt = np.array(ak.flatten(array[branches[2]]))
    status = np.array(ak.flatten(array[branches[3]]))
    pid = np.array(ak.flatten(array[branches[4]]))
    delphes_particle_pt.append(p_pt[(status==1) * (abs(pid)==pionPID)])

    # track properties
    cota = particle.momentum.phi()
    # based on the image here https://github.com/kdp-lab/pixelav/blob/ppixelav2v2/ppixelav2_operating_inst.pdf
    cota = 1./np.tan(particle.momentum.phi()) # phi = alpha - pi -> cot(alpha) = cot(phi+pi) = cot(phi) = 1/tan(phi)
    cotb = 1./np.tan(particle.momentum.theta()) # theta = beta - pi -> cot(beta) = cot(theta+pi) = cot(theta) = 1/tan(theta)
    p = particle.momentum.p3mod()
    flp = 0
    localx = prod_vector.x # this needs to be particle position at start of pixel detector or only save particles that are produced within epsilon distance of detector
    localy = prod_vector.y # [mm]
    pT = particle.momentum.pt()
    tracks.append([cota, cotb, p, flp, localx, localy, pT])

# save to file
float_precision=4
with open(ops.outFileName, 'w') as file:
    for track in tracks:
        formatted_sublist = [f"{element:.{ops.float_precision}f}" if isinstance(element, float) else element for element in track]
        line = ' '.join(map(str, formatted_sublist)) + '\n'
        file.write(line)
        
