'''
Author: Anthony Badea
Date: 02/05/24
Purpose: Extract track parameters from delphes output root files and save to track list input for PixelAV
'''

import argparse
import uproot
import glob
import awkward as ak
import numpy as np

if __name__ == "__main__":

    # user options
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inFileName", help="Input file name")
    parser.add_argument("-o", "--outFileName", help="Output file name", default="./")
    parser.add_argument("-p", "--float_precision", help="Float precision to save to track_list. ", default=5, type=int)
    ops = parser.parse_args()
    
    # track list
    tracks = [] # cota cotb p flp localx localy pT

    # load the root files
    # files = "/local/d1/badea/tracker/smartpix/simulation/outdir/cmsMatch/10/*.root"
    tree = "Delphes"
    delphes_track_pt = []
    delphes_particle_pt = []
    branches = ["Track.PID", "Track.PT", "Track.P", "Track.Eta", "Track.Phi", "Track.XOuter", "Track.YOuter"]
    pionPID = 211 # plus/minus

    # for array in uproot.iterate(f"{files}:{tree}", branches):
    with uproot.open(ops.inFileName) as f:
        # load the branches
        temp = {}
        for branch in branches:
            temp[branch] = np.array(ak.flatten(f[tree][branch].array()))
        
        # selection
        cut = (abs(temp["Track.PID"])==pionPID)

        # apply selection
        for branch in branches:
            temp[branch] = temp[branch][cut]
        
        # track properties
        # based on the image here https://github.com/kdp-lab/pixelav/blob/ppixelav2v2/ppixelav2_operating_inst.pdf
        cota = 1./np.tan(temp["Track.Phi"]) # phi = alpha - pi -> cot(alpha) = cot(phi+pi) = cot(phi) = 1/tan(phi)
        cotb = 1./np.tan(temp["Track.Eta"]) # should be theta but need to get get it # theta = beta - pi -> cot(beta) = cot(theta+pi) = cot(theta) = 1/tan(theta)
        p = temp["Track.P"] # [GeV]
        flp = np.zeros(p.shape)
        localx = temp["Track.XOuter"] # [mm]
        localy = temp["Track.YOuter"] # [mm]
        pT = temp["Track.PT"] # [GeV]
        tracks.append([cota, cotb, p, flp, localx, localy, pT])

    
    tracks = np.concatenate(tracks,-1).T
    print("Tracks shape: ", tracks.shape)
    
    # save to file
    float_precision=4
    with open(ops.outFileName, 'w') as file:
        for track in tracks:

            # set flp to an int
            track = list(track)
            track[3] = int(track[3])

            formatted_sublist = [f"{element:.{ops.float_precision}f}" if isinstance(element, float) else element for element in track]
            line = ' '.join(map(str, formatted_sublist)) + '\n'
            file.write(line)
        
