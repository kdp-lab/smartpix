'''
Author: Anthony Badea
Date: December 31, 2023
Purpose: Convert hepmc output to track list input for PixelAV
'''

import pyhepmc
import numpy as np
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    
    # user options
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inFileName", help="Input file name", default="/home/abadea/asic/smartpix/simulation/test.hepmc")
    parser.add_argument("-o", "--outFileName", help="Output file name", default="track_list.txt")
    parser.add_argument("-n", "--nparticles", help="Number of particles to save in track list (-1=All). ", default=100, type=int)
    ops = parser.parse_args()

    # list for tracks
    tracks = [] # cota cotb p flp localx localy pT
    doBreak = False
    
    # pyhepmc.open can read most HepMC formats using auto-detection
    with pyhepmc.open(ops.inFileName) as f:
        # loop over events
        for iF, event in tqdm(enumerate(f)):
            # loop over particles
            # print(f"Event {iF} has {len(event.particles)} particles")
            for particle in event.particles:

                if ops.nparticles != -1 and len(tracks) >= ops.nparticles:
                    doBreak = True
                    break
                
                # # to get the production vertex and position
                prod_vertex = particle.production_vertex
                prod_vector = prod_vertex.position
                prod_R = prod_vector.perp() # [mm]

                # # to get the decay vertex
                # decay_vertex = particle.end_vertex
                # decay_vector = decay_vertex.position
                
                # decide which particles to keep
                accept = (particle.status == 1) # final state particle
                accept *= (abs(particle.momentum.eta()) != float("inf")) # non infinite eta
                accept *= ( (prod_R >= 30) * (prod_R <= 130) ) # within pixel detector ~ 30-130mm

                if not accept:
                    continue

                # print(particle.id, particle.pid, particle.status, particle.momentum.pt(), particle.momentum.eta(), particle.momentum.phi(), prod_vector.x, prod_vector.y, accept)

                # track properties
                cota = particle.momentum.phi()
                cotb = particle.momentum.eta()
                p = particle.momentum.p3mod()
                flp = 0
                localx = prod_vector.x # this needs to be particle position at start of pixel detector or only save particles that are produced within epsilon distance of detector
                localy = prod_vector.y # [mm]
                pT = particle.momentum.pt()
                tracks.append([cota, cotb, p, flp, localx, localy, pT])

            if doBreak:
                break # need to understand what to put in track_list between events

        # save to file
        # need to update this so that flp is an int 0/1 not 0.0000 or 1.0000
        np.savetxt(ops.outFileName, tracks, delimiter=' ', fmt="%f")
