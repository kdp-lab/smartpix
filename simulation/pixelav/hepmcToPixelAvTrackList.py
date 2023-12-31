'''
Author: Anthony Badea
Date: December 31, 2023
Purpose: Convert hepmc output to track list input for PixelAV
'''

import pyhepmc
import numpy as np
import argparse

if __name__ == "__main__":
    
    # user options
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inFileName", help="Input file name", default="/home/abadea/asic/smartpix/simulation/test.hepmc")
    parser.add_argument("-o", "--outFileName", help="Output file name", default="track_list.txt")
    ops = parser.parse_args()

    # list for tracks
    tracks = [] # cotb cota p flp localx localy pT

    # pyhepmc.open can read most HepMC formats using auto-detection
    with pyhepmc.open(ops.inFileName) as f:
        # loop over events
        for event in f:
            # loop over particles
            print(len(event.particles))
            for particle in event.particles:
                
                # # to get the production vertex and position
                prod_vertex = particle.production_vertex
                prod_vector = prod_vertex.position
                
                # # to get the decay vertex
                # decay_vertex = particle.end_vertex
                # decay_vector = decay_vertex.position
                
                # decide which particles to keep
                accept = (particle.status > 9) * (abs(particle.momentum.eta()) != float("inf"))
                print(particle.id, particle.pid, particle.status, particle.momentum.pt(), particle.momentum.eta(), particle.momentum.phi(), accept)
                if not accept:
                    continue

                # track properties
                cotb = particle.momentum.eta()
                cota = particle.momentum.phi()
                p = particle.momentum.p3mod()
                flp = 0
                localx = prod_vector[0] # this needs to be particle position at start of pixel detector or only save particles that are produced within epsilon distance of detector
                localy = prod_vector[1]
                pT = particle.momentum.pt()
                tracks.append([cotb, cota, p, flp, localx, localy, pT])
            break

        # save to file
        np.savetxt(ops.outFileName, tracks, delimiter=' ', fmt="%f")
