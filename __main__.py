

#OPTIMIZATION  

import argparse
# import os
import pickle
import torch
from .train import execute

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--topology", default='fc')     #topology (fc or ring)
    parser.add_argument("--size", default=3)            #size of the final yield (3=trimer)

    parser.add_argument("--k", default=1)               #default association rate
    parser.add_argument("--c_init", default=100)        #initial concentration of monomers (microM)
    parser.add_argument("--protocol", default=None)     #protocol (es A:rategrowth)
    parser.add_argument("--monomer_only", default=True) #reactions only involves monomers

    parser.add_argument("--dG", default=20)             #free energy of the reactions

    parser.add_argument("--runtime", default=1)         #runtime of the simulation
    parser.add_argument("--c_scale", default=1)         #parameter for the simulation
    parser.add_argument("--c_thresh", default=1e-1)     #parameter of the simulation
    parser.add_argument("--optim", default='time')      #parameter of the simulation (time or yield)

    parser.add_argument("--lr", default=1e-1)           #learning rate
    parser.add_argument("--iter", default=100)          #optim iterations

    parser.add_argument("--output", type=str, required=True)


    args = parser.parse_args().__dict__

    with open(args['output'], 'wb') as handle:
        pickle.dump(args, handle)


    for data in execute(**args, yield_time=10.0):
        data['args'] = args
        with open(args['output'], 'wb') as handle:
            pickle.dump(args, handle)
            pickle.dump(data, handle)


if __name__ == "__main__":
    main()