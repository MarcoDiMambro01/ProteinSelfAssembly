

#OPTIMIZATION  



self.lr = learning_rate
self.optim_iterations = optim_iterations
self.sim_observables = []
self.parameter_history = []
self.yield_per_iter = []
self.flux_per_iter = []
self.is_optimized = False
self.dt = None
self.final_solns = []
self.final_yields = []
self.curr_time= []
self.final_t50 = []
self.final_t85 = []
self.final_t95 = []
self.final_t99 = []
self.final_unused_mon = []
self.dimer_max=[]
self.chap_max=[]
self.endtimes=[]

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
    parser.add_argument("--protocol", default=None)     #protocol (es A:rategrowth)
    parser.add_argument("--monomer_only", default=True) #reactions only involves monomers

    parser.add_argument("--dG", default=20)             #free energy of the reactions

    parser.add_argument("--runtime", default=1)         #runtime of the simulation
    parser.add_argument("--c_scale", default=1)         #parameter for the simulation
    parser.add_argument("--c_thresh", default=1e-1)     #parameter of the simulation
    parser.add_argument("--optim", default='time')      #parameter of the simulation (time or yield)

    parser.add_argument("--lr", default=1e-1)           #learning rate
    parser.add_argument("--iter", default=100)          #optim iterations


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