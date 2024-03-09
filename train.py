# import threading
import time
# from functools import partial
from itertools import count
import math
import matplotlib.pyplot as plt
import numpy as np

import copy
from .reaction_network import ReactionNetwork, gtostr
from .vectorized_rxn_net import VectorizedRxnNet 
from .vec_sim import VecSim
from .optimizer import Optimizer
from .trap_metric import TrapMetric
from .compute_trap_factor import ComputeTrapFactor

import networkx as nx
import torch
from torch import DoubleTensor as Tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(Size="tetramer", **args):

    if args.get("size")==3:
        Size="trimer"
    elif args.get("size")==4:
        Size="tetramer"
    else:
        pass
    #print(f"Size: {Size}")
    
    if args.get("protocol")==None:
        base_input = f'ProteinSelfAssembly/{args.get("topology")}_{Size}_dG_{args.get("dG")}.pwr'
    elif args.get("protocol")=='A':
        base_input = f'{args.get("topology")}_{Size}_dG_{args.get("dG")}_rategrowth.pwr'
    else:
        pass
        #aggiungere gli altri protocolli

    print('File found', flush=True)
   
    return base_input


def execute(yield_time=0.0, **args):
    print(f"device={device} dtype={torch.ones(3).dtype}", flush=True)

    yield {
        'finished': False,
    }

    base_input = init(**args)

    #create reaction network
    rn = ReactionNetwork(base_input, one_step=True)
    #rn.resolve_tree()  

    # #Changing k_on
    # uid_dict = {}    
    # for n in rn.network.nodes():
    #     for k,v in rn.network[n].items():
    #         uid = v['uid']
    #         r1 = set(gtostr(rn.network.nodes[n]['struct']))
    #         p = set(gtostr(rn.network.nodes[k]['struct']))
    #         r2 = p-r1
    #         reactants = (r1,r2)
    #         uid_dict[(n,k)] = uid

    # new_kon = torch.zeros([rn._rxn_count], requires_grad=True).double()
    # new_kon = new_kon + Tensor([1.]*np.array(args.get("k")))

    # update_kon_dict = {}
    # for edge in rn.network.edges:
    #     #print(rn.network.get_edge_data(edge[0],edge[1]))
    #     update_kon_dict[edge] = new_kon[uid_dict[edge]]

    # nx.set_edge_attributes(rn.network,update_kon_dict,'k_on')


    # #Creating the vectorized network
    # vec_rn = VectorizedRxnNet(rn, dev='cpu')
    # vec_rn.reset(reset_params=True)

    
    rn.reset()
    rn.intialize_activations()
    #Optimizer
    optim = Optimizer(reaction_network=rn,
                    sim_runtime=args.get("runtime"),
                    optim_iterations=args.get("iter"),
                    learning_rate=args.get("lr"),
                    device=device,method="Adam")

    #get the index of the final yield    
    labels=nx.get_node_attributes(rn.network, 'struct')
    nodes_list_2 = [gtostr(labels[key]) for key in labels.keys()]
    longest_name = max(nodes_list_2, key=len)
    indx = nodes_list_2.index(longest_name)     
    
    #Perform the optimization
    optim.rn.update_reaction_net(rn)
    optim.optimize(conc_scale=args.get("conc_scale"),conc_thresh=args.get("conc_thresh"),mod_bool=True,optim=args.get("optim"),yield_species=indx)
    

    yield {
        't95': np.array(optim.final_t95),
        'sim_observables': optim.sim_observables,
        'history': optim.parameter_history,
        'finished': True,
    }

