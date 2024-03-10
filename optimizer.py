import sys
#sys.path.append("../../")

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import psutil

from .vec_sim import VecSim
from .vectorized_rxn_net  import VectorizedRxnNet

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiplicativeLR
import random
import pandas as pd

class Optimizer:
    def __init__(self, reaction_network,
                 sim_runtime: float,
                 optim_iterations: int,
                 learning_rate: float,
                 device='cpu',
                 method='Adam',
                 lr_change_step=None,
                 gamma=None,
                 mom=0,
                 random_lr=False):

        # Load device for PyTorch (e.g. GPU or CPU)
        if torch.cuda.is_available():# and "cpu" not in device:
            self.dev = torch.device(device)
            #print("Using " + device)
        else:
            self.dev = torch.device("cpu")
            device = 'cpu'
            print("Using CPU")
        self._dev_name = device

        self.sim_class = VecSim
        
        if type(reaction_network) is not VectorizedRxnNet :
        #if type(reaction_network) is ReactionNetwork :
            try:
                self.rn = VectorizedRxnNet(reaction_network, dev=self.dev)
            except Exception:
                raise TypeError("Must be type ReactionNetwork or VectorizedRxnNetwork.")
        else:
            self.rn = reaction_network
        self.sim_runtime = sim_runtime
        param_itr = self.rn.get_params()

        if method == 'Adam':
            if self.rn.partial_opt:
                params_list=[]
                self.lr_group=[]
                print("Params: ",param_itr)
                for i in range(len(param_itr)):
                    # print("Learn Rate: ",learning_rate)
                    learn_rate = random.uniform(learning_rate,learning_rate*10)
                    params_list.append({'params':param_itr[i], 'lr':torch.mean(param_itr[i]).item()*learn_rate})
                    self.lr_group.append(learn_rate)
                self.optimizer = torch.optim.Adam(params_list)
            elif self.rn.chap_is_param:
                param_list = []
                for i in range(len(param_itr)):
                    lr_val = torch.mean(param_itr[i]).item()*learning_rate[i]
                    if lr_val>=torch.min(param_itr[i]).item()*0.1:
                        lr_val = torch.min(param_itr[i]).item()*1
                    param_list.append({'params':param_itr[i], 'lr':lr_val})
                self.optimizer = torch.optim.Adam(param_list)
            else:
                self.optimizer = torch.optim.Adam(param_itr, learning_rate)
        elif method =='RMSprop':
            if self.rn.chap_is_param:
                param_list = []
                # param_list2 = []

                for i in range(len(param_itr)):
                    lr_val = torch.mean(param_itr[i]).item()*learning_rate[i]
                    if lr_val>=torch.min(param_itr[i]).item()*0.1:
                        lr_val = torch.min(param_itr[i]).item()*1
                    param_list.append({'params':param_itr[i], 'lr':lr_val})
                self.optimizer = torch.optim.RMSprop(param_list,momentum=mom)
            else:
                if self.rn.partial_opt and not self.rn.coupling:
                    params_list=[]
                    self.lr_group=[]
                    print("Params: ",param_itr)
                    for i in range(len(param_itr)):
                        # print("Learn Rate: ",learning_rate)
                        if random_lr:
                            learn_rate = random.uniform(learning_rate,learning_rate*10)
                        else:
                            learn_rate = learning_rate[i]
                        params_list.append({'params':param_itr[i], 'lr':learn_rate})
                        self.lr_group.append(learn_rate)
                    self.optimizer = torch.optim.RMSprop(params_list,momentum=mom)
                else:
                    self.optimizer = torch.optim.RMSprop(param_itr, learning_rate)

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
        if lr_change_step is not None:
            if gamma == None:
                gamma = 0.5
            # self.scheduler = StepLR(self.optimizer,step_size=lr_change_step,gamma=gamma)
            # self.scheduler = ReduceLROnPlateau(self.optimizer,'max',patience=30)
            if self.rn.assoc_is_param:
                if self.rn.partial_opt:
                    self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=[self.creat_lambda for i in range(len(self.rn.params_kon))])
                    self.lambda_ct = -1
                    self.gamma = gamma
                else:
                    self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=self.assoc_lambda)
            if self.rn.chap_is_param:
                self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=[self.lambda_c,self.lambda_k])
            self.lr_change_step = lr_change_step
        else:
            self.lr_change_step = None

    def assoc_lambda(self, opt_itr):
        new_lr = torch.min(self.rn.kon).item() * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr / curr_lr)
    def creat_lambda(self, opt_itr):
        return(self.gamma)
    def lambda1(self, opt_itr):
        new_lr = torch.min(self.rn.params_k[0]).item() * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr/curr_lr)
    def lambda2(self, opt_itr):
        new_lr = torch.min(self.rn.params_k).item() * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr / curr_lr)
    def lambda_c(self, opt_itr):
        new_lr = torch.min(self.rn.chap_params[0]).item() * 100 * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr / curr_lr)
    def lambda_k(self, opt_itr):
        new_lr = torch.min(self.rn.chap_params[1]).item() * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][1]['lr']
        return(new_lr / curr_lr)
    def lambda5(self, opt_itr):
        new_lr = torch.min(self.rn.params_k[2]).item() * self.lr_group[2]
        curr_lr = self.optimizer.state_dict()['param_groups'][2]['lr']
        return(new_lr / curr_lr)

    def update_counter(self): # Currently does nothing
        lr_ct = 1
        # if self.counter == len(self.rn.params_k):
        #     self.counter=0

    def lambda_master(self, opt_itr):
        # update_counter()
        # print("***** LAMBDA MASTER:  {:d}*****".format(self.lambda_counter))
        # self.counter+=
        self.lambda_ct += 1
        return(torch.min(self.rn.params_k[self.lambda_ct % len(self.rn.params_k)]).\
               item() * self.lr_group[self.lambda_ct%len(self.rn.params_k)] / \
                self.optimizer.state_dict()['param_groups'][self.lambda_ct % len(self.rn.params_k)]['lr'])

    def plot_observable(self, iteration, nodes_list, ax=None):
        t = self.sim_observables[iteration]['steps']

        for key in self.sim_observables[iteration].keys():
            if key == 'steps':
                continue

            elif self.sim_observables[iteration][key][0] in nodes_list:
                data = np.array(self.sim_observables[iteration][key][1])
                if not ax:
                    plt.plot(t, data, label=self.sim_observables[iteration][key][0])
                else:
                    ax.plot(t, data, label=self.sim_observables[iteration][key][0])
        lgnd = plt.legend(loc='best',ncol=3)
        for i in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[i]._sizes = [30]
        plt.title = 'Sim iteration ' + str(iteration)
        ax.set_xscale("log")

        plt.show()

    def plot_yield(self,flux_bool=False):
        steps = np.arange(len(self.yield_per_iter))
        data = np.array(self.yield_per_iter, dtype=np.float)
        flux = np.array(self.flux_per_iter,dtype=np.float)
        plt.plot(steps, data,label='Yield')
        if flux_bool:
            plt.plot(steps,flux,label='Flux')
        #plt.ylim((0, 1))
        plt.title = 'Yield at each iteration'
        plt.xlabel("Yield (%)")
        plt.ylabel("Iterations")
        plt.show()




    def optimize(self,optim='yield',
                 node_str=None,
                 max_yield=0.5,
                 corr_rxns=[[1],[5]],
                 max_thresh=10,
                 lowvar=False,
                 conc_scale=1.0,
                 mod_factor=1.0,
                 conc_thresh=1e-5,
                 mod_bool=True,
                 verbose=False,
                 change_runtime=False,
                 yield_species=-1,
                 creat_yield=-1,
                 varBool=True,
                 chap_mode=1,
                 change_lr_yield=0.98,
                 var_thresh=10):
        print("Reaction Parameters before optimization: ")
        print(self.rn.get_params())

        print("Optimizer State:", self.optimizer.state_dict)
        calc_flux_optim = False
        if optim == 'flux_coeff':
            calc_flux_optim = True
        for i in range(self.optim_iterations):
            # Reset for new simulator
            print("\n")
            print("STEP ",i)
            #print("check 0")
            #print(f"kon:{self.rn.kon}")
            self.rn.reset()

            if self.rn.boolCreation_rxn and change_runtime:
                pass
            else:
                #print("check 1")
                #print(f"kon: {self.rn.kon}")
                sim = self.sim_class(self.rn,
                                     self.sim_runtime,
                                     device=self.dev,
                                     calc_flux=calc_flux_optim)
                #print("check 2")
                #print(f"kon: {self.rn.kon}")


            # Perform simulation
            self.optimizer.zero_grad()
            if self.rn.boolCreation_rxn:
                pass
            elif self.rn.chaperone:
                pass
            else:
                #print("check 3")
                total_yield, total_flux = sim.simulate(optim,
                                                        node_str,
                                                        corr_rxns=corr_rxns,
                                                        conc_scale=conc_scale,
                                                        mod_factor=mod_factor,
                                                        conc_thresh=conc_thresh,
                                                        mod_bool=mod_bool,
                                                        verbose=True,
                                                        yield_species=yield_species
                                                    )
                #print(f"kon:{self.rn.kon}")
                #print("check 4")
            
            #print("check 5")

            self.yield_per_iter.append(total_yield.item())
            # update tracked data
            #print("check 6")
            self.sim_observables.append(self.rn.observables.copy())
            #print("check 7")
            self.sim_observables[-1]['steps'] = np.array(sim.steps)
            #print("check 8")
            self.parameter_history.append(self.rn.kon.clone().detach().to(torch.device("cpu")).numpy())
            #self.parameter_history.append(self.rn.kon.clone().detach().numpy())
            #print("check 9")

            if optim in ['yield', 'time']:
                if optim == 'yield':
                    print(f'Yield on sim. iteration {i} was {str(total_yield.item() * 100)[:4]}%.')
                elif optim == 'time':
                    print(f'Yield on sim iteration {i} was {str(total_yield.item() * 100)[:4]}%')
                    #+ \'Time :', str(cur_time))
                # print(self.rn.copies_vec)
                # preform gradient step
                if i != self.optim_iterations - 1:
                                      
                    if self.rn.homo_rates and self.rn.assoc_is_param:
                        print("check a10")
                        new_params = self.rn.params_kon.clone().detach()
                    else:
                        print("check 10")
                        new_params = self.rn.kon.clone().detach()
                    #print('New reaction rates: ' + str(self.rn.kon.clone().detach()))
                    print('current params:', str(new_params))

                    #Store yield and params data
                    if total_yield-max_yield > 0:
                        if self.rn.chap_is_param:
                            self.final_yields.append([total_yield,dimer_yield,chap_sp_yield])
                            self.dimer_max.append(dimer_max)
                            self.chap_max.append(chap_max)
                            self.endtimes.append(endtime)
                        else:
                            self.final_yields.append(total_yield)

                        self.final_solns.append(new_params)
                        self.final_t50.append(total_flux[0])
                        self.final_t85.append(total_flux[1])
                        self.final_t95.append(total_flux[2])
                        self.final_t99.append(total_flux[3])
                        if self.rn.boolCreation_rxn:
                            self.final_unused_mon.append(unused_monomer)
                            self.curr_time.append(cur_time)

                    if self.rn.assoc_is_param:
                        if self.rn.homo_rates:
                            print("check a11")
                            print("kon: ",self.rn.params_kon)
                            #l_k=self.rn.compute_log_constants(self.rn.params_kon, self.rn.params_rxn_score_vec,scalar_modifier=1.)
                            #k = torch.exp(l_k)

                            k_off=(torch.exp(self.rn.params_rxn_score_vec))*self.rn.params_kon*1e6 #self.rn._C0
                            print("K_off: ",k_off)
                            k=torch.cat([self.rn.params_kon, k_off], dim=0).to(self.dev)


                            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                            physics_penalty = torch.sum(F.relu(-1 * (k - curr_lr * 10))).to(self.dev) + torch.sum(F.relu(1 * (k - 10))).to(self.dev) # stops zeroing or negating params
                            #physics_penalty = nn.ReLU(-1 * (k - curr_lr * 10)).to(self.dev) + nn.ReLU(1 * (k - 10)).to(self.dev) # stops zeroing or negating params
                            cost = -total_yield + physics_penalty

                            print("cost: ",cost)

                            cost.backward(retain_graph=True)

                            print("Grad: ",self.rn.params_kon.grad)
                            print("Finite:",torch.isfinite(self.rn.params_kon.grad))
                            print("K:", k)
                            print("curr_lr",curr_lr)
                        else:
                            print("check 11")
                            log_k=self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec, scalar_modifier=1.)
                            # l_kon = torch.log(self.rn.kon)       
                            # l_koff = (self.rn.rxn_score_vec) + l_kon + torch.log(self.rn._C0).to(self.dev)
                            # l_k = torch.cat([l_kon, l_koff], dim=0)
                            # log_k=l_k.clone().to(self.dev)
                            print("rn score: ",self.rn.rxn_score_vec)
                            print("K_on: ",self.rn.kon)
                            print("c0: ",self.rn._C0)

                            #with torch.no_grad:
                            #k_off=(torch.exp(self.rn.rxn_score_vec))*self.rn.kon*1e6 #self.rn._C0
                            #print("K_off: ",k_off)

                            #k=torch.cat([self.rn.kon, k_off], dim=0).to(self.dev)

                            #print("log_k: ",log_k)

                            k = torch.exp(log_k)
                            print("K: ",k)

                            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                            physics_penalty = torch.sum(10 * F.relu(-1 * (k - curr_lr * 10))).to(self.dev) + torch.sum(10 * F.relu(1 * (k - max_thresh))).to(self.dev)
                            if lowvar:
                                mon_rxn = self.rn.rxn_class[1]
                                var_penalty = 100*F.relu(1 * (torch.var(k[mon_rxn])))
                                print("Var penalty: ",var_penalty,torch.var(k[:3]))
                            else:
                                var_penalty=0
                            # ratio_penalty = 1000*F.relu(1*((torch.max(k[3:])/torch.min(k[:3])) - 500 ))
                            # print("Var penalty: ",var_penalty,torch.var(k[:3]))
                            # print("Ratio penalty: ",ratio_penalty,torch.max(k[3:])/torch.min(k[:3]))

                            # dimer_penalty = 10*F.relu(1*(k[16] - self.lr*20))+10*F.relu(1*(k[17] - self.lr*20))+10*F.relu(1*(k[18] - self.lr*20))
                            cost = -total_yield + physics_penalty + var_penalty #+ dimer_penalty#+ var_penalty #+ ratio_penalty
                            print("cost:",cost)
                            cost.backward()
                            print("Grad: ",self.rn.kon.grad)
                            print("Finite:",torch.isfinite(self.rn.kon.grad))
                            print("K:", k)
                            print("curr_lr",curr_lr)

                    
                    # self.scheduler.step(metric)
                    if (self.lr_change_step is not None) and (total_yield>=change_lr_yield):
                        change_lr = True
                        print("Curr learning rate : ")
                        for param_groups in self.optimizer.param_groups:
                            print(param_groups['lr'])
                            if param_groups['lr'] < 1e-2:
                                change_lr=False
                        if change_lr:
                            self.scheduler.step()


                    #Changing learning rate
                    if (self.lr_change_step is not None) and (i%self.lr_change_step ==0) and (i>0):
                        print("New learning rate : ")
                        for param_groups in self.optimizer.param_groups:
                            print(param_groups['lr'])

                    self.optimizer.step()

                    #print("Previous reaction rates: ",str(self.rn.kon.clone().detach()))


            values = psutil.virtual_memory()
            mem = values.available / (1024.0 ** 3)
            if mem < .5:
                # kill program if it uses to much ram
                print("Killing optimization because too much RAM being used.")
                print(values.available,mem)
                return self.rn
            if i == self.optim_iterations - 1:
                print("optimization complete")
                print("Final params: " + str(new_params))
                return self.rn

            del sim

