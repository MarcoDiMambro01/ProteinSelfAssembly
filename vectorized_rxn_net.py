import sys
#sys.path.append("../../")
#sys.path.append("../")

from typing import Tuple

from .reaction_network import ReactionNetwork, gtostr

import networkx as nx
import torch
from torch import DoubleTensor as Tensor
from torch import rand
from torch import nn



class VectorizedRxnNet:
    """
    Provides a lightweight class that represents the core information needed for
    simulation as torch tensors. Acts as a base object for optimization simulations.
    Data structure is performance optimized, not easily readable/accessible.

    Units:
    units of Kon assumed to be [copies]-1 S-1, units of Koff S-1
    units of reaction scores are treated as J * c / mol where c is a user defined scalar
    """

    def __init__(self,
                 rn: ReactionNetwork,
                 assoc_is_param=True,
                 copies_is_param=False,
                 chap_is_param=False,
                 dissoc_is_param=False,
                 dG_is_param=False,
                 cplx_dG=0,
                 mode=None,
                 type='a',
                 dev='cpu',
                 coupling=False,
                 cid={-1:-1},
                 rxn_coupling=False,
                 rx_cid={-1:-1},
                 std_c=1e6,
                 optim_rates=None,
                 slow_rates=None,
                 slow_ratio=1):
        """

        :param rn: The reaction network template
        :param assoc_is_param: whether the association constants should be treated as parameters for optimization
        :param copies_is_param: whether the initial copy numbers should be treated as parameters for optimization
        :param dev: the device to use for torch tensors
        :param coupling : If two reactions have same kon. i.e. Addition of new subunit is same as previous subunit
        :param cid : Reaction ids in a dictionary format. {child_reaction:parent_reaction}. Set the rate of child_reaction to parent_reaction
        """
        #rn.reset()
        self.dev = torch.device(dev)
        self._avo = Tensor([6.02214e23])  # copies / mol
        self._R = Tensor([8.314])  # J / mol * K
        self._T = Tensor([273.15])  # K
        self._C0 = Tensor([std_c])    #Std. Conc in uM
        #self.dev=dev

        #Variables for zeroth order reactions
        self.boolCreation_rxn = rn.boolCreation_rxn
        self.creation_nodes = rn.creation_nodes
        self.creation_rxn_data = rn.creation_rxn_data
        self.titration_end_conc=rn.titration_end_conc
        if self.boolCreation_rxn and self.titration_end_conc != -1:
            self.titration_time_map={v['uid'] : self.titration_end_conc/v['k_on'] for v in self.creation_rxn_data.values()}

        #Variables for Destruction order reactions
        self.boolDestruction_rxn = rn.boolDestruction_rxn
        self.destruction_nodes = rn.destruction_nodes
        self.destruction_rxn_data = rn.destruction_rxn_data

        self.chaperone = rn.chaperone
        if self.chaperone:
            self.chap_uid_map = rn.chap_uid_map
            self.optimize_species=rn.optimize_species

        self.M, self.kon, self.rxn_score_vec, self.copies_vec = self.generate_vectorized_representation(rn)
        print("check generate:")
        print("dev kon:",self.kon.get_device())
        print("dev rxn:",self.rxn_score_vec.get_device())
        self.rxn_coupling = coupling
        self.coupling = rn.rxn_coupling
        self.num_monomers = rn.num_monomers
        self.max_subunits = rn.max_subunits
        self.homo_rates = rn.homo_rates



        if optim_rates is not None:
            self.partial_opt = True
            self.optim_rates = optim_rates
        else:
            self.partial_opt = False

        self.slow_rates = slow_rates
        self.slow_ratio=slow_ratio
        self.cid = cid
        self.rx_cid = rn.rxn_cid
        self.coup_map = {}
        self.rxn_class = rn.rxn_class
        self.dG_map = rn.dG_map

        
        if self.homo_rates == True:
            print("Check a1")
            self.params_kon = torch.zeros([len(self.rxn_class.keys())],requires_grad=True).double()
            self.params_rxn_score_vec = torch.zeros([len(self.rxn_class.keys())]).double()
            counter=0
            for k,rid in self.rxn_class.items():

                self.params_kon[counter] = self.kon.clone().detach()[rid[0]]  ##Get the first uid of each class.Set that as the param for that class of rxns
                self.params_rxn_score_vec[counter] = self.rxn_score_vec[rid[0]]
                counter+=1
            self.params_kon.requires_grad_(True)
            self.initial_params = Tensor(self.params_kon).clone().detach()
            print("check a2")

        else:
            self.initial_params = Tensor(self.kon).clone().detach()


        self.initial_copies = self.copies_vec.clone().detach()
        self.assoc_is_param = assoc_is_param
        self.copies_is_param = copies_is_param
        self.dissoc_is_param = dissoc_is_param
        self.dG_is_param = dG_is_param
        self.chap_is_param=chap_is_param


        if assoc_is_param:
            if self.coupling:
                self.params_kon = nn.Parameter(self.params_kon, requires_grad=True)
            elif self.partial_opt:
                # self.params_kon = nn.Parameter(self.params_kon, requires_grad=True)
                self.kon.requires_grad_(True)
            elif self.homo_rates:
                self.params_kon = nn.Parameter(self.params_kon, requires_grad=True)
            else:
                self.kon = nn.Parameter(self.kon, requires_grad=True)
        
        self.observables = rn.observables
        self.flux_vs_time = rn.flux_vs_time
        self.is_energy_set = rn.is_energy_set
        self.num_monomers = rn.num_monomers
        self.reaction_ids = []
        self.reaction_network = rn

        print("Shifting to device: ", dev)
        self.to(dev)
        print("check after shift:")
        print("dev kon:",self.kon.get_device())
        print("dev rxn:",self.rxn_score_vec.get_device())


    def reset(self, reset_params=False):
        self.copies_vec = self.initial_copies.clone()
        
        if reset_params:
            if self.homo_rates:
                self.params_kon = nn.Parameter(self.initial_params.clone(), requires_grad=True)
            else:
                self.kon = nn.Parameter(self.initial_params.clone(), requires_grad=True)
        for key in self.observables:
            self.observables[key] = (self.observables[key][0], [])

    def get_params(self):
        if self.assoc_is_param:
            if self.coupling:
                return [self.params_kon]
            elif self.partial_opt:
                return self.params_kon
            elif self.homo_rates:
                return [self.params_kon]
            else:
                return [self.kon]
        elif self.dissoc_is_param:
            return self.params_koff
        elif self.dG_is_param:
            if self.dG_mode==1:
                return self.params_k
            else:
                #return [self.params_k]
                return self.params_k
        elif self.chap_is_param:
            return self.chap_params

    def to(self, dev):
        self.M = self.M.to(dev)
        self._avo = self._avo.to(dev)
        self._R = self._R.to(dev)
        self._T = self._T.to(dev)
        self._C0 = self._C0.to(dev)

        if self.homo_rates and self.assoc_is_param:
            self.params_kon = nn.Parameter(self.params_kon.data.clone().detach().to(dev), requires_grad=True)
            self.params_rxn_score_vec = self.params_rxn_score_vec.to(dev)
            self.kon = self.kon.to(dev)

            #mine
            #self.kon = nn.Parameter(self.kon.data.clone().detach().to(dev)
        else:
            self.kon = nn.Parameter(self.kon.data.clone().detach().to(dev), requires_grad=True)
        self.copies_vec = self.copies_vec.to(dev)
        self.initial_copies = self.initial_copies.to(dev)
        self.rxn_score_vec = self.rxn_score_vec.to(dev)
        self.dev = dev
        return self

    def generate_vectorized_representation(self, rn: ReactionNetwork) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Get a matrix mapping reactions to state updates. Since every reaction has a forward
        and reverse, dimensions of map matrix M are (rxn_count*2 x num_states). The forward
        reactions are placed in the first half along the reaction axis, and the reverse
        reactions in the second half. Note that the reverse map is simply -1 * the forward map.

        Returns: M, k_on_vec, rxn_score_vec, copies_vec
            M: Tensor, A matrix that maps a vector in reaction space to a vector in state space
                shape (num_states, rxn_count * 2).
            k_vec: Tensor, A vector of rate constants in reaction space. shape (rxn_count).
            rxn_score_vec: Tensor, A vector of the rosetta resolved reaction scores in reaction space.
                shape (rxn_count), though note both halves are symmetric.
            copies_vec: Tensor, A vector of the state copy numbers in state space. shape (num_states).
        """
        num_states = len(rn.network.nodes)
        # initialize tensor representation dimensions
        M = torch.zeros((num_states, rn._rxn_count * 2)).double()
        kon = torch.zeros([rn._rxn_count], requires_grad=True).double()
        rxn_score_vec = torch.zeros([rn._rxn_count]).double()
        copies_vec = torch.zeros([num_states]).double()

        for n in rn.network.nodes():
            # print(RN.gtostr(rn.network.nodes[n]['struct']))
            copies_vec[n] = rn.network.nodes[n]['copies']
            # print("Reactant Sets:")

            
            for r_set in rn.get_reactant_sets(n):
                r_tup = tuple(r_set)
                #print(r_tup)
                data = rn.network.get_edge_data(r_tup[0], n)
                reaction_id = data['uid']
                try:
                    kon[reaction_id] = data['k_on']
                except Exception:
                    kon[reaction_id] = 1.
                rxn_score_vec[reaction_id] = data['rxn_score']
                # forward
                if len(r_tup) == 2:   #Bimolecular reaction; Two reactants
                    M[n, reaction_id] = 1.
                    for r in r_tup:
                        M[r, reaction_id] = -1.
                elif len(r_tup) == 1:  #Only one reactant; Have to check if its a Bimolecular
                    if rn.network.nodes[n]['struct'].number_of_edges()>0:
                        #This means there is a bond formation. Therefore it has to be a Bimolecular
                        #But it has same reactant. Reaction stoich = 2
                        M[n,reaction_id] = 1.
                        for r in r_tup:
                            M[r,reaction_id] = -2.
                    else:
                        #If edges are zero then this species is a monomer.
                        #If it has only one reactant then it is in a dissociation. Possibly chaperone
                        if self.chaperone:
                            M[n,reaction_id] = 1.
                            M[r_tup[0],reaction_id] = -1.

        # generate the reverse map explicitly
        # M[0,11]=0
        M[:, rn._rxn_count:] = -1 * M[:, :rn._rxn_count]
        # print("Before: ",M)
        if self.chaperone:
            for chap,uids in self.chap_uid_map.items():
                # M[chap,uid] = 0
                # M[:,-1] = 0
                for id in uids:
                    M[:,rn._rxn_count+id] = 0

        #To adjust for creation reactions. No reversible destruction
        if self.boolCreation_rxn or self.boolDestruction_rxn:
            num_creat_dest_rxn = len(self.creation_rxn_data) + len(self.destruction_rxn_data)

            new_M = M[:,:-num_creat_dest_rxn:]
            return new_M,kon,rxn_score_vec, copies_vec
        # print(M)
        # print(kon)
        # print(copies_vec)
        #print("Stoichiometric Matrix: ",M)
        print("Reaction rates: ",kon)
        print('dGs: ', rxn_score_vec)
        print("Species Concentrations: ",copies_vec)

        return M, kon, rxn_score_vec, copies_vec

    def compute_log_constants(self, kon: Tensor, dGrxn: Tensor, scalar_modifier) -> Tensor:
        """
        Returns log(k) for each reaction concatenated with log(koff) for each reaction
        """
        # above conversions cancel
        # std_c = Tensor([1e6])  # units umols / L
        print("compute log")
        print("k_on: ",kon)
        print("check kon device:",kon.get_device())
        print("rn score: ",dGrxn)
        print("check rn device:",dGrxn.get_device())


        l_kon = torch.log(kon)  # umol-1 s-1
        print("l_kon: ",l_kon)
        print("check l_kon device:",l_kon.get_device())
        # l_koff = (dGrxn * scalar_modifier / (self._R * self._T)) + l_kon + torch.log(std_c)       #Units of dG in J/mol
        #l_koff = (dGrxn * scalar_modifier) + l_kon + torch.log(self._C0).to(self.dev)
        l_koff = (dGrxn) + l_kon + torch.log(self._C0).to(self.dev)

        # print(torch.exp(l_kon))
        # print(torch.exp(l_koff))       #Units of dG in J/mol
        l_k = torch.cat([l_kon, l_koff], dim=0)

        return l_k.clone().to(self.dev)
        #return l_k.clone()
    
    def update_reaction_net(self, rn, scalar_modifier: int = 1):
            print("check a-update")
            print("dev kon:",self.kon.get_device())
            print("dev rxn:",self.rxn_score_vec.get_device())


            for n in rn.network.nodes:
                rn.network.nodes[n]['copies'] = self.copies_vec[n].item()
                for r_set in rn.get_reactant_sets(n):
                    r_tup = tuple(r_set)
                    reaction_id = rn.network.get_edge_data(r_tup[0], n)['uid']
                    for r in r_tup:
                        print("k_on: ",self.kon)
                        print("rn score: ",self.rxn_score_vec)
                        k = self.compute_log_constants(self.kon, self.rxn_score_vec, scalar_modifier)
                        k = torch.exp(k)
                        # print("RATEs: ",k)
                        rn.network.edges[(r, n)]['k_on'] = k[reaction_id].item()
                        rn.network.edges[(r, n)]['k_off'] = k[reaction_id + int(k.shape[0] / 2)].item()
            return rn




















    def get_log_copy_prod_vector(self):
        """
          get the vector storing product of copies for each reactant in each reaction.
        Returns: Tensor
            A tensor with shape (rxn_count * 2)
        """
        r_filter = -1 * self.M.T.clone()        #Invert signs of reactants amd products.
        # r_filter = -1 * M.T.clone()
        r_filter[r_filter == 0] = -1            #Also changing molecules not involved in reactions to -1. After this, only reactants in each rxn are positive.
        # r_filter[6,3]=1
        # print(r_filter)
        #Old code
        # c_temp_mat = torch.mul(r_filter, self.copies_vec)
        # l_c_temp_mat = torch.log(c_temp_mat)
        # l_c_temp_mat[c_temp_mat < 0] = 0      #Make zero for non-reactants with non-zero copy number -> Flag 1
        # c_mask = r_filter + self.copies_vec
        # l_c_temp_mat[c_mask == -1] = 0  # 0 = log(1)  #Make zero for non-reactants with zero copy number -> Flag 2
        # l_c_prod_vec = torch.sum(l_c_temp_mat, dim=1)  # compute log products

        #New code
        #Use a non_reactant mask
        nonreactant_mask = r_filter<0       #Combines condition of Flag1 and Flag2. Basically just selecting all non_reactants w.r.t to each reaction
        c_temp_mat = torch.pow(self.copies_vec,r_filter)        #Different from previous where torch.mul was used. The previous only works for stoich=1, since X^1=X*1. But in mass action kinetics, conc. is raised to the power
        l_c_temp_mat = torch.log(c_temp_mat)                #Same as above
        l_c_temp_mat[nonreactant_mask]=0
        # print(l_c_temp_mat)                    #Setting all conc. values of non-reactants to zero before taking the sum. Matrix dim - No. of rxn x No. of species
        l_c_prod_vec = torch.sum(l_c_temp_mat, dim=1)       #Summing for each row to get prod of conc. of reactants for each reaction
        # print("Actual Prod: ",torch.exp(l_c_prod_vec))
        return l_c_prod_vec

    

    def get_max_edge(self,n):
        """
        Calculates the max rate (k_on) for a given node
        To find out the maximum flow path to the final complex starting from the current node.

        Can also calculate the total rate of consumption of a node by summing up all rates.
        Can tell which component is used quickly.
        """
        try:
            edges = self.reaction_network.network.out_edges(n)
            #Loop over all edges
            #Get attributes
            kon_max = -1
            next_node = -1

            kon_sum = 0
            total_flux_outedges = 0
            total_flux_inedges = 0
            if len(edges)==0:
                return(False)

            for edge in edges:
                data = self.reaction_network.network.get_edge_data(edge[0],edge[1])
                #print(data)
                #Get uid
                uid = data['uid']

                #Get updated kon
                temp_kon = self.kon[uid]
                kon_sum+=temp_kon

                if temp_kon > kon_max:
                    kon_max = temp_kon
                    next_node=edge[1]

            return(kon_max,next_node,kon_sum)
        except Exception as err:
            raise(err)

    def get_node_flux(self,n):
        node_map = {}
        for node in self.reaction_network.network.nodes():
            node_map[gtostr(self.reaction_network.network.nodes[node]['struct'])] = node
        total_flux_outedges = 0
        total_flux_inedges = 0
        #Go over all the out edges
        edges_out = self.reaction_network.network.out_edges(n)
        if len(edges_out)>0:

            for edge in edges_out:
                data = self.reaction_network.network.get_edge_data(edge[0],edge[1])
                #print(data)
                #Get uid
                uid = data['uid']

                #Get updated kon
                temp_kon = self.kon[uid]

                #Calculate k_off also
                std_c = Tensor([1e6])
                l_kon = torch.log(temp_kon)
                l_koff = (self.rxn_score_vec[uid]) + l_kon + torch.log(std_c).to(self.dev)
                koff = torch.exp(l_koff)

                #Getting conc. of reactants and products
                #Get product
                prod = gtostr(self.reaction_network.network.nodes[edge[1]]['struct'])
                #Get other reactant
                react = "".join(sorted(list(set(prod) - set(gtostr(self.reaction_network.network.nodes[edge[0]]['struct']) ))))

                #Net flux from this edge = Generation - consumption
                edge_flux = koff*self.copies_vec[edge[1]] - temp_kon*(self.copies_vec[edge[0]])*(self.copies_vec[node_map[react]])
                #edge_flux = koff*vec_rn.copies_vec[edge[1]]

                # print("Reaction: ", RN.gtostr(rn.network.nodes[edge[0]]['struct']), "+",react," -> ",prod)
                # print("Net flux: ",edge_flux)
                # print("kon : ",temp_kon)
                # print("koff: ",koff)
                # print("Reaction data OUTWARD: ")
                # print(data)

                total_flux_outedges+=edge_flux

        #Now go over all the in edges
        edges_in = self.reaction_network.network.in_edges(n)
        react_list = []
        if len(edges_in) > 0:
            for edge in edges_in:
                if edge[0] in react_list:
                    continue
                data = self.reaction_network.network.get_edge_data(edge[0],edge[1])
                uid = data['uid']


                #Get generation rates; which would be kon
                temp_kon = self.kon[uid]

                #Get consumption rates; which is k_off
                std_c = Tensor([1e6])
                l_kon = torch.log(temp_kon)
                l_koff = (self.rxn_score_vec[uid]) + l_kon + torch.log(std_c).to(self.dev)
                koff = torch.exp(l_koff)

                #Get conc. of reactants and products
                prod = gtostr(self.reaction_network.network.nodes[edge[1]]['struct'])
                #Get other reactant
                react = "".join(sorted(list(set(prod) - set(gtostr(self.reaction_network.network.nodes[edge[0]]['struct']) ))))
                react_list.append(node_map[react])
                #Net flux from this edge = Generation - consumption
                edge_flux_in = temp_kon*(self.copies_vec[edge[0]])*(self.copies_vec[node_map[react]])- koff*self.copies_vec[edge[1]]
                #edge_flux_in = koff*vec_rn.copies_vec[edge[1]]



                # print("Reaction: ", prod ," -> ",RN.gtostr(self.reaction_network.network.nodes[edge[0]]['struct']), "+",react)
                # print("Net flux: ",edge_flux_in)
                # print("kon : ",temp_kon)
                # print("koff: ",koff)
                # print("Raction data INWARD: ")
                # print(data)

                total_flux_inedges+=edge_flux_in
        net_node_flux = total_flux_outedges + total_flux_inedges

        return(net_node_flux)

    def get_reaction_flux(self):
        react_flux = torch.zeros(1,2*self.reaction_network._rxn_count)
        uid_dict = {}
        react_dict = {}
        for n in self.reaction_network.network.nodes():
            #print(n)
            #print(rn.network.nodes()[n])
            for r_set in self.reaction_network.get_reactant_sets(n):
                r_tup = tuple(list(r_set)+[n])
                data = self.reaction_network.network.get_edge_data(r_tup[0], n)
                reaction_id = data['uid']
                react_dict[r_tup]=reaction_id
            for k,v in self.reaction_network.network[n].items():
                uid = v['uid']
                r1 = set(gtostr(self.reaction_network.network.nodes[n]['struct']))
                p = set(gtostr(self.reaction_network.network.nodes[k]['struct']))
                r2 = p-r1
                reactants = ("".join(r1),"".join(r2))
                uid_dict[(n,k)] = uid


        #Get flux in each reaction
        #The react_dict is of the form {(reactant,reactant,product):uid}
        #Easy access to species concentrations from uid to calc flux
        for species,uid in react_dict.items():
            #Get generation rates; which would be kon
            temp_kon = self.kon[uid]
            #Get consumption rates; which is k_off
            std_c = Tensor([1e6])
            l_kon = torch.log(temp_kon)
            l_koff = (self.rxn_score_vec[uid]) + l_kon + torch.log(std_c)
            koff = torch.exp(l_koff)

            #Net flux from this edge = Generation - consumption
            assoc_flux = temp_kon*(self.copies_vec[species[0]])*(self.copies_vec[species[1]])
            reverse_flux = koff*self.copies_vec[species[2]]
            react_flux[0,uid]=assoc_flux
            react_flux[0,self.reaction_network._rxn_count + uid] = reverse_flux

        return(react_flux)


    def calculate_total_flux(self):
        net_flux = {}
        for n in self.reaction_network.network.nodes():
            n_str = gtostr(self.reaction_network.network.nodes[n]['struct'])
            node_flux = self.get_node_flux(n)
            net_flux[gtostr(self.reaction_network.network.nodes[n]['struct'])] = node_flux

        return(net_flux)

    def compute_total_dG(self,k):
        on_rates = k[self.rxn_class[1]]
        off_rates = k[int(k.shape[0] / 2):][self.rxn_class[1]]

        Keq = torch.prod(on_rates*self._C0/off_rates)
        dG = -1*torch.log(Keq)
        return(dG)
