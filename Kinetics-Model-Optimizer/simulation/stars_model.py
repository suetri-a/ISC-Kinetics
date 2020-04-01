import numpy as np 
import scipy as sp
import pandas as pd
import networkx as nx
import os

from scipy.optimize import nnls

import utils.stars as stars
import utils.stars.rto as rto
from utils.utils import mkdirs, solve_const_lineq
from .kinetic_cell_base import KineticCellBase

class STARSModel(KineticCellBase):

    @staticmethod
    def modify_cmd_options(parser):
        # STARS Simulation Parameters
        parser.add_argument('--stars_sim_folder', type=str, default='stars', help='folder where run files are written')
        parser.add_argument('--stars_base_model', type=str, default='master', help='which model to use [master]')
        parser.add_argument('--stars_exe_path', type=str, default='"C:\\Program Files (x86)\\CMG\\STARS\\2017.10\\Win_x64\\EXE\\st201710.exe"', 
                            help='path where to execute CMG-STARS')

        # Add reaciton model parameters
        parser.add_argument('--heat_reaction', type=str, default=None, help='heats of reaction')
        parser.add_argument('--pre_exp_fwd', type=str, default=None, help='forward pre-exponential factors in Arrhenius reaction model')
        parser.add_argument('--act_eng_fwd', type=str, default=None, help='foward activation energies in Arrhenius reaction model')
        parser.add_argument('--pre_exp_rev', type=str, default=None, help='reverse pre-exponential factors in Arrhenius reaction model')
        parser.add_argument('--act_eng_rev', type=str, default=None, help='reverse activation energies in Arrhenius reaction model')

        return parser


    def __init__(self, opts):
        super().__init__(opts)

        self.base_model = opts.stars_base_model
        self.base_filename = 'stars_model'
        self.cd_path = os.path.join(self.results_dir + opts.stars_sim_folder, '')
        mkdirs([self.cd_path])
        self.exe_path = opts.stars_exe_path
        
        vkc_class = rto.get_vkc_model(opts.stars_base_model)
        self.vkc_model = vkc_class(folder_name=self.cd_path, input_file_name=self.base_filename)
        self.num_stars_calls = 0


    def initialize_parameters(self, opts):
        '''
        Initialize the update functions for each reaction depending on number of unknown
            molecular weights, reaction type, etc. and initialize the corresponding parameters
            (and their parameter types) for each reaction. 

        '''

        # Initialize the lists for update functions, parameters, and parameter types
        self.param_types = []
        
        # Determine reactions for mappings
        self.get_mappings() 

        # Iterate over reactions and add parameters
        for i in range(self.num_rxns):
            self.param_types.append(['preexp',i])
            self.param_types.append(['acteng',i])
            param_types_temp = self.add_stoic_coeff_params(i) # stoiciometric constants
            self.param_types += param_types_temp


    def get_mappings(self):
        '''
        Determine reactions to use for which stage of mappings
        '''

        # Perform maximal matching with given weighting scheme
        U, V, G = self.get_rxn_fuel_graph()
        matches = nx.algorithms.bipartite.minimum_weight_full_matching(G)

        # Reactions for mapping material properties or coefficients
        self.map_rxns_material = [U.index(matches[v]) for v in V] # rxns to solve for fuels
        self.map_rxns_coeff = [r for r in range(self.num_rxns) if r not in self.map_rxns_material] # coeffs

        # Reactions for determining coefficients before material mapping
        oxy_fuels = self.get_oxy_fuels() # coeffs on material rxns
        self.map_rxns_oxy = [r for r in self.map_rxns_material if all([c not in oxy_fuels for c in self.reac_names[r]+self.prod_names[r]])]


    def add_stoic_coeff_params(self, r):
        '''
        Add stoiciometric coefficient parameters for a reaction.

        Input:
            r - reaction number
        '''

        # Get reaction information
        spec_rxn = self.reac_names[r] + self.prod_names[r]
        spec_const = [c[2] for c in self.rxn_constraints if c[0]==r+1]
        weight = len(spec_rxn) - len(spec_const) - 1 

        if r in self.map_rxns_material:
            if r in self.map_rxns_oxy:
                pk = np.maximum(0, weight - 1)
            else:
                pk = np.maximum(0, weight)
        else:
            pk = np.maximum(0, weight - len(self.opts.balances))

        # Iterate over species to add parameters
        param_count, spec_ind = 0, 1
        param_types_temp = []
        while param_count < pk:
            if spec_rxn[spec_ind] not in spec_const:
                param_types_temp.append(['stoic', r, spec_rxn[spec_ind]])
                param_count+=1
            spec_ind+=1
        return param_types_temp


    def get_rxn_fuel_graph(self):
        '''
        Generate bipartitate graph with appropriate weights
        '''

        # Initialize bipartite graph
        G = nx.Graph() 
        U = ['Reaction {}'.format(i+1) for i in range(self.num_rxns)] # Create partition of reaction nodes
        V = [c for c in self.comp_names if c in self.pseudo_fuel_comps] # Create partition of unknown fuels
        G.add_nodes_from(U, bipartite = 0)
        G.add_nodes_from(V, bipartite = 1)
        
        spec_counts = np.array([len(self.reac_names[i] + self.prod_names[i]) for i in range(self.num_rxns)])
        spec_const = np.array([len([c[2] for c in self.rxn_constraints if c[0]==r+1]) for r in range(self.num_rxns)])
        weights = spec_counts - spec_const - 1
        
        oxy_fuels = self.get_oxy_fuels()
        for i, _ in enumerate(U):
            for j, v in enumerate(V):
                if v in oxy_fuels:
                    G.add_edge(U[i], V[j], weight = np.maximum(0,weights[i]))
                else:
                    G.add_edge(U[i], V[j], weight = np.maximum(0,weights[i]-1))

        return U, V, G


    def get_rxn_fuels(self, rxn_number):
        '''
        Get indices of fuel species for a given reaction
        '''
        rxn_spec_names = self.reac_names[rxn_number] + self.prod_names[rxn_number]
        inds = [self.comp_names.index(spec) for spec in list(set(rxn_spec_names) & set(self.opts.fuel_comps))]
        specs = [self.comp_names[i] for i in inds]
        return specs, inds


    def get_rxn_nonfuels(self, rxn_number):
        '''
        Get indices of fuel species for a given reaction
        '''
        rxn_spec_names = self.reac_names[rxn_number] + self.prod_names[rxn_number]
        inds = [self.comp_names.index(spec) for spec in list(set(rxn_spec_names) - set(self.opts.fuel_comps))]
        specs = [self.comp_names[i] for i in inds]
        return specs, inds


    def get_oxy_fuels(self):
        '''
        Get list of oxygenated fuels. We define this as a fuel that is a product of oxygen or an oxygenated fuel.
            Oxygenated fuels will have at least one walk between them and oxygen in the checmical reaction graph. 
        '''
        
        reac_coeffs, prod_coeffs = np.zeros((self.num_rxns, self.num_comp)), np.zeros((self.num_rxns, self.num_comp))
        for r in range(self.num_rxns):
            for n in self.reac_names[r]:
                reac_coeffs[r, self.comp_names.index(n)] = 1.0
            for n in self.prod_names[r]:
                prod_coeffs[r, self.comp_names.index(n)] = 1.0

        M = np.matmul(np.not_equal(reac_coeffs,0).T, np.not_equal(prod_coeffs,0)) # Calculate adjacency matrix
        S = np.greater(sp.linalg.expm(M), 0)[self.comp_names.index('O2'), :] # Calculate paths, select O2 row

        # If there is at least one rxn_number - 1 walk between O2 and the fuel species
        #   Note: this works because graphs in combustion equations are acyclic
        oxy_fuels = [f for f in self.pseudo_fuel_comps if S[self.comp_names.index(f)]]

        return oxy_fuels


    ###############################################################
    ##### PARAMETER MAPPING FUNCTIONS #####
    ###############################################################

    def map_parameters(self, x):
        '''
        Function to map parameters from a single vector (used by the optimizer)
            to the attributes of the KineticCell object. 

        Inputs:
            params - vector of the reaction parameters

        '''
        
        if any(p is None for p in x): # Check that there are no non-type parameters
            raise Exception('map_params() not implemented for None type parameters.')


        # Reset parameters
        reac_coeffs, prod_coeffs = np.zeros((self.num_rxns, self.num_comp)), np.zeros((self.num_rxns, self.num_comp))
        oxy_fuels = self.get_oxy_fuels()
        for c in self.pseudo_fuel_comps:
            for b in self.balances:
                if b == 'O' and c not in oxy_fuels:
                    self.balance_dict[b][self.comp_names.index(c)] = 0
                else:
                    self.balance_dict[b][self.comp_names.index(c)] = np.nan
        
        for r in range(self.num_rxns):
            for i, c in enumerate(self.comp_names):
                # If c is reactant species in rxn r
                if c in self.reac_names[r]:
                    # If reactant fuel set coefficient to 1
                    if c in self.fuel_names+self.pseudo_fuel_comps:
                        reac_coeffs[r,i] = 1
                    else:
                        reac_coeffs[r,i] = np.nan
                    prod_coeffs[r,i] = 0

                # If c is product species
                elif c in self.prod_names[r]:
                    prod_coeffs[r,i] = np.nan
                    reac_coeffs[r,i] = 0

        reac_coeffs, prod_coeffs, act_energy, preexp_fac = self.parse_params(x, reac_coeffs, prod_coeffs)


        # Map reactions using mapping algorithm
        if self.map_rxns_oxy:
            reac_coeffs, prod_coeffs = self.oxy_solver(reac_coeffs, prod_coeffs)

        # Map reactions using balance equations
        reac_coeffs, prod_coeffs = self.balance_solver(reac_coeffs, prod_coeffs)
        
        # Map coefficient solver reactions
        if self.map_rxns_coeff:
            reac_coeffs, prod_coeffs = self.coeff_solver(reac_coeffs, prod_coeffs)

        reac_orders = 1.0*np.greater(reac_coeffs, 0)
        prod_orders = np.zeros((self.num_rxns, self.num_comp))

        return reac_coeffs, prod_coeffs, reac_orders, prod_orders, act_energy, preexp_fac


    def parse_params(self, x, reac_coeffs, prod_coeffs):
        '''
        Parse through parameter vector to update internal parameters or overwrite nan 
            data in the stoichiometric coefficients. 

        '''

        act_eng_fwd, pre_exp_fwd = np.zeros((self.num_rxns)), np.zeros((self.num_rxns))

        # Iterate over parameters to parse
        for i, p in enumerate(self.param_types):
            # Pre-exponential factors
            if p[0] == 'preexp':
                pre_exp_fwd[p[1]] = np.exp(x[i]) if self.log_params else x[i]
            
            # Activation energies
            elif p[0] == 'acteng': 
                act_eng_fwd[p[1]] = np.exp(x[i]) if self.log_params else x[i]
            
            # Stoiciometric coefficients
            elif p[0] == 'stoic':
                if p[2] in self.reac_names[p[1]]:
                    reac_coeffs[p[1], self.comp_names.index(p[2])] = x[i]

                if p[2] in self.prod_names[p[1]]:
                    prod_coeffs[p[1], self.comp_names.index(p[2])] = x[i]

            else:
                raise Exception('Invalid parameter type entered.')
        
        return reac_coeffs, prod_coeffs, act_eng_fwd, pre_exp_fwd


    def get_constraint_matrix(self, r, comps = None):
        '''
        Create matrix to map from independent species in reaction r to all species

        Constraint matrix M is of form:
            M[species i, independent species j] = c_{ij}
        where c_{ij} is a constant such that
            nu_{ri} = c_{ij} nu_{rj}
        This allows us to enforce constraint by computing
            All species = M x Independent species

        '''

        if comps is None:
            comps = self.comp_names

        independent_specs = self.get_independent_species(r)

        M = np.zeros((len(comps), len(independent_specs)))

        for i, c in enumerate(comps):
            if c in independent_specs:
                M[i, independent_specs.index(c)] = 1

        for C in self.rxn_constraints:
            if C[0] - 1 == r:
                M[comps.index(C[2]), independent_specs.index(C[1])] = C[3]

        return M


    def get_independent_species(self, r):
        '''
        Get the independent species in a reaction r

        '''
        const_specs = [c[2] for c in self.rxn_constraints if c[0]==r+1]
        ind_specs = [c for c in self.comp_names if c not in const_specs]
        return ind_specs


    def oxy_solver(self, reac_coeffs, prod_coeffs):
        '''
        Solve balance to fill in unknown stoichiometric coefficients for non-fuel species.

        '''

        # Set oxygen of unoxidized fuels to zero
        oxy_fuels = self.get_oxy_fuels()
        unoxy_fuel_inds = [i for i,c in enumerate(self.comp_names) if c in self.fuel_names and c not in oxy_fuels]
        self.balance_dict['O'][unoxy_fuel_inds] = 0

        # Create oxygen content vector to act as RHS matrix
        # Zero out the nan entries since those correspond to oxidized pseudocomponents
        A = np.nan_to_num(np.expand_dims(self.balance_dict['O'], 0),0)

        for r in self.map_rxns_oxy:
            # Get indices for reactants and products
            reac_inds = [i for i, c in enumerate(self.comp_names) if c in self.reac_names[r]]
            prod_inds = [i for i, c in enumerate(self.comp_names) if c in self.prod_names[r]]

            # Get indices for 
            reac_oxy_inds = [i for i, c in enumerate(self.comp_names) if c in self.reac_names[r] and c not in self.fuel_names]
            prod_oxy_inds = [i for i, c in enumerate(self.comp_names) if c in self.prod_names[r] and c not in self.fuel_names]

            # Negate product columns
            A_temp = A
            A_temp[:,prod_inds] *= -1

            # Enforce equality constraints
            M = self.get_constraint_matrix(r)
            AM = np.dot(A_temp, M)

            # Get independent species
            independent_specs = self.get_independent_species(r)
            independent_inds = [self.comp_names.index(c) for c in independent_specs ]

            # Set coefficients for fuel species to zero
            reac_coeff_temp = reac_coeffs[r,:]
            reac_coeff_temp[self.fuel_inds] = 0
            reac_coeff_temp = reac_coeff_temp[independent_inds]
            prod_coeff_temp = prod_coeffs[r,:]
            prod_coeff_temp[self.fuel_inds] = 0
            prod_coeff_temp = prod_coeff_temp[independent_inds]

            # Solve for coefficients of independent non-fuel species
            coeff_independent = self.solve_nnls_problem(AM,  reac_coeff_temp + prod_coeff_temp)
            
            # Map independent species into all species
            coeff_all = np.dot(M, coeff_independent)
            coeff_all = np.maximum(coeff_all, 0.01)

            # Extract indices from calculated coefficients
            reac_coeffs[r,reac_oxy_inds] = coeff_all[reac_oxy_inds]
            prod_coeffs[r,prod_oxy_inds] = coeff_all[prod_oxy_inds]

        return reac_coeffs, prod_coeffs
    

    def balance_solver(self, reac_coeffs, prod_coeffs):
        '''
        Solve material/elemental balances for pseudocomponents

        '''
        # Enforce coefficient constraints 
        for r in self.map_rxns_material:
            M = self.get_constraint_matrix(r)

            independent_specs = self.get_independent_species(r)
            independent_inds = [i for i,c in enumerate(self.comp_names) if c in independent_specs]
            coefftemp = reac_coeffs[r,independent_inds] + prod_coeffs[r,independent_inds]
            allcoeff = np.dot(M, coefftemp)

            reac_inds = [i for i, c in enumerate(self.comp_names) if c in self.reac_names[r]]
            reac_coeffs[r,reac_inds] = allcoeff[reac_inds]

            prod_inds = [i for i, c in enumerate(self.comp_names) if c in self.prod_names[r]]
            prod_coeffs[r,prod_inds] = allcoeff[prod_inds]

        # Assemble coefficient matrix
        A = np.stack([reac_coeffs[r,:] - prod_coeffs[r,:] for r in self.map_rxns_material])
        for b in self.balances:
            if np.isnan(np.sum(self.balance_dict[b])): # check if any unknowns for balance b
                self.balance_dict[b] = np.maximum(self.solve_nnls_problem(A, self.balance_dict[b]), 1e-3)

        return reac_coeffs, prod_coeffs


    def coeff_solver(self, reac_coeffs, prod_coeffs):
        '''
        Solve balance to fill in uknown stoichiometric coefficients
        '''
        # Stack vectors from material balances
        A = np.stack([self.balance_dict[b] for b in self.balances])

        for r in self.map_rxns_coeff:
            # Get indices for reactants and products
            reac_inds = [i for i, c in enumerate(self.comp_names) if c in self.reac_names[r]]
            prod_inds = [i for i, c in enumerate(self.comp_names) if c in self.prod_names[r]]

            # Negate product columns
            A_temp = A
            A_temp[:,prod_inds] *= -1

            # Enforce equality constraints
            M = self.get_constraint_matrix(r)
            AM = np.dot(A_temp, M)

            # Solve coefficients for independent species
            independent_specs = self.get_independent_species(r)
            independent_inds = [self.comp_names.index(c) for c in independent_specs]
            coeff_independent = self.solve_nnls_problem(AM,  reac_coeffs[r,independent_inds] + prod_coeffs[r,independent_inds])
            
            # Map independent species into all species
            coeff_all = np.dot(M, coeff_independent)
            coeff_all = np.maximum(coeff_all, 0.01)

            # Extract indices from calculated coefficients
            reac_coeffs[r,reac_inds] = coeff_all[reac_inds]
            prod_coeffs[r,prod_inds] = coeff_all[prod_inds]

        return reac_coeffs, prod_coeffs


    @staticmethod
    def solve_nnls_problem(A_in, x):
        '''
        Solves non-negative least-squares problem for mapping parameters

        Inputs: from equation of form
                    A_in x = 0
            where x[known_inds] are known 
        '''
        
        known_inds = [i for i in range(x.shape[0]) if not np.isnan(x[i])] 
        unknown_inds = [i for i in range(x.shape[0]) if np.isnan(x[i])]
        b = -1*np.dot(A_in[:,known_inds], x[known_inds])
        A = A_in[:,unknown_inds]
        result = nnls(A, b)
        x[unknown_inds], _ = np.squeeze(result)
        return x


    ###############################################################
    ##### RUNNING EXPERIMENTS SECTION
    ###############################################################

    def run_RTO_simulation(self, REAC_COEFFS=None, PROD_COEFFS=None, REAC_ORDERS=None, PROD_ORDERS=None,
                            ACT_ENERGY=None, PREEXP_FAC=None, HEATING_DATA=None, IC=None):

        # Assemble STARS components
        stars_components = stars.get_component_dict(self.comp_names)
        for i, name in enumerate(self.comp_names):
            stars_components[name].CMM = [self.balance_dict['M'][i]]


        # Assemble STARS reactions list
        req_comps = ['Oil', 'O2', 'H2O', 'CO', 'CO2', 'N2'] # required components
        comp_names_aug = self.comp_names + [c for c in req_comps if c not in self.comp_names]
        phases = [stars_components[c].phase for c in comp_names_aug] # get phases
        comp_names = [c for _, c in sorted(zip(phases, comp_names_aug))] # sort comp_names according to phase
        comp_names_inds = [comp_names_aug.index(c) for c in comp_names]

        stars_reactions = []
        num_pad = len(comp_names_aug) - len(self.comp_names)
        for i in range(self.num_rxns):
            storeac = np.pad(REAC_COEFFS[i,:],(0,num_pad))
            stoprod = np.pad(PROD_COEFFS[i,:],(0,num_pad))
            rorder = np.pad(REAC_ORDERS[i,:],(0,num_pad))

            stars_reactions.append(stars.Kinetics(NAME="RXN"+str(i+1),
                                    STOREAC=storeac[comp_names_inds].tolist(),
                                    STOPROD=stoprod[comp_names_inds].tolist(),
                                    RORDER=rorder[comp_names_inds].tolist(),
                                    FREQFAC=PREEXP_FAC[i], 
                                    EACT=ACT_ENERGY[i],
                                    RENTH=0.0)
                                    )

        # Create dictionary of parameters for writing the runfile
        write_dict = {
                'COMPS': stars_components, 
                'REACTS': stars_reactions, 
                'IC_dict': IC, 
                'HEATING_DATA': HEATING_DATA,
                'TFINAL': HEATING_DATA['Time'][-1],
                'O2_con_in': IC['O2']
                }

        # Run stars runfile
        self.vkc_model.write_dat_file(**write_dict)
        self.vkc_model.run_dat_file(self.exe_path, self.cd_path)

        # Parse output from STARS
        self.vkc_model.parse_stars_output()
        t, ydict = self.vkc_model.get_reaction_dict()
        
        # Update logging information
        self.num_stars_calls += 1

        # Save simulation results
        if not self.isOptimization:
            df_out = pd.DataFrame()
            df_out['Time'] = t
            df_out['Oil'] = ydict['Oil']
            df_out['O2'] = ydict['O2']
            df_out['H2O'] = ydict['H2O']
            df_out['CO'] = ydict['CO']
            df_out['CO2'] = ydict['CO2']
            df_out['N2'] = ydict['N2']
            df_out['Temp'] = ydict['Temp']
            df_out.to_excel(os.path.join(self.results_dir, 'sim_data.xls'))

        return t, ydict


    #################################################
    ##### OTHER FUNCTIONS
    #################################################
    def logging_model(self, x, log_file):
        '''
        Callback function specific to the model. Optional to implement.
        '''
        print('Total number of STARS simulation calls: {calls}'.format(calls=self.num_stars_calls), file=log_file)
