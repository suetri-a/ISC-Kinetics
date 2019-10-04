import numpy as np 
import scipy as sp
import cvxpy as cvx
import networkx as nx
import os

import utils.stars as stars
from utils.utils import mkdirs, solve_const_lineq
from .kinetic_cell_base import KineticCellBase

class STARSModel(KineticCellBase):

    @staticmethod
    def modify_cmd_options(parser):
        # STARS Simulation Parameters
        parser.add_argument('--stars_sim_folder', type=str, default='stars', help='folder where run files are written')
        parser.add_argument('--stars_base_model', type=str, default='Chen', help='which model to use [Chen | Cinar | Kristensen]')
        parser.add_argument('--stars_exe_path', type=str, default='\'C:\\Program Files (x86)\\CMG\\STARS\\2017.10\\Win_x64\\EXE\\st201710.exe\'', 
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
        self.cd_path = self.results_dir + opts.stars_sim_folder
        mkdirs([self.cd_path])
        self.exe_path = opts.stars_exe_path

        self.time_line = None
        self.heat_reaction = opts.heat_reaction
        self.pre_exp_fwd = opts.pre_exp_fwd
        self.act_eng_fwd = opts.act_eng_fwd
        self.pre_exp_rev = opts.pre_exp_rev
        self.act_eng_rev = opts.act_eng_rev

        self.get_mappings() # Determine reactions for mappings
        self.init_params(opts) # Use mappings to determine variables
        self.map_params(self.params) # Map to be a balanced reaction


    def init_params(self, opts):
        '''
        Initialize the update functions for each reaction depending on number of unknown
            molecular weights, reaction type, etc. and initialize the corresponding parameters
            (and their parameter types) for each reaction. 

        '''

        # Initialize the lists for update functions, parameters, and parameter types
        self.params, self.param_types = [], []
        for i in range(self.num_rxns):
            if self.log_params:
                self.params.append(np.log(self.pre_exp_fwd[i]))
                self.params.append(np.log(self.act_eng_fwd[i]))
            else:
                self.params.append(self.pre_exp_fwd[i])
                self.params.append(self.act_eng_fwd[i])
            self.param_types.append(['preexp',i])
            self.param_types.append(['acteng',i])

            params_temp, param_types_temp = self.add_stoic_coeff_params(i) # stoiciometric constants
            self.params += params_temp
            self.param_types += param_types_temp

        self.params = np.array(self.params)
        self.reac_coeff, self.prod_coeff, self.reac_order, self.prod_order = self.init_reaction_matrices()


    def get_mappings(self):
        '''
        Determine reactions to use for which stage of mappings
        '''

        # Perform maximal matching with given weighting scheme
        U, V, G = self.get_rxn_fuel_graph()
        matches = nx.bipartite.maximum_matching(G)

        # Reactions for mapping material properties or coefficients
        self.map_rxns_material = [U.index(matches[v]) for v in V] # rxns to solve for fuels
        self.map_rxns_coeff = [r for r in range(self.num_rxns) if r not in self.map_rxns_material] # coeffs

        # Reactions for determining coefficients before material mapping
        oxy_fuels = self.get_oxy_fuels() # coeffs on material rxns
        self.map_rxns_oxy = [i for i in range(self.num_rxns) if ((i!=0) and \
            (i not in self.map_rxns_material) and (self.fuel_names[i] not in oxy_fuels))] 


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
        nu = self.reac_coeff + self.prod_coeff # all stoichiometric coefficients (for convenience)
        param_count, spec_ind = 0, 1
        params_temp, param_types_temp = [], []
        while param_count < pk:
            if spec_rxn[spec_ind] not in spec_const:
                params_temp.append(nu[r, self.comp_names.index(spec_rxn[spec_ind])])
                param_types_temp.append(['stoic', r, spec_rxn[spec_ind]])
                param_count+=1
            spec_ind+=1
        return params_temp, param_types_temp


    def get_rxn_fuel_graph(self):
        '''
        Generate bipartitate graph with appropriate weights
        '''

        # Initialize bipartite graph
        G = nx.Graph() 
        U = ['Reaction {}'.format(i+1) for i in range(self.num_rxns)] # Create partition of reaction nodes
        V = [f for f in self.pseudo_fuel_comps] # Create partition of unknown fuels
        G.add_nodes_from(U, bipartite = 0)
        G.add_nodes_from(V, bipartite = 1)
        
        spec_counts = np.array([len(self.reac_names[i] + self.prod_names[i]) for i in range(self.num_rxns)])
        spec_const = np.array([len([c[2] for c in self.rxn_constraints if c[0]==r+1]) for r in range(self.num_rxns)])
        weights = spec_counts - spec_const - 1

        for i, _ in enumerate(U):
            for j, v in enumerate(V):
                
                if v not in self.get_oxy_fuels():
                    G.add_edge(U[i], V[j], weight = np.maximum(0,weights[i]-1))
                else:
                    G.add_edge(U[i], V[j], weight = np.maximum(0,weights[i]))


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

        M = (self.reac_coeff.T).dot(self.prod_coeff) # Calculate adjacency matrix
        S = np.greater(sp.linalg.expm(M), 0)[self.comp_names.index('O2'), :] # Calculate paths, select O2 row

        # If there is at least one rxn_number - 1 walk between O2 and the fuel species
        #   Note: this works because graphs in combustion equations are acyclic
        oxy_fuels = [s for i, s in enumerate(self.comp_names) if s in self.opts.fuel_comps and S[i] > 0]

        return oxy_fuels


    ###############################################################
    ##### RUNNING EXPERIMENTS SECTION
    ###############################################################

    def run_RTO_experiment(self, x, heating_rate, IC):
        
        self.params = x # Make sure parameters and mapped parameters are synchronized
        self.map_params(self.params) # Map parameters into reaction variables

        # self.sim_completed, self.parsed = False, False
        # clear_stars_files(self.folder_name)
        # self.write_dat_file(components, kinetics, IC_dict, heating_rate)
        # self.run_dat_file(None, None, None)
        # self.parse_stars_output(None)
        
        reaction_list = self.get_stars_reaction_list()

        stars_components = stars.get_component_dict(self.comp_names, self.base_model)
        component_dict = {}
        for i, name in enumerate(self.comp_names):
            comp_temp = stars_components[name]
            comp_temp.CMM = [self.material_dict['M'][i]]
            component_dict[name] = comp_temp

        filename = 'stars_runfile'
        stars.create_stars_runfile(filename, self.base_model, component_dict, reaction_list, heating_rate)
        stars.run_stars_runfile(filename, self.exe_path, self.cd_path)
        t, ydict = stars.read_stars_output(filename, self.base_model)
        spec_names_temp = self.comp_names + 'Temp'
        y = np.stack([ydict[name] for name in spec_names_temp]) # assemble output vector

        return t, y

    
    def get_stars_reaction_list(self):
        reaction_list = []
        for i in range(self.num_rxns):
            reaction_list.append(stars.Reaction(NAME="RXN"+str(i+1),STOREAC=self.reac_coeff[i,:].tolist(),
                                    STOPROD=self.prod_coeff[i,:].tolist(),
                                    RORDER=self.reac_order[i,:].tolist(),
                                    FREQFAC=self.pre_exp_fwd[i], EACT=self.act_eng_fwd[i],
                                    RENTH=self.heat_reaction[i])
                                    )
        return reaction_list


    def clone_from_data(self, data_cell):
        # Clone general information
        self.O2_con = data_cell.O2_con
        self.heating_rates = data_cell.heating_rates
        self.num_heats = len(self.heating_rates)
        self.time_line = data_cell.time_line # times are loaded as dict with hr as key
        self.max_temp = data_cell.max_temp
    

    ###############################################################
    ##### PARAMETER MAPPING FUNCTIONS #####
    ###############################################################

    def map_params(self, params, res_tol = 1e-7):
        '''
        Function to map parameters from a single vector (used by the optimizer)
            to the attributes of the KineticCell object. 

        Inputs:
            params - vector of the reaction parameters

        '''

        if any(p is None for p in params): # Check that there are no non-type parameters
            raise Exception('map_params() not implemented for None type parameters.')

        self.reset_balance_dict() 
        self.parse_params() 

        # Map parameters
        if self.map_rxns_oxy:
            self.oxy_solver()
        
        self.balance_solver()
        self.coeff_solver()


    # Initialize unknown fuels, etc to NaNs
    def reset_balance_dict(self):
        '''
        Reset psuedocomponent properties to be NaNs

        '''

        for i, name in enumerate(self.comp_names):
            if name in self.pseudo_fuel_comps:
                for b in self.balances:
                    self.balance_dict[b][i] = np.nan


    # Map parameters (besides coeff's) into reaction
    def parse_params(self, order_type = 'reac'):
        '''
        Parse through parameter vector to update internal parameters or overwrite nan 
            data in the stoichiometric coefficients. 

        '''

        pre_exp_facs_forward = np.zeros_like(self.pre_exp_fwd) # pre-exponential factor (forward reaction) [1/(mole)]
        act_energies_forward = np.zeros_like(self.act_eng_fwd) # activation energy (forward reaction) [J/mole]

        # Initialize constraints for mapping material reaction parameters
        constraints = [[]]*len(self.map_rxns_material)
        
        # Iterate over parameters to parse
        for i, p in enumerate(self.params):
            # Pre-exponential factors
            if self.param_types[i][0] == 'preexp': 
                if self.log_params:
                    pre_exp_facs_forward[self.param_types[i][1]] = np.exp(p)
                else:
                    pre_exp_facs_forward[self.param_types[i][1]] = p
            
            # Activation energies
            elif self.param_types[i][0] == 'acteng': 
                if self.log_params:
                    act_energies_forward[self.param_types[i][1]] = np.exp(p)
                else:
                    act_energies_forward[self.param_types[i][1]] = p
            
            # Stoiciometric coefficients
            elif self.param_types[i][0] == 'stoic':
                if self.param_types[i][2] in self.reac_names[self.param_types[i][1]]:
                    self.reac_coeff[self.param_types[i][1], self.comp_names.index(self.param_types[i][2])] = p

                if self.param_types[i][2] in self.prod_names[self.param_types[i][1]]:
                    self.prod_coeff[self.param_types[i][1], self.comp_names.index(self.param_types[i][2])] = p
                
            # Reaction orders
            elif self.param_types[i][0] == 'order': 
                if order_type =='reac': # if reactant reaction order
                    ind = self.comp_names.index(self.param_types[i][1])
                    self.reac_order[:,ind] = np.greater(self.reac_order[:,ind],0)*p
            
            # Initial oil saturation
            elif self.param_types[i][0] == 'oilsat': 
                pass
            
            else:
                raise Exception('Invalid parameter type entered.')
        
        # Enforce constraints for coefficients in balance reactions
        for i, r in enumerate(self.map_rxns_material):
            constraints = []
            nu_r_opt, nu_p_opt = cvx.Variable(shape=(self.num_comp)), cvx.Variable(shape=(self.num_comp))
            
            constrained_specs = [C[2] for C in self.rxn_constraints if C[0]-1 == r]
            for s in self.reac_names[r]:
                ind = self.comp_names.index(s)
                if s in self.fuel_names:
                    constraints.append(nu_r_opt[ind]==1) # fix reactant fuel coeff to 1
                elif s not in constrained_specs:
                    constraints.append(nu_r_opt[ind]==self.reac_coeff[r,ind]) # coefficient entered in parsing
            
            for s in self.prod_names[r]:
                ind = self.comp_names.index(s)
                if s not in constrained_specs:
                    constraints.append(nu_p_opt[ind]==self.prod_coeff[r,ind])

            # Add species constraints
            for c in [C for C in self.rxn_constraints if C[0]-1 == r]:
                ind2 = self.comp_names.index(c[2])
                ind1 = self.comp_names.index(c[1])
                if c[2] in self.reac_names[r]:
                    if c[1] in self.reac_names[r]:
                        constraints.append(nu_r_opt[ind2] == c[3]*nu_r_opt[ind1])
                    else:
                        constraints.append(nu_r_opt[ind2] == c[3]*nu_p_opt[ind1])
                else:
                    if c[1] in self.reac_names[r]:
                        constraints.append(nu_p_opt[ind2] == c[3]*nu_r_opt[ind1])
                    else:
                        constraints.append(nu_p_opt[ind2] == c[3]*nu_p_opt[ind1])
            
            x = nu_r_opt - nu_p_opt
            obj = cvx.Minimize(1)
            prob = cvx.Problem(obj, constraints)
            prob.solve()
            self.reac_coeff[r,:] = np.maximum(x.value, 0)
            self.prod_coeff[r,:] = np.maximum(-1*x.value, 0)

        
        self.pre_exp_fwd, self.act_eng_fwd = pre_exp_facs_forward, act_energies_forward


    def create_linsys_mat_constraints(self, v):
        '''
        Produce constraints so that non-nan entries of v are constrained to 
            their value. 
        Input:
            v - numpy array of material properties we want to optimize 
        Returns:
            v_out - cvx Variable the same size as v
            constraints - list of cvx constraints for fixed entries of v
        '''
        v_out, constraints = cvx.Variable(v.shape), []
        for i in range(v.shape[0]):
            if not np.isnan(v[i]):
                constraints.append(v_out[i] == v[i])
        constraints.append(v_out >= 0)
        return v_out, constraints


    def create_linsys_coeff_constraints(self, v_r, v_p, rxns, comps):
        '''
        Produce constraints so that non-nan entries of v are constrained to 
            their value. 
        Input:
            v_r, v_p - numpy arrays for reactants and products, shape n x len(comps)
            rxns - list of reactions associated with v
            comps - names of components
        Returns:
            v_out - cvx Variable the same size as v
            constraints - list of cvx constraints for fixed entries of v
        '''
        
        v_r_out, v_p_out, constraints = cvx.Variable((len(rxns), len(comps))), cvx.Variable((len(rxns),len(comps))), []

        for i, r in enumerate(rxns):
            # Fix species not in reactants/products to zero
            for j, c in enumerate(comps):
                if c not in self.reac_names[r]:
                    constraints.append(v_r_out[i,j]==0)
                if c not in self.prod_names[r]:
                    constraints.append(v_p_out[i,j]==0)

            # Fix parameter species at given values
            for j in [k for k, p in enumerate(self.param_types) if p[0]=='stoic' and p[1]==r]:
                if self.param_types[j][2] in comps:
                    if self.param_types[j][2] in self.reac_names[r]:
                        constraints.append(v_r_out[i,comps.index(self.param_types[j][2])]==self.params[j])
                    if self.param_types[j][2] in self.prod_names[r]:
                        constraints.append(v_p_out[i,comps.index(self.param_types[j][2])]==self.params[j])

            # Iterate over reaction constraints
            for c in [C for C in self.rxn_constraints if C[0]-1 == r]:
                if c[2] in comps:
                    if c[2] in self.reac_names[r]:
                        if c[1] in self.reac_names[r]:
                            constraints.append(v_r_out[i,comps.index(c[2])] == \
                                c[3]*v_r_out[i,comps.index(c[1])])
                        else:
                            constraints.append(v_r_out[i,comps.index(c[2])] == \
                                c[3]*v_p_out[i,comps.index(c[1])])
                    else:
                        if c[1] in self.reac_names[r]:
                            constraints.append(v_p_out[i,comps.index(c[2])] == \
                                c[3]*v_r_out[i,comps.index(c[1])])
                        else:
                            constraints.append(v_p_out[i,comps.index(c[2])] == \
                                c[3]*v_p_out[i,comps.index(c[1])])
            
            # Set reactant fuel to have coefficient of 1
            if comps == self.comp_names:
                constraints.append(v_r_out[i,self.fuel_inds[r]]==1)

        # Enforce positivity
        constraints.append(v_r_out >= 0)
        constraints.append(v_p_out >= 0)

        return v_r_out, v_p_out, constraints


    ##### Parameter update solvers
    def oxy_solver(self):
        # Create oxygen vector
        nonfuels = [c for c in self.comp_names if c not in self.fuel_names]
        nonfuel_inds = [i for i in range(len(self.comp_names)) if i not in self.fuel_inds]
        A = np.reshape(np.array([self.material_dict['O'][i] for i in nonfuel_inds]), (-1,1))

        # Iterate over reactions -- address dimensionality issue later
        for r in self.map_rxns_oxy:

            # Create optimizaiton variables and constraints
            nu_r_temp = self.reac_coeff[r, nonfuel_inds]
            nu_p_temp = self.prod_coeff[r, nonfuel_inds]
            nu_r, nu_p, constraints = self.create_linsys_coeff_constraints(nu_r_temp, nu_p_temp, [r], comps=nonfuels)
            nu = nu_r - nu_p

            # Solve non-negative least squares problem
            x, _ = solve_const_lineq(A, nu, 0, constraints)
            
            # Update reaction parameters
            self.reac_coeff[r, nonfuel_inds] = np.maximum(x, 0)
            self.prod_coeff[r, nonfuel_inds] = np.maximum(-x, 0)


    def balance_solver(self):
        '''
        Material balance to obtain missing molecular weight and/or oxygen content of a fuel species. 
        ''' 

        # Assemble A matrix of coefficients
        A = (self.reac_coeff - self.prod_coeff)[self.map_rxns_material,:]

        # Solve balances individually
        for B in self.opts.balances:
            # Create optimization variables and constraints
            v = self.material_dict[B]
            nu, constraints = self.create_linsys_mat_constraints(v)

            # Solve non-negative least squares problem and update parameters
            self.material_dict[B], _ = solve_const_lineq(A, nu, 0, constraints)


    def coeff_solver(self):
        '''
        Material/oxygen balance to obtain missing stoichiometric coefficients. Assumes all 
            weights/oxygen counts are known. Balances can be applied separately.
        '''

        # Assemble matrices of material properties for balances
        A = np.stack([self.material_dict[B] for B in self.opts.balances])

        # Iterate over reactions and solve individually
        for r in self.map_rxns_coeff:
            # Create optimization variables and constraints
            nu_r, nu_p, constraints = self.create_linsys_coeff_constraints(self.reac_coeff[r,:], self.prod_coeff[r,:], 
                                                                            [r], self.comp_names)
            nu = nu_r - nu_p

            # Solve non-negative least squares problem
            x, _ = solve_const_lineq(A, nu.T, 0, constraints)

            # Update parameters
            self.reac_coeff[r,:] = np.squeeze(np.maximum(x, 0))
            self.prod_coeff[r,:] = np.squeeze(np.maximum(-x, 0))
        

    #################################################
    ##### OTHER UTILITY FUNCTIONS
    #################################################

    def compute_residuals(self, x):
        '''
        Compute residuals from balance equations
        '''

        self.map_params(x)
        
        res = []

        # Balances matrix
        A = np.stack([self.material_dict[B] for B in self.opts.balances])

        for i in range(self.num_rxns):
            res.append(np.sum((A.dot(self.reac_coeff[i,:].T - self.prod_coeff[i,:].T))**2))

        return res