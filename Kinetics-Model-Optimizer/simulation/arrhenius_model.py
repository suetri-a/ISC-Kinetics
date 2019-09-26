import autograd.numpy as np
from utils.autograd_ivp import solve_ivp
from .kinetic_cell_base import KineticCellBase


class ArrheniusModel(KineticCellBase):

    @staticmethod
    def modify_cmd_options(parser):

        # Add simulation arguments
        parser.add_argument('--solver_method', type=str, default='BDF', help='numerical solver to use with solve_ivp()')
        parser.add_argument('--num_sim_steps', type=int, default=250, help='number of time steps to interpolate for simulation')
        
        # Add reaction arguments
        parser.add_argument('--heat_reaction', type=str, default=None, help='heats of reaction')
        parser.add_argument('--pre_exp_fwd', type=str, default=None, help='forward pre-exponential factors in Arrhenius reaction model')
        parser.add_argument('--act_eng_fwd', type=str, default=None, help='foward activation energies in Arrhenius reaction model')
        parser.add_argument('--pre_exp_rev', type=str, default=None, help='reverse pre-exponential factors in Arrhenius reaction model')
        parser.add_argument('--act_eng_rev', type=str, default=None, help='reverse activation energies in Arrhenius reaction model')

        parser.set_defaults(autodiff_enable=True)
        
        return parser

    
    def __init__(self, opts):
        super().__init__(opts)

        # Load parameters from options
        self.heat_reaction = opts.heat_reaction
        self.pre_exp_factors = opts.pre_exp_fwd
        self.act_energies = opts.act_eng_fwd

        self.O2_ind = self.comp_names.index('O2')
        self.heating_rates = [h/60 for h in self.heating_rates] # convert HR to /min
        self.time_line = np.linspace(60*opts.Tspan[0], 60*opts.Tspan[1], num=opts.num_sim_steps)
        
        self.init_params(opts)

        # Parameter bounds dictionary
        self.lb = {'reaccoeff': opts.stoic_coeff_lower, 'prodcoeff': opts.stoic_coeff_lower, 
                    'reacorder': opts.rxn_order_lower, 'prodorder': opts.rxn_order_lower,
                    'preexpfwd': opts.pre_exp_lower, 'preexprev': opts.pre_exp_lower,
                    'actengfwd': opts.act_eng_lower, 'actengrev': opts.act_eng_lower, 
                    'balance-M': 0, 'balance-O': 0, 'balance-C': 0}
        self.ub = {'reaccoeff': opts.stoic_coeff_upper, 'prodcoeff': opts.stoic_coeff_upper, 
                    'reacorder': opts.rxn_order_upper, 'prodorder': opts.rxn_order_upper,
                    'preexpfwd': opts.pre_exp_upper, 'preexprev': opts.pre_exp_upper,
                    'actengfwd': opts.act_eng_upper, 'actengrev': opts.act_eng_upper, 
                    'balance-M': np.inf, 'balance-O': np.inf, 'balance-C': np.inf}


    def init_params(self, opts):

        numel = self.num_comp*self.num_rxns
        
        self.params = np.zeros(4*self.num_comp*self.num_rxns + 4*self.num_rxns + self.num_comp*len(self.opts.balances))
        self.param_types = []

        reac_coeff, prod_coeff, reac_order, prod_order = self.init_reaction_matrices()


        # Reaction matrices and reaction orders (4*num_comp*num_rxns parameters)
        rxn_num_list = np.reshape(np.expand_dims(np.arange(1,self.num_rxns+1),1)*np.ones(self.num_comp), (-1)).tolist()
        comp_num_list = np.reshape(np.ones(self.num_rxns,1)*np.arange(0,self.num_comp), (-1)).tolist()

        self.params[:numel] = np.reshape(reac_coeff, (-1))
        self.param_types += [['reaccoeff', rxn_num_list[i], comp_num_list[i]] for i in range(len(rxn_num_list))]
        self.params[numel:2*numel] = np.reshape(prod_coeff,(-1))
        self.param_types += [['prodcoeff', rxn_num_list[i], comp_num_list[i]] for i in range(len(rxn_num_list))]
        self.params[2*numel:3*numel] = np.reshape(reac_order,(-1))
        self.param_types += [['reacorder', rxn_num_list[i], comp_num_list[i]] for i in range(len(rxn_num_list))]
        self.params[3*numel:4*numel] = np.reshape(prod_order,(-1))
        self.param_types += [['prodorder', rxn_num_list[i], comp_num_list[i]] for i in range(len(rxn_num_list))]
        

        # Activation energies and pre-exponential factors (4*num_rxns parameters)
        self.params[4*numel:4*numel + self.num_rxns] = np.array(opts.act_eng_fwd)
        self.param_types += [['actengfwd', i+1] for i in range(self.num_rxns)]
        self.params[4*numel+self.num_rxns:4*numel+2*self.num_rxns] = np.array(opts.pre_exp_fwd)
        self.param_types += [['preexpfwd', i+1] for i in range(self.num_rxns)]
        
        if opts.act_eng_rev is None:
            self.params[4*numel+2*self.num_rxns:4*numel+3*self.num_rxns] = np.zeros_like(opts.act_eng_fwd)
        else:
            self.params[4*numel+2*self.num_rxns:4*numel+3*self.num_rxns] = np.array(opts.act_eng_rev)
        self.param_types += [['actengrev', i+1] for i in range(self.num_rxns)]

        if opts.pre_exp_rev is None:
            self.params[4*numel+3*self.num_rxns:4*numel+4*self.num_rxns] = np.zeros_like(opts.pre_exp_fwd)
        else:
            self.params[4*numel+3*self.num_rxns:4*numel+4*self.num_rxns] = np.array(opts.pre_exp_rev)
        self.param_types += [['preexprev', i+1] for i in range(self.num_rxns)]


        # Appeand material vectors for balance equations (num_comp*#balances parameters)
        rxn_param_count = 4*numel+4*self.num_rxns
        for i, b in enumerate(self.balances):
            self.params[rxn_param_count+i*self.num_comp:rxn_param_count+(i+1)*self.num_comp] += self.material_dict[b]
            self.param_types += [['balance-'+b, c] for c in self.comp_names]
        

    def get_gov_eqn(self, heating_rate, IC):
        
        def ArrheniusGovEqn(t, y, params):
            '''

            Inputs:
                t - time of current step
                y - current state vector (mole concentrations and temperature)
                kc - KineticCell object containing the information for the simulation

            Outputs:
                dydt - derivative of state vector at time t

            '''
            
            # Unpack parameters vector
            numel = self.num_comp*self.num_rxns
            reac_coeff = np.reshape(params[:numel],(self.num_rxns, self.num_comp))
            prod_coeff = np.reshape(params[numel:2*numel],(self.num_rxns, self.num_comp))
            reac_order = np.reshape(params[2*numel:3*numel],(self.num_rxns, self.num_comp))
            prod_order = np.reshape(params[3*numel:4*numel],(self.num_rxns, self.num_comp))
            act_eng_fwd = params[4*numel:4*numel + self.num_rxns]
            pre_exp_fwd = params[4*numel+self.num_rxns:4*numel+2*self.num_rxns]
            # act_eng_rev = params[4*numel+2*self.num_rxns:4*numel+3*self.num_rxns]
            # pre_exp_rev = params[4*numel+3*self.num_rxns:4*numel+4*self.num_rxns]

            dydt = np.zeros(y.shape) # Initialize derivative vector
            y = y*np.equal(np.imag(y), 0)*np.greater_equal(y, 0) # Zero out elements that are complex or negative

            # FLOW TERMS
            F_O2 = self.O2_con_sim*self.q # inflow rate for oxygen
            gas_ind_vec = (1 - np.equal(self.comp_phase, 2))*(1 - np.equal(self.comp_phase, 4))
            F_out = y[:-1]*self.q*gas_ind_vec / self.V
            
            # REACTION TERMS
            # Reaction Rates
            kf = pre_exp_fwd*np.exp(-act_eng_fwd / (self.R*y[-1])) # Forward rate
            kr = np.zeros(pre_exp_fwd.shape) # Reverse rate (ignore for now)

            # Product Rates
            mole_frac_reac = np.prod(np.power(y[:-1], reac_order), axis=1)
            mole_frac_prod = np.prod(np.power(y[:-1], prod_order), axis=1)

            nu = prod_coeff - reac_coeff 
            production_rates = np.sum(np.expand_dims(kf*mole_frac_reac - kr*mole_frac_prod,1)*nu, 0)
            
            # Assemble derivative
            dydt[:-1] += production_rates - F_out
            dydt[self.O2_ind] += F_O2 / self.V

            # TEMPERATURE
            if y[-1] > self.opts.max_temp + 273.1:
                dydt[-1] = 0
            else:
                dydt[-1] = heating_rate
            
            return dydt

        return ArrheniusGovEqn


    def run_RTO_experiment(self, x, heating_rate, IC):
        '''
        Run simulated RTO experiment for given heating rate and time span.
        '''
        
        gov_eqn = self.get_gov_eqn(heating_rate, IC)
        sol = solve_ivp(gov_eqn, IC, self.time_line, (x,), method = self.opts.solver_method)
        return sol.T


    def compute_residuals(self, x):
        '''
        Compute residuals from balance equations
        
        '''
        
        res = []

        # Balances constraints
        numel = self.num_comp*self.num_rxns
        reac_coeff = np.reshape(x[:numel],(self.num_rxns, self.num_comp))
        prod_coeff = np.reshape(x[numel:2*numel],(self.num_rxns, self.num_comp))
        A_list = []
        rxn_param_count = 4*numel+4*self.num_rxns
        for i in len(self.balances):
            A_list.append(x[rxn_param_count+i*self.num_comp:rxn_param_count+(i+1)*self.num_comp])
        A = np.stack(A_list)

        for i in range(self.num_rxns):
            res.append(np.sum((A.dot(reac_coeff[i,:].T - prod_coeff[i,:].T))**2))


        # Non-pseudocomponent material values (fixed)
        for i, c in enumerate(self.comp_names):
            if c not in self.pseudo_fuel_comps:
                for b in self.balances:
                    lookup_list = ['balance-'+b, c]
                    res.append((self.material_dict[b][i] - x[self.param_types.index(lookup_list)])**2)


        # Stoichiometric constraints for species not in reactions
        for i in range(self.num_rxns):
            for c in self.comp_names:
                if c not in self.reac_names[i]:
                    lookup_list = ['reaccoeff', i, self.comp_names.index(c)]
                    res.append(x[self.param_types.index(lookup_list)]**2)
                
                if c not in self.prod_names[i]:
                    lookup_list = ['prodcoeff', i, self.comp_names.index(c)]
                    res.append(x[self.param_types.index(lookup_list)]**2)


        # Reaction constraints
        for c in self.rxn_constraints:
            if c[1] in self.reac_names[c[0]]:
                lookup_list1 = ['reaccoeff',c[0],self.comp_names.index(c[1])]
            else:
                lookup_list1 = ['prodcoeff',c[0],self.comp_names.index(c[1])]
            
            if c[2] in self.reac_names[c[0]]:
                lookup_list2 = ['reaccoeff',c[0],self.comp_names.index(c[2])]
            else:
                lookup_list2 = ['prodcoeff',c[0],self.comp_names.index(c[2])]

            res.append((x[self.param_types.index(lookup_list1)] - c[3]*x[self.param_types.index(lookup_list2)])**2)

        
        # Parameter bounds (inequality constraints)
        def g(x): return np.maximum(0, x)**2
        for i, p in enumerate(self.param_types):
            res.append(g(self.lb[p[0]] - x[i])) # Lower bound
            res.append(g(x[i] - self.ub[p[0]])) # Lower bound

        res = np.asarray(res)

        return res