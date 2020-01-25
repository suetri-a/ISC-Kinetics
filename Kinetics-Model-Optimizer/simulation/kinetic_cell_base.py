import autograd.numpy as np
from abc import ABC, abstractmethod
import argparse
import os

import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import resample

from utils.utils import mkdirs

class KineticCellBase(ABC):


    @staticmethod
    def modify_cmd_options(parser):

        return parser


    def __init__(self, opts):

        self.opts = opts
        self.isOptimization = opts.isOptimization

        #### Initialize save folders
        self.results_dir = os.path.join(opts.results_dir, opts.name,'')
        mkdirs([self.results_dir])

        #### RUN OPTIONS
        self.V = opts.porosity * opts.kc_V
        self.q = opts.flow_rate # [580 cm^3/min]
        self.P = opts.O2_partial_pressure # O2 partial pressure 
        self.R = opts.R 
        self.T0 = opts.T0
        self.O2_con_sim = 0.2070*self.P /(self.R*self.T0)

        #### SIMULATION RESULTS
        self.reaction_model = opts.reaction_model
        self.O2_consumption = None
        self.y = [] # solutions
        self.Temperature = None
        self.activation_energy = None
        self.conversion = None

        #### Initialize reaction information
        self.log_params = opts.log_params
        self.reac_names, self.prod_names = opts.reac_names, opts.prod_names
        self.comp_names = list(set([spec for rxn in self.reac_names for spec in rxn]).union(
            set([spec for rxn in self.prod_names for spec in rxn])))
        self.comp_phase = [opts.comp_phase[comp] for comp in self.comp_names]
        self.num_rxns, self.num_comp = len(self.reac_names), len(self.comp_names)
        self.rxn_constraints, self.init_coeff = opts.rxn_constraints, opts.init_coeff

        self.material_dict = opts.material_dict
        self.balances = opts.balances
        self.balance_dict = {}
        for b in self.balances:
            self.balance_dict[b] = np.array([self.material_dict[b][c] for c in self.comp_names])

        self.fuel_names = opts.fuel_comps
        self.pseudo_fuel_comps = [f for f in opts.pseudo_fuel_comps if f in self.comp_names]
        self.fuel_inds = [self.comp_names.index[r] for r in opts.reac_names if r in opts.fuel_comps or r in opts.pseudo_fuel_comps]

        # Optimization-related information
        if opts.isOptimization:
            self.IC = None
            self.heating_rates = None
            self.num_heats = None
            self.time_line = None
            self.O2_con = None

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
            
        else:
            self.max_temp = opts.max_temp
            self.heating_rates = [h/60 for h in opts.heating_rates] # convert HR to /min
            self.num_heats = len(self.heating_rates)
            self.Tspan = opts.Tspan

            self.IC = [] 
            for _ in range(self.num_heats):
                IC_temp = []
                for spec in self.comp_names:
                    if spec == 'O2' and opts.IC_dict[spec] == None:
                        IC_temp.append(self.O2_con_sim)
                    else:
                        IC_temp.append(opts.IC_dict[spec])
                IC_temp.append(opts.T0)
                self.IC.append(np.array(IC_temp))
    

    ###########################################################
    #### BEGIN REQUIRED METHODS FOR EACH MODEL
    ###########################################################
    
    @abstractmethod
    def init_params(self, opts):
        '''
        Initialize parameters vector for the kinetic cell optimization

            param_types - type of the parameters with corresponding information (list of lists) 
            Key:
            'preexp' - Pre-exponential factors (frequency factors). Stored as log or the value depending on log_params. 
                ['preexp', rxn#]
            'acteng' - Activation energy. Stored as log or the value depending on log_params.
                ['acteng', rxn#]
            'stoic' - Stoichiometric coefficients. Format:
                ['stoic', rxn#, species]
            'order' - Reaction order. Format:
                ['order', species]
            'oilsat' - Initial oil saturation. Format:
                ['oilsat']
            Note: the rxn# in the parameter vector is 0-indexed for convenience.

        All other factors are either fixed a priori or calculated based on the 
            free parameters for the system.

        '''

        pass


    @abstractmethod
    def run_RTO_experiment(self, x, heating_rate, IC):
        '''
        Wrapper to execute simulation of a single RTO experiment with entered heating rates. 

        Inputs:
            x - parameters vector (the simulator in each type of kinetic cell must
                map from the parameters vector to parameters usable for simulation)
            heating_rate - heating rate for RTO experiment
            IC - initial condition needed to feed into the simulator
        
        Returns:
            t - time vector
            y - [#Components, #Time steps] array as result from RTO experiment simulation

        '''

        pass

    
    @abstractmethod
    def clone_from_data(self, data_cell):
        pass


    @abstractmethod
    def compute_residuals(self, x):
        pass


    ###########################################################
    #### END REQUIRED METHODS
    ###########################################################
    
    def run_RTO_experiments(self, params, return_vals = False):
        '''
        Short wrapper to run the heating rates
        '''
        self.y = []
        for i, h in enumerate(self.heating_rates):
            self.y.append(self.run_RTO_experiment(params, h, self.IC[i]))

        if return_vals:
            return self.y


    def get_O2_consumption(self, x):
        '''
        Method to return O2 consumption when handed parameters x

        Inputs: 
            x - parameters vector

        Returns:
            O2_consumption = [#Heating rates, #Time steps] array of O2 consumption curves

        '''

        if self.isOptimization:
            y_temp = self.run_RTO_experiments(x, return_vals=True)
            self.O2_consumption = np.stack([self.O2_con[hr] - y_temp[i][self.O2_ind,:] for i, hr in enumerate(self.heating_rates)])
        
        else:
            self.run_RTO_experiments(x)
            self.O2_consumption = self.O2_con_sim - np.stack([y[self.O2_ind,:] for y in self.y])
            self.Temperature = np.stack([y[-1,:] for y in self.y])
        
        return self.O2_consumption


    def init_reaction_matrices(self):
        '''
        Assign stoichiometric and reaction order matrices based on input data

        '''

        size_tup = (self.num_rxns, self.num_comp)
        reac_coeff, prod_coeff = np.zeros(size_tup), np.zeros(size_tup) 
        reac_order, prod_order = np.zeros(size_tup), np.zeros(size_tup) 

        for coeff in self.opts.init_coeff:
            if coeff[1] in self.reac_names[coeff[0]-1]:
                reac_coeff[coeff[0]-1, self.comp_names.index(coeff[1])] = coeff[2]
            elif coeff[1] in self.prod_names[coeff[0]-1]:
                prod_coeff[coeff[0]-1, self.comp_names.index(coeff[1])] = coeff[2]
            else:
                raise Exception('Species {} not found in reaction {} when initializing \
                    stoichiometric coefficients'.format(coeff[1], coeff[0]))

        # Set default reactant coefficients
        for i, rxn in enumerate(self.reac_names):
            for spec in rxn:
                if reac_coeff[i, self.comp_names.index(spec)] == 0:
                    reac_coeff[i, self.comp_names.index(spec)] = 1.0 # Include default coefficient of 1
                reac_order[i, self.comp_names.index(spec)] = 1.0 # Add reaction order of 1 

        # Set default product coefficients
        for i, rxn in enumerate(self.prod_names):
            for spec in rxn:
                if prod_coeff[i, self.comp_names.index(spec)] == 0:
                    prod_coeff[i, self.comp_names.index(spec)] = 1.0 # Include default coefficient of 1
                prod_order[i, self.comp_names.index(spec)] = 0.0 # Include later if we want    

        return reac_coeff, prod_coeff, reac_order, prod_order


    def plot_O2_data(self):
        '''
        Plot O2 data from the simulated or stored RTO experiments. 

        '''

        O2_consumption = self.get_O2_consumption(self.params)

        O2_fig, O2_plot = plt.subplots()
        for i in range(len(self.heating_rates)):
            O2_plot.plot(self.time_line/60, O2_consumption[i,:])
        O2_plot.set_xlabel('Time, min')
        O2_plot.set_ylabel(r'$O_2$ Consumption [mol]')
        O2_plot.legend(['{} C/min'.format(np.around(r*60, decimals = 2)) for r in self.heating_rates]) 

        temp_fig, temp_plot = plt.subplots()
        for i in range(len(self.heating_rates)):
            temp_plot.plot(self.time_line/60, self.Temperature[i,:]-273.15)
        temp_plot.set_xlabel('Time, min') 
        temp_plot.set_ylabel('Temperature, C')
        temp_plot.legend(['{} C/min'.format(np.around(r*60, decimals = 2)) for r in self.heating_rates]) 

        return O2_fig, temp_fig


    def save_plots(self, O2_filename = 'O2_data.png', temp_filename = 'temperature_data.png'):
        O2_fig, temp_fig = self.plot_O2_data()
        O2_fig.savefig(self.results_dir + O2_filename)
        temp_fig.savefig(self.results_dir + temp_filename)


    def show_plots(self):
        O2_fig, temp_fig = self.plot_O2_data()
        O2_fig.show()
        temp_fig.show()


    def calculate_act_energy(self, T_int = 1000, num_interp = 100, method='Friedman', est_range = [-1e2, 1e6]):
        '''
        Isoconvertional method activation energy solver. Code adapted from Chen (2012).

        Inputs:
            T_int - number of time intervals for the time_line variable
            num_interp - number of points to interpolate the activation energy
            method - which type of activation energy computation to used
            est_range - estimated range for the activation energies

        '''

        if len(self.y) == 0:
            raise Exception('Must execute run_solver() or load data before activation energy can be calculated.')

        x = np.linspace(0.001, 0.999, num_interp)

        O2_total = np.zeros(self.num_heats)
        Temp_x = np.zeros((num_interp, self.num_heats))
        x_Time = np.zeros((num_interp, self.num_heats))

        for i in range(self.num_heats):
            index = np.where(self.O2_consumption[i,:]>1e-3)[0]
            if index.shape[0] < 2:
                index_start = 0
                index_end = self.O2_consumption[i,:].shape[0]
            else:
                index_start = index[0]
                index_end = index[-1]
            
            Temp_temp = self.temp_line[index_start:index_end,i]
            
            Time_temp = self.time_line[index_start:index_end]
            Time_span_temp = np.append(Time_temp[1:], [0]) - Time_temp
            Time_span = np.append(Time_span_temp[:-1], Time_span_temp[-2])
            
            isoconvertionO2 = self.O2_consumption[i,index_start:index_end]*Time_span
            O2_total[i] = np.sum(isoconvertionO2)
            
            xx = np.cumsum(isoconvertionO2) / O2_total[i]
            
            Temp_x[:,i] = np.interp(x, xx, Temp_temp)
            x_Time[:,i] = np.interp(x, xx, Time_temp)


        def phiEnergy(Ex, Temp_x, x_Time):
            '''
            Objective function for recovering activation energy

            '''

            J_E_Tt = np.trapz(np.exp(-Ex/(self.opts.R*Temp_x)), x = x_Time, axis = 0)
            phi_Ex = np.sum(np.outer(J_E_Tt, 1 / J_E_Tt)) - self.num_heats

            return phi_Ex

        # Friedman
        if method == 'Friedman':
            dxdt = np.zeros((num_interp, self.num_heats))
            for i in range(self.num_heats):
                f = interp1d(self.time_line, self.consumption_O2[:,i], kind = 'cubic')
                dxdt[:,i] = f(x_Time[:, i]) / O2_total[i]
            ln_dxdt = np.log(dxdt)

            self.activation_energy = np.zeros(num_interp)

            for j in range(num_interp):
                LineSlope = np.polyfit(-1/(self.opts.R*Temp_x[j, :]), ln_dxdt[j ,:], 1)
                self.activation_energy[j] = LineSlope[0]
            
            self.conversion = x


        # Vyazovkin
        elif method == 'Vyazovkin 1997':
            delta_t = num_interp
            Cal_Ex = np.zeros(num_interp-1)
            for index_x in range(1,num_interp):
                if index_x <= delta_t:
                    def f1(Ex): return phiEnergy(Ex, Temp_x[:index_x,:], xTime[:index_x,:])
                    Cal_Ex[index_x-1] = fminbound(f1, est_range[0], est_range[1])
                else:
                    def f2(Ex): return phiEnergy(Ex,Temp_x[index_x-delta_t:index_x, :], x_Time[index_x-delta_t:index_x,:])
                    Cal_Ex[index_x-1] = fminbound(f2, est_range[0], est_range[1])

            self.activation_energy = Cal_Ex
            self.conversion = x[1:]


        elif method == 'Vyazovkin 2001':
            delta_t = 2
            Cal_Ex = np.zeros(num_interp-1)
            for index_x in range(1,num_interp):
                if index_x < delta_t:
                    def f1(Ex): return phiEnergy(Ex, Temp_x[:index_x+1,:], x_Time[:index_x+1,:])
                    Cal_Ex[index_x-1] = fminbound(f1, est_range[0], est_range[1])
                else:
                    def f2(Ex): return phiEnergy(Ex, Temp_x[index_x-delta_t:index_x, :], x_Time[index_x-delta_t:index_x, :])
                    Cal_Ex[index_x-1] = fminbound(f2, est_range[0], est_range[1])

            self.activation_energy = Cal_Ex
            self.conversion = x[1:]

        else:
            raise Exception('Invalid activation energy option.')
            self.activation_energy = 0
            self.conversion = 0


    ##### REACTION OPTIMIZATION FUNCTIONS #####
    def get_bounds(self): 
        '''
        This function calculates the bounds for the global optimizer.

        Returns:
            ub - upper bound for the optimization (np.ndarray)
            lb - lower bound for the optimization (np.ndarray)
        '''

        lb, ub = [], []

        for param_type in self.param_types:
            
            if param_type[0] == 'preexp': # pre-exponenial factors
                if self.log_params:
                    lb.append(np.log(1e-2))
                    ub.append(np.log(1e6))
                else:
                    lb.append(1e-2)
                    ub.append(1e6)
            
            elif param_type[0] == 'acteng': # activation energies
                if self.opts.log_params:
                    lb.append(np.log(1e3))
                    ub.append(np.log(1e6))
                else:
                    lb.append(1e-3)
                    ub.append(1e6)
            
            elif param_type[0] in ['stoic', 'order']: # stoic. coefficients, reaction orders
                lb.append(1e-2)
                ub.append(2e1)
            
            elif param_type[0] == 'oilsat': # initial oil saturation
                lb.append(0)
                ub.append(2e-1)

        return lb, ub

    
    ##### CONVENIENCE FUNCTIONS
    def print_params(self):
        '''
        Output formatted table of parameters

        '''
        print('Type       Value      (Rxn #)   (Species) ')

        for i, p in enumerate(self.param_types):
            if p[0]=='preexp':
                print('Pre-exp     '+str(np.around(self.params[i], decimals=2))+'       '+str(p[1]+1))
            if p[0]=='acteng':
                print('Act. eng.   '+str(np.around(self.params[i], decimals=2))+'       '+str(p[1]+1))
            if p[0]=='stoic':
                print('Coeff.      '+str(np.around(self.params[i], decimals=2))+'        '+str(p[1]+1)+'        '+p[2])
            if p[0]=='order':
                print('Rxn. order '+str(np.around(self.params[i], decimals=2))+'                '+str(p[1]+1))
            if p[0]=='oilsat':
                print('Oil sat.   '+str(np.around(self.params[i], decimals=2)))
        
    def print_reaction(self):
        '''
        Output formatted reaction with coefficients and species

        '''

        for i in range(self.num_rxns):
            out_str = ''
            
            # Print reactants
            for j, c in enumerate(self.comp_names):
                if c in self.reac_names[i]:
                    out_str += str(np.around(self.reac_coeff[i,j], decimals=3)) + ' '
                    out_str += c + ' + '
            
            out_str = out_str[:-3]
            out_str += ' -> '
            
            # Print products
            for j, c in enumerate(self.comp_names):
                if c in self.prod_names[i]:
                    out_str += str(np.around(self.prod_coeff[i,j], decimals=3)) + ' '
                    out_str += c + ' + '
            out_str = out_str[:-3]
            
            print(out_str)