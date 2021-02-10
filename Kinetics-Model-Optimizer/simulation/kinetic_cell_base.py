import autograd.numpy as np
from abc import ABC, abstractmethod
import argparse
import os, sys, pickle

import matplotlib.pyplot as plt
from scipy.optimize import fminbound, minimize, nnls
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
        load_dir = os.path.join(self.results_dir,'load_dir','')
        mkdirs([self.results_dir, load_dir])

        #### Initialize reaction information
        self.reaction_model = opts.reaction_model
        self.log_params = opts.log_params
        self.reac_names, self.prod_names = opts.reac_names, opts.prod_names
        self.comp_names = list(set([spec for rxn in self.reac_names+self.prod_names for spec in rxn]))
        self.num_rxns, self.num_comp = len(self.reac_names), len(self.comp_names)
        self.rxn_constraints, self.init_coeff = opts.rxn_constraints, opts.init_coeff

        self.material_dict = opts.material_dict
        self.balances = opts.balances
        self.balance_dict = {}
        for b in self.balances:
            self.balance_dict[b] = np.array([self.material_dict[b][c] for c in self.comp_names])

        self.fuel_names = opts.fuel_comps
        self.pseudo_fuel_comps = [f for f in opts.pseudo_fuel_comps if f in self.comp_names]
        self.fuel_inds = [self.comp_names.index(r) for names in opts.reac_names for r in names if r in opts.fuel_comps or r in opts.pseudo_fuel_comps]

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

        if opts.load_from_saved:
            with open(os.path.join(self.results_dir,'load_dir','param_types.pkl'), 'rb') as fp:
                self.param_types = pickle.load(fp)
        
        else:
            self.initialize_parameters(opts)
            with open(os.path.join(self.results_dir,'load_dir','param_types.pkl'), 'wb') as fp:
                pickle.dump(self.param_types, fp)

    
    ###########################################################
    #### BEGIN REQUIRED METHODS FOR EACH MODEL
    ###########################################################
    
    @abstractmethod
    def initialize_parameters(self, opts):
        '''
        Initialize parameters vector for the kinetic cell optimization

            param_types - type of the parameters with corresponding information (list of lists) 
            Key:
            'preexpfwd', 'preexprev' - Pre-exponential factors (frequency factors). Stored as log or the value depending on log_params. 
                ['preexp', rxn#]
            'actengfwd', 'actengrev' - Activation energy. Stored as log or the value depending on log_params.
                ['acteng', rxn#]
            'stoic' - Stoichiometric coefficients. Format:
                ['stoic', rxn#, species]

            Note: the rxn# in the parameter vector is 0-indexed for convenience.

        These will be stored in the self.param_types list.

        All other factors are either fixed a priori or calculated based on the 
            free parameters for the system.

        '''

        pass


    @abstractmethod
    def map_parameters(self, x):
        '''
        Map parameters vector into components needed to run simulation.

        Input: 
            x - parameter vector
        
        Output:
            reac_coeffs - reactant coefficients
            prod_coeffs - product coefficients
            reac_orders - reactant orders
            prod_orders - product orders
            act_energy - activation energies
            preexp_fac - preexponential factors

        '''

        pass
    

    @abstractmethod
    def run_RTO_simulation(self, REAC_COEFFS=None, PROD_COEFFS=None, REAC_ORDERS=None, PROD_ORDERS=None,
                            ACT_ENERGY=None, PREEXP_FAC=None, HEATING_DATA=None, IC=None):
        '''
        Wrapper to execute simulation of a single RTO experiment with entered heating rates. 

        Inputs:
            ########TO FILL IN###########

        Returns:
            t - time vector
            y - dict{component name: np.array} dictionary

        '''

        pass

    
    @abstractmethod
    def logging_model(self, x, log_file):
        '''
        Callback function specific to the model. Optional to implement.
        '''
        pass

        
    def clean_up(self):
        '''
        Callback function to clean up extra files from the simulation

        '''
        pass

    ###########################################################
    #### END REQUIRED METHODS
    ###########################################################


    def get_rto_data(self, x, heating_data, IC):
        '''
        Method to return O2 consumption when handed parameters x

        Inputs: 
            x - parameters vector
            heating_data - data structure to feed into heating data term of sefl.run_RTO_simulation()
            IC - dictionary of initial conditions. Must contain at least 'Oil', 'O2', and 'Temp'. 

        Returns:
            y_dict_out = dictionary containing time, O2 consumption, CO2 production, and temperature

        '''
        # Check if parameters are different from current parameters or O2 consumption is empty
        reac_coeff, prod_coeff, reac_order, prod_order, e_act, pre_exp_fac = self.map_parameters(x)
        # print('Running RTO simulation....')
        t, y_dict = self.run_RTO_simulation(REAC_COEFFS=reac_coeff, PROD_COEFFS=prod_coeff, REAC_ORDERS=reac_order,
                                            PROD_ORDERS=prod_order, ACT_ENERGY=e_act, PREEXP_FAC=pre_exp_fac,
                                            HEATING_DATA=heating_data, IC=IC)
        # print('Finished RTO simulation!')

        y_dict_out = {'Time': t, 'O2': np.maximum(IC['O2'] - y_dict['O2'], 0), 'CO2': y_dict['CO2'], 'Temp': y_dict['Temp']}
        return y_dict_out


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


    def plot_data(self, Time, Temp, O2):
        '''
        Plot O2 data from the simulated or stored RTO experiments. 

        Inputs:
            Time - dictionary with keys [heating rate] time vectors
            Temp - dictionary with keys [heating rate] temperature vectors
            O2 - dictionary with keys [heating rate] O2 consumption vectors

        '''
        
        O2_fig, O2_plot = plt.subplots()
        for hr in Time.keys():
            O2_plot.plot(Time[hr], 100*O2[hr])

        O2_plot.set_xlabel('Time, min')
        O2_plot.set_ylabel(r'$O_2$ Consumption [% mol]')
        

        temp_fig, temp_plot = plt.subplots()
        for hr in Time.keys():
            temp_plot.plot(Time[hr], Temp[hr])
        
        temp_plot.set_xlabel('Time, min') 
        temp_plot.set_ylabel('Temperature, C')
        
        O2_plot.legend(['{} C/min'.format(np.around(r, decimals = 2)) for r in Time.keys()])
        temp_plot.legend(['{} C/min'.format(np.around(r, decimals = 2)) for r in Time.keys()])

        return O2_fig, temp_fig


    def save_plots(self, Time, Temp, O2, O2_filename = 'O2_data.png', temp_filename = 'temperature_data.png'):
        O2_fig, temp_fig = self.plot_data(Time, Temp, O2)
        O2_fig.savefig(O2_filename)
        temp_fig.savefig(temp_filename)


    def show_plots(self, Time, Temp, O2):
        O2_fig, temp_fig = self.plot_data(Time, Temp, O2)
        O2_fig.show()
        temp_fig.show()


    # Calculate activation energy from simulated data

    # def calculate_act_energy(self, T_int = 1000, num_interp = 100, method='Friedman', est_range = [-1e2, 1e6]):
    #     '''
    #     Isoconvertional method activation energy solver. Code adapted from Chen (2012).

    #     Inputs:
    #         T_int - number of time intervals for the time_line variable
    #         num_interp - number of points to interpolate the activation energy
    #         method - which type of activation energy computation to used
    #         est_range - estimated range for the activation energies

    #     '''

    #     if len(self.y) == 0:
    #         raise Exception('Must execute run_solver() or load data before activation energy can be calculated.')

    #     x = np.linspace(0.001, 0.999, num_interp)

    #     O2_total = np.zeros(self.num_heats)
    #     Temp_x = np.zeros((num_interp, self.num_heats))
    #     x_Time = np.zeros((num_interp, self.num_heats))

    #     for i in range(self.num_heats):
    #         index = np.where(self.O2_consumption[i,:]>1e-3)[0]
    #         if index.shape[0] < 2:
    #             index_start = 0
    #             index_end = self.O2_consumption[i,:].shape[0]
    #         else:
    #             index_start = index[0]
    #             index_end = index[-1]
            
    #         Temp_temp = self.temp_line[index_start:index_end,i]
            
    #         Time_temp = self.time_line[index_start:index_end]
    #         Time_span_temp = np.append(Time_temp[1:], [0]) - Time_temp
    #         Time_span = np.append(Time_span_temp[:-1], Time_span_temp[-2])
            
    #         isoconvertionO2 = self.O2_consumption[i,index_start:index_end]*Time_span
    #         O2_total[i] = np.sum(isoconvertionO2)
            
    #         xx = np.cumsum(isoconvertionO2) / O2_total[i]
            
    #         Temp_x[:,i] = np.interp(x, xx, Temp_temp)
    #         x_Time[:,i] = np.interp(x, xx, Time_temp)


    #     def phiEnergy(Ex, Temp_x, x_Time):
    #         '''
    #         Objective function for recovering activation energy

    #         '''

    #         J_E_Tt = np.trapz(np.exp(-Ex/(self.opts.R*Temp_x)), x = x_Time, axis = 0)
    #         phi_Ex = np.sum(np.outer(J_E_Tt, 1 / J_E_Tt)) - self.num_heats

    #         return phi_Ex

    #     # Friedman
    #     if method == 'Friedman':
    #         dxdt = np.zeros((num_interp, self.num_heats))
    #         for i in range(self.num_heats):
    #             f = interp1d(self.time_line, self.consumption_O2[:,i], kind = 'cubic')
    #             dxdt[:,i] = f(x_Time[:, i]) / O2_total[i]
    #         ln_dxdt = np.log(dxdt)

    #         self.activation_energy = np.zeros(num_interp)

    #         for j in range(num_interp):
    #             LineSlope = np.polyfit(-1/(self.opts.R*Temp_x[j, :]), ln_dxdt[j ,:], 1)
    #             self.activation_energy[j] = LineSlope[0]
            
    #         self.conversion = x


    #     # Vyazovkin
    #     elif method == 'Vyazovkin 1997':
    #         delta_t = num_interp
    #         Cal_Ex = np.zeros(num_interp-1)
    #         for index_x in range(1,num_interp):
    #             if index_x <= delta_t:
    #                 def f1(Ex): return phiEnergy(Ex, Temp_x[:index_x,:], xTime[:index_x,:])
    #                 Cal_Ex[index_x-1] = fminbound(f1, est_range[0], est_range[1])
    #             else:
    #                 def f2(Ex): return phiEnergy(Ex,Temp_x[index_x-delta_t:index_x, :], x_Time[index_x-delta_t:index_x,:])
    #                 Cal_Ex[index_x-1] = fminbound(f2, est_range[0], est_range[1])

    #         self.activation_energy = Cal_Ex
    #         self.conversion = x[1:]


    #     elif method == 'Vyazovkin 2001':
    #         delta_t = 2
    #         Cal_Ex = np.zeros(num_interp-1)
    #         for index_x in range(1,num_interp):
    #             if index_x < delta_t:
    #                 def f1(Ex): return phiEnergy(Ex, Temp_x[:index_x+1,:], x_Time[:index_x+1,:])
    #                 Cal_Ex[index_x-1] = fminbound(f1, est_range[0], est_range[1])
    #             else:
    #                 def f2(Ex): return phiEnergy(Ex, Temp_x[index_x-delta_t:index_x, :], x_Time[index_x-delta_t:index_x, :])
    #                 Cal_Ex[index_x-1] = fminbound(f2, est_range[0], est_range[1])

    #         self.activation_energy = Cal_Ex
    #         self.conversion = x[1:]

    #     else:
    #         raise Exception('Invalid activation energy option.')
    #         self.activation_energy = 0
    #         self.conversion = 0


    ##### OPTIMIZATION REALTED FUNCTIONS #####
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
                    ub.append(np.log(5e2))
                else:
                    lb.append(1e-2)
                    ub.append(5e2)
            
            elif param_type[0] == 'acteng': # activation energies
                if self.opts.log_params:
                    lb.append(np.log(1e3))
                    ub.append(np.log(1e6))
                else:
                    lb.append(1e3)
                    ub.append(1e6)
            
            elif param_type[0] in ['stoic', 'order']: # stoic. coefficients, reaction orders
                lb.append(1e-1)
                ub.append(2.5e1)
            
            elif param_type[0] == 'oilsat': # initial oil saturation
                lb.append(0)
                ub.append(2e-1)

        bnds = [(lb[i], ub[i]) for i in range(len(self.param_types))]

        return bnds

    
    def compute_residuals(self, x):
        reac_coeffs, prod_coeffs, _, _, _, _ = self.map_parameters(x)
        res = []

        # Balances matrix
        A = np.stack([self.balance_dict[B] for B in self.opts.balances])
        
        for i in range(self.num_rxns):
            res.append((A.dot(reac_coeffs[i,:].T - prod_coeffs[i,:].T)))
        res = np.array(res)
        
        return res


    def log_status(self, x, log_file):
        '''
        Callback function to log information specific to the kinetic cell model.
        '''

        # print('Current parameter values:', file=log_file)
        self.print_params(x, fileID=log_file)

        self.print_fuels(x, fileID=log_file)

        # print('Current reaction:', file=log_file)
        self.print_reaction(x, fileID=log_file)

        self.logging_model(x, log_file)



    def print_params(self, x, fileID=sys.stdout):
        '''
        Output formatted table of parameters

        '''
        message = '| ----- Type ----- | ----- Value ----- | -- (Rxn #) -- | ---- (Species) ---- |\n'

        for i, p in enumerate(self.param_types):
            
            if p[0]=='preexp':
                if self.log_params:
                    print_tup = ('Pre-exp factor', str(np.around(np.exp(x[i]),decimals=2)), str(p[1]+1),'')
                else:
                    print_tup = ('Pre-exp factor', str(np.around(x[i],decimals=2)), str(p[1]+1),'')
            if p[0]=='acteng':
                if self.log_params:
                    print_tup = ('Activation energy', str(np.around(np.exp(x[i]),decimals=2)), str(p[1]+1),'')
                else:
                    print_tup = ('Activation energy', str(np.around(x[i],decimals=2)), str(p[1]+1),'')
            if p[0]=='stoic':
                print_tup = ('Coefficient', str(np.around(x[i],decimals=2)), str(p[1]+1),str(p[2]))

            message += '{:<20}{:<20}{:<16}{:25}\n'.format(*print_tup)

        print(message, file=fileID)

    
    def print_fuels(self, x, fileID=sys.stdout):
        '''
        Print information for parent fuel and pseudocomponents

        '''
        _ = self.map_parameters(x)

        message = '| ----- Fuel ----- | -- Molecular Weight -- | --- Oxygen Content --- |\n'

        fuel_names = sorted(list(set([s for reac in self.reac_names for s in reac if s in self.fuel_names+self.pseudo_fuel_comps])))
        for f in fuel_names:
            print_tup = (f, self.balance_dict['M'][self.comp_names.index(f)], self.balance_dict['O'][self.comp_names.index(f)])
            message += '     {:<20}{:<25}{:<25}\n'.format(*print_tup)
        
        print(message, file=fileID)


    def print_reaction(self, x, fileID=sys.stdout):
        '''
        Output formatted reaction with coefficients and species

        '''

        reac_coeffs, prod_coeffs, _, _, act_energy, preexp_fac = self.map_parameters(x)

        message = '| ---------------------------- Reaction ------------------------------------------- | ----- A ---- | ----- E ---- |\n'

        for i in range(self.num_rxns):
            reac_str = ''
            
            # Print reactants
            for j, c in enumerate(self.comp_names):
                if c in self.reac_names[i]:
                    reac_str += str(np.around(reac_coeffs[i,j], decimals=3)) + ' '
                    reac_str += c + ' + '
            
            reac_str = reac_str[:-3] + ' -> '
            
            prod_str = ''
            
            # Print products
            for j, c in enumerate(self.comp_names):
                if c in self.prod_names[i]:
                    prod_str += str(np.around(prod_coeffs[i,j], decimals=3)) + ' '
                    prod_str += c + ' + '
            prod_str = prod_str[:-3]

            message += '{:>30}{:<50}{:>15}{:>15}\n'.format(reac_str, prod_str, str(np.around(preexp_fac[i],decimals=3)), str(np.around(act_energy[i],decimals=3)))
            
        print(message, file=fileID)