import time
import pickle
import os
import shutil
import copy
import argparse
from random import shuffle
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from pathos.multiprocessing import Pool

# import autograd.numpy as np
import numpy as np
# from autograd import grad
import matplotlib.pyplot as plt
import scipy as sp
import networkx as nx

from networkx.algorithms.simple_paths import all_simple_paths
from scipy.optimize import minimize, differential_evolution, brute, NonlinearConstraint
from pyswarm import pso
import pyswarms as ps
from scipy.signal import find_peaks_cwt
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal

from utils.utils import mkdirs, numerical_hessian

from SALib.sample import saltelli
from SALib.analyze import sobol
import emcee

# Wrapper function for using joblib with optimizers and UQ frameworks
def joblib_map(func, iter):
    return Parallel(n_jobs=-1)(delayed(func)(x) for x in iter)


class BaseOptimizer(ABC):

    @staticmethod
    def modify_cmd_options(parser):
        
        parser.add_argument('--log_file', type=str, default='optim_write_file.txt', help='file to output optimization data')
        parser.add_argument('--optim_verbose', type=eval, default=False, help='output messages during optimization')
        parser.add_argument('--optim_write_output', type=eval, default=True, help='write results of optimization to disk')
        parser.add_argument('--num_peaks', type=int, nargs='+', action='append', default=[2, 2, 2, 2, 2, 2], 
            help='number of peaks for each RTO experiment')
        parser.add_argument('--peak_weight_vec', type=float, nargs='+', action='append', default=[75.0, 75.0, 250.0, 250.0], 
            help='weight for each part of O2 peaks cost funciton')
        parser.add_argument('--reac_tol', type=float, default=1e-4, help='tolerance for start and end of reaction')

        # loss function options
        parser.add_argument('--O2_sigma', type=float, default=1e-3, help='sigma for O2 covariance')
        parser.add_argument('--CO2_sigma', type=float, default=1e-3, help='sigma for CO2 covariance')
        parser.add_argument('--loss_sigma', type=float, default=1e-2, help='sigma for noise in signal')
        parser.add_argument('--CO2_O2_sigma', type=float, default=1e-3, help='sigma for CO2/O2 covariance')

        parser.set_defaults(log_params=True)

        return parser


    def __init__(self, kinetic_cell, data_container, opts):
        
        self.autodiff_enable = opts.autodiff_enable
        self.data_container = data_container
        self.kinetic_cell = kinetic_cell
        self.base_cost = self.get_base_cost_fun(kinetic_cell, data_container, opts)
        self.sol = None
        
        # Logging information
        self.log_file = os.path.join(self.kinetic_cell.results_dir, opts.log_file)
        self.report_file = os.path.join(self.kinetic_cell.results_dir, 'results_report.txt')
        self.figs_dir = os.path.join(self.kinetic_cell.results_dir, 'figures')
        self.load_dir = os.path.join(self.kinetic_cell.results_dir,'load_dir')
        self.load_from_saved = opts.load_from_saved

        if self.load_from_saved:
            self.function_evals = np.load(os.path.join(self.load_dir,'function_evals.npy'))
            self.loss_values = np.load(os.path.join(self.load_dir,'total_loss.npy')).tolist()

            if os.path.exists(os.path.join(self.load_dir,'warm_start_complete.pkl')):
                with open(os.path.join(self.load_dir,'warm_start_complete.pkl'),'rb') as fp:
                    self.warm_start_complete = pickle.load(fp)
            else:
                self.warm_start_complete = False

            if os.path.exists(os.path.join(self.load_dir,'optim_complete.pkl')):
                with open(os.path.join(self.load_dir,'optim_complete.pkl'),'rb') as fp:
                    self.optim_complete = pickle.load(fp)
            else:
                self.optim_complete = False
        
        else:
            self.function_evals = 0
            self.loss_values = []

            with open(self.log_file,'w') as fileID: # clear any existing log file
                pass
            with open(self.report_file,'w') as fileID: # clear any existing warm start log file
                pass
            
            if os.path.exists(self.figs_dir):
                shutil.rmtree(self.figs_dir)
            
            fig_dirs_list = [os.path.join(self.figs_dir, 'warm_start_1'), os.path.join(self.figs_dir, 'warm_start_2'), 
                            os.path.join(self.figs_dir, 'warm_start_3'), 
                            os.path.join(self.figs_dir, 'optimization'), os.path.join(self.figs_dir, 'final_results')]
            mkdirs(fig_dirs_list)

            self.warm_start_complete = False
            self.optim_complete = False
            with open(os.path.join(self.load_dir,'warm_start_complete.pkl'),'wb') as fp:
                pickle.dump(self.warm_start_complete, fp)
            with open(os.path.join(self.load_dir,'optim_complete.pkl'),'wb') as fp:
                pickle.dump(self.optim_complete, fp)
            

        self.data_container.print_curves(save_path=os.path.join(self.figs_dir, 'final_results', 'rto_curves.png'))
        self.data_container.print_isoconversional_curves(save_path=os.path.join(self.figs_dir, 'final_results', 'isoconv_curves.png'))

        self.output_loss = self.get_output_loss(opts)


    ##### USER DEFINED FUNCTIONS #####
    @abstractmethod
    def optimize_cell(self):
        '''
        Function to optimize the parameters for the kinetic cell 

        Returns:
            x - optimized parameter vector

        '''
        pass


    ##### BEGIN OTHER FUNCTIONS #####
    def get_output_loss(self, opts):

        if opts.output_loss == 'gaussian':

            def output_loss(data_dict, y_dict):
                Time = data_dict['Time']
                N = Time.shape[0]
                time_diffs = np.abs((np.expand_dims(Time, 0) - np.expand_dims(Time, 1)) / Time.max())
                
                # O2 only case
                if 'O2' in opts.output_loss_inputs and 'CO2' not in opts.output_loss_inputs:
                    x_data = data_dict['O2']
                    x_sim = np.interp(Time, y_dict['Time'], y_dict['O2'])
                    M = opts.O2_sigma*np.exp(-np.power(time_diffs,2) / 2 / opts.O2_sigma**2) + opts.loss_sigma*np.eye(N)

                # CO2 only case
                elif 'O2' not in opts.output_loss_inputs and 'CO2' in opts.output_loss_inputs:
                    x_data = data_dict['CO2']
                    x_sim = np.interp(Time, y_dict['Time'], y_dict['CO2'])
                    M = opts.O2_sigma*np.exp(-np.power(time_diffs,2) / 2 / opts.CO2_sigma**2) + opts.loss_sigma*np.eye(N)

                elif 'O2' in opts.output_loss_inputs and 'CO2' in opts.output_loss_inputs:
                    x_data = np.concatenate([data_dict['O2'], data_dict['CO2']])
                    x_sim = np.concatenate([np.interp(Time, y_dict['Time'],y_dict['O2']), np.interp(Time, y_dict['Time'],y_dict['CO2'])])

                    A = opts.O2_sigma*np.exp(-np.power(time_diffs,2)/ 2 / opts.loss_sigma**2) 
                    B = np.zeros_like(A)
                    C = opts.CO2_sigma*np.exp(-np.power(time_diffs,2)/ 2 / opts.loss_sigma**2) 
                    M = np.block([[A, B], [B, C]])
                                
                x = x_data - x_sim
                loss = 0.5*np.sum(np.power(x,2))

                return loss

        elif opts.output_loss == 'exponential':
            raise Exception('exponential covariance loss function not implemented yet')
            
        else:
            raise Exception('Invalid loss function {} entered.'.format(opts.output_loss))

        return output_loss


    def get_base_cost_fun(self, kinetic_cell, data_cell, opts):
        '''
        Generate base cost function for the optimization. This is the part of the loss function that
            deals with the data. Other parts (parameter regularization, penalty methods) is implemented
            in the specific optimizers written by the user. 
        
        Inputs:
            kinetic_cell - KineticCell() or child class object with a kinetic cell model
            data_cell - DataContainer() or child class object containing the data to optimize to
            opts - OptimizationOptions() class object containing the options for the optimizers
        
        Returns: 
            base_fun(x) - base cost functions

        '''

        def base_fun(x, save_filename=None, nofig=False):
            '''

            Note: when this function is called, the number of function evaluations is incremented, 
                the data loss value is recorded, and create the overlay plot. The status of the 
                parameters is not recorded.

            '''
            self.function_evals += 1
            
            loss = 0

            # Plot experimet
            colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']
            plt.figure()

            for i, hr in enumerate(sorted(self.data_container.heating_rates)):
                data_dict = self.data_container.heating_data[hr]
                Oil_con_init = self.compute_init_oil_sat(x, data_dict)
                heating_data = {'Time': data_dict['Time'], 'Temp': data_dict['Temp']}
                IC = {'Temp': data_dict['Temp'][0], 'O2': data_dict['O2_con_in'], 'Oil': Oil_con_init}

                # Plot experimental data
                plt.plot(data_dict['Time'], data_dict['O2'], colors[i]+'-', label=str(hr))
                            
                try:
                    y_dict = self.kinetic_cell.get_rto_data(x, heating_data, IC) 
                    loss += self.output_loss(data_dict, y_dict)
                    plt.plot(y_dict['Time'], y_dict['O2'], colors[i]+'--')
                
                except:
                    loss += 1e4
            
            plt.xlabel('Time')
            plt.ylabel(r'$O_2$ consumption')
            plt.title(r'$O_2$ consumption')
            plt.legend()
            
            # Print figure with O2 consumption data
            if not nofig:
                if save_filename is None:
                    plt.savefig(os.path.join(self.figs_dir, 'optimization', '{}.png'.format(time.time())))
                else:
                    plt.savefig(save_filename)
            plt.close()

            # Save loss value
            self.loss_values.append(loss)

            return loss

        return base_fun

    
    def warm_start(self, x0, param_types):
        '''
        Get objective function used to warm start optimization by optimizing activation energies

        '''
        print('Starting warm start procedure...')

        # Set up logging information
        with open(self.report_file, 'a+') as fileID:
            print('================================== Warm Start Logging ==================================\n\n', file=fileID)
        
        hr = min(self.data_container.heating_rates)
        data_dict = self.data_container.heating_data[hr]
        heating_data = {'Time': data_dict['Time'], 'Temp': data_dict['Temp']}
        IC = {'Temp': data_dict['Temp'][0], 'O2': data_dict['O2_con_in'], 'Oil': self.data_container.Oil_con_init}


        # Initialize return variables
        lb_out = np.zeros((len(param_types)))
        ub_out = np.zeros((len(param_types)))
        
        # Initialize list of indices
        preexp_inds = [i for i,p in enumerate(param_types) if p[0]=='preexp']
        acteng_inds = [i for i,p in enumerate(param_types) if p[0]=='acteng']
        stoic_inds = [i for i,p in enumerate(param_types) if p[0]=='stoic']
        
        
        # STEP 0: Get initial guess that is physical solution
        print('Warm start step 0: initializing balanced stoichiometry...')
        lb_out[acteng_inds] = 0.0
        lb_out[preexp_inds] = 0.0
        ub_out[acteng_inds] = 1e-1
        ub_out[preexp_inds] = 1e-1
        lb_out[stoic_inds] = 1e-3
        ub_out[stoic_inds] = 100.0

        n_stoic = np.sum([len(s) for s in self.kinetic_cell.reac_names]) + np.sum([len(s) for s in self.kinetic_cell.prod_names])
        n_pseudo = len(self.kinetic_cell.pseudo_fuel_comps)

        # map from full vector of parameters into reactant/product coeff matrices
        def map_stoic(x_in):
            nu_r = np.zeros((self.kinetic_cell.num_rxns, self.kinetic_cell.num_comp))
            nu_p = np.zeros((self.kinetic_cell.num_rxns, self.kinetic_cell.num_comp))

            # perform initial mapping
            j = 0
            for i in range(self.kinetic_cell.num_rxns):
                # map reactant coefficients
                for s in self.kinetic_cell.reac_names[i]:
                    if s in self.kinetic_cell.fuel_names+self.kinetic_cell.pseudo_fuel_comps:
                        nu_r[i, self.kinetic_cell.comp_names.index(s)] = 1.0
                    else:
                        nu_r[i, self.kinetic_cell.comp_names.index(s)] = x_in[j]
                    j+=1

                # map product coefficients
                for s in self.kinetic_cell.prod_names[i]:
                    nu_p[i, self.kinetic_cell.comp_names.index(s)] = x_in[j]
                    j+=1

            # fix constrained species
            nu_all = nu_r + nu_p
            for c in self.kinetic_cell.rxn_constraints:
                parent_coeff = nu_all[c[0]-1, self.kinetic_cell.comp_names.index(c[1])]
                if c[2] in self.kinetic_cell.reac_names[c[0]-1]:
                    nu_r[c[0]-1, self.kinetic_cell.comp_names.index(c[2])] = c[3]*parent_coeff
                else:
                    nu_p[c[0]-1, self.kinetic_cell.comp_names.index(c[2])] = c[3]*parent_coeff
            return nu_r, nu_p

        def res_fun(x_in):
            nu_r, nu_p = map_stoic(x_in)

            mass = x_in[n_stoic:n_stoic+n_pseudo]
            oxy = x_in[n_stoic+n_pseudo:]

            M = np.zeros(self.kinetic_cell.num_comp)
            Ox = np.zeros(self.kinetic_cell.num_comp)

            j = 0
            for i, c in enumerate(self.kinetic_cell.comp_names):
                if c in self.kinetic_cell.pseudo_fuel_comps:
                    M[i] = mass[j]
                    Ox[i] = oxy[j]
                    j+=1
                else:
                    M[i] = self.kinetic_cell.material_dict['M'][c]
                    Ox[i] = self.kinetic_cell.material_dict['O'][c]

            res = np.concatenate((10*np.dot(nu_r - nu_p, M), np.dot(nu_r - nu_p, Ox)))
            res_out = np.sum(res**2)
            return res_out

        # z0 = [reac coeffs, prod coeffs, m's, o's]
        lb_temp = [0.0]*n_stoic + [1e-3]*n_pseudo + [0.0]*n_pseudo
        ub_temp = [100.0]*n_stoic + [1.0]*n_pseudo + [20.0]*n_pseudo
        z0 = [1.0]*n_stoic + [5e-1]*n_pseudo + [0.0]*n_pseudo
        result = minimize(res_fun, z0, bounds=list(zip(lb_temp, ub_temp)), method='SLSQP')
        nu_r, nu_p = map_stoic(result.x)

        # Map result back into initial guess for stoichiometric parameters
        x0 = np.zeros((len(param_types)))
        for i, p in enumerate(param_types):
            if p[0] == 'stoic':
                if p[2] in self.kinetic_cell.reac_names[p[1]]:
                    x0[i] = nu_r[p[1], self.kinetic_cell.comp_names.index(p[2])]
                else:
                    x0[i] = nu_p[p[1], self.kinetic_cell.comp_names.index(p[2])]
        
        print('Stoichiometry done! Initialized reaction:')
        self.kinetic_cell.print_reaction(x0)
        # self.kinetic_cell.print_fuels(x0)
        # self.kinetic_cell.print_params(x0)

        #### STEP 1: Optimize peaks to match in time treating model as single reaction
        print('Warm start step 1: optimizing E to match peak consumption...')
        x_out = np.copy(x0) 
        warm_start_cost1 = self.get_warm_start_cost(cost_fun_type='points', points=['peak', 'peak'], stage='1', log_dir='warm_start_1')

        def fun1(x_in):
            # Setup parameter vector
            x = np.copy(x_out)
            x[preexp_inds] = np.log(5e4)
            x[acteng_inds] = x_in

            # Compute cost
            cost = warm_start_cost1(x)

            # Log optimization status
            with open(self.log_file, 'a+') as fileID:
                print('========================================== Status at Warm Start (Stage 1) - Iteration {} ========================================='.format(str(self.function_evals)), file=fileID)
                self.kinetic_cell.log_status(x, fileID)
                print('Cost: {}'.format(cost), file=fileID)
                print('======================================== End Status at Warm Start (Stage 1) - Iteration {} =======================================\n\n'.format(str(self.function_evals)), file=fileID)

            return cost

        result = minimize(fun1, np.log(1e5), method='Nelder-Mead', options={'fatol': 1e0})
        x_opt = result.x
        x_out[preexp_inds] = np.log(5e4) 
        x_out[acteng_inds] = np.copy(x_opt)*np.ones_like(acteng_inds)
        
        # Plot and record stage 1 results
        plt.figure()
        plt.plot(data_dict['Time'], 100*data_dict['O2'], 'b-', label=str(hr))
        IC['Oil'] = self.compute_init_oil_sat(x_out, data_dict)
        y_dict = self.kinetic_cell.get_rto_data(x_out, heating_data, IC) 
        plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')
        plt.legend(['Observed', 'Predicted'])
        plt.xlabel('Time')
        plt.ylabel(r'$O_2$ consumption [% mol]')
        plt.title(r'Warm Start $O_2$ Stage 1')
        plt.savefig(os.path.join(self.figs_dir, 'final_results', 'warm_start_stage1.png'))
        plt.close()

        with open(self.report_file, 'a+') as fileID:
            print('======================================= Stage 1 Results ===================================\n', file=fileID)
            self.kinetic_cell.log_status(x_out, fileID)
            print('==================================== End Stage 1 Results ==================================\n', file=fileID)

        self.kinetic_cell.print_reaction(x_out)

        # Assemble final bounds to use
        lb_out[preexp_inds] = np.log(1e-2) 
        lb_out[acteng_inds] = np.maximum(np.copy(x_out[acteng_inds]) - 4.6, np.log(2e4))
        ub_out[preexp_inds] = np.log(5e5)
        ub_out[acteng_inds] = np.minimum(np.copy(x_out[acteng_inds]) + 4.6, 1e6)
        lb_out[stoic_inds] = 1e-3
        ub_out[stoic_inds] = 100.0
        
        print('Peak match complete! Computed activation energy: {}'.format(np.exp(result.x)))

        self.kinetic_cell.clean_up()

        #### STEP 2: Optimize parameters to match peaks in effluence data
        print('Starting warm start step 2: optimize parameters to match start-peaks-end loss...')
        warm_start_cost2 = self.get_warm_start_cost(cost_fun_type='peaks', stage='2', log_dir='warm_start_2')

        def fun2(x_in):
            # Setup parameter vector
            x = x_in 

            # Compute cost
            bad_x = False
            if np.any(np.greater(np.abs(self.kinetic_cell.compute_residuals(x).flatten()), 1e-3)):
                bad_x = True
                cost = 1e6
            else:
                cost = warm_start_cost2(x) / 1e3

            # Log optimization status
            with open(self.log_file, 'a+') as fileID:
                print('========================================== Status at Warm Start (Stage 2) - Iteration {} ========================================='.format(str(self.function_evals)), file=fileID)
                self.kinetic_cell.log_status(x, fileID)
                if bad_x:
                    print('Unbalanced stoichiometry. Cost: {}'.format(cost), file=fileID)
                else:
                    print('Cost: {}'.format(cost), file=fileID)
                print('======================================== End Status at Warm Start (Stage 2) - Iteration {} =======================================\n\n'.format(str(self.function_evals)), file=fileID)

            return cost

        popsize = 3*x_out.shape[0] 
        init = np.expand_dims(x_out, 0) + 0.1*np.random.randn(popsize, x_out.shape[0])
        bnds = [(lb_out[i], ub_out[i]) for i in range(x_out.shape[0])] 
        result = differential_evolution(fun2, bnds, 
                                        init=init, 
                                        maxiter=100, 
                                        popsize=popsize, 
                                        polish=False, 
                                        workers=joblib_map, 
                                        atol=1e0,
                                        tol=1e-1,
                                        recombination=0.25)
        x_opt = result.x
        x_out = np.copy(x_opt)

        # Plot and record stage 2 results
        plt.figure()
        plt.plot(data_dict['Time'], 100*data_dict['O2'], 'b-', label=str(hr))
        y_dict = self.kinetic_cell.get_rto_data(x_out, heating_data, IC)
        IC['Oil'] = self.compute_init_oil_sat(x_out, data_dict)
        plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')
        plt.legend(['Observed', 'Predicted'])
        plt.xlabel('Time')
        plt.ylabel(r'$O_2$ consumption [% mol]')
        plt.title(r'Warm Start $O_2$ Stage 2')
        plt.savefig(os.path.join(self.figs_dir, 'final_results', 'warm_start_stage2.png'))
        plt.close()

        with open(self.report_file, 'a+') as fileID:
            print('======================================= Stage 2 Results ===================================\n', file=fileID)
            self.kinetic_cell.log_status(x_out, fileID)
            print('===================================== End Stage 2 Results =================================\n\n', file=fileID)

        print('Step 2 complete! Computed reaction:')
        self.kinetic_cell.print_reaction(x_out)

        self.kinetic_cell.clean_up()


        #### STEP 3: Optimize all parameters to fit lowest heating rate
        print('Start warm start step 3: fitting parameters to lowest heating rate...')
        warm_start_cost3 = self.get_warm_start_cost(cost_fun_type='mse', stage='3', log_dir='warm_start_3')
        
        def fun3(x):
            # Compute cost
            bad_x = False
            if np.any(np.greater(np.abs(self.kinetic_cell.compute_residuals(x).flatten()), 1e-3)):
                bad_x = True
                cost = 1e6
            else:
                cost = warm_start_cost3(x)

            # Log optimization status
            with open(self.log_file, 'a+') as fileID:
                print('========================================== Status at Warm Start (Stage 3) - Iteration {} ========================================='.format(str(self.function_evals)), file=fileID)
                self.kinetic_cell.log_status(x, fileID)
                if bad_x:
                    print('Unbalanced stoichiometry. Cost: {}'.format(cost), file=fileID)
                else:
                    print('Cost: {}'.format(cost), file=fileID)
                print('======================================== End Status at Warm Start (Stage 3) - Iteration {} =======================================\n\n'.format(str(self.function_evals)), file=fileID)

            return cost
        
        bnds_out = [(lb_out[i], ub_out[i]) for i in range(ub_out.shape[0])]
        
        popsize = 3*x_out.shape[0]
        maxiter = 100
        init = np.expand_dims(x_out, 0) + 0.01*np.random.randn(popsize, x_out.shape[0])
        
        result = differential_evolution(fun3, bnds_out, 
                                        popsize=popsize, 
                                        init=init, 
                                        polish=False, 
                                        maxiter=maxiter, 
                                        workers=joblib_map)
        x_out = np.copy(result.x)

        # Plot and record stage 3 results
        plt.figure()
        plt.plot(data_dict['Time'], 100*data_dict['O2'], 'b-', label=str(hr))
        IC['Oil'] = self.compute_init_oil_sat(x_out, data_dict)
        y_dict = self.kinetic_cell.get_rto_data(x_out, heating_data, IC) 
        plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')
        plt.legend(['Observed', 'Predicted'])
        plt.xlabel('Time')
        plt.ylabel(r'$O_2$ consumption [% mol]')
        plt.title(r'Warm Start $O_2$ Stage 3')
        plt.savefig(os.path.join(self.figs_dir, 'final_results', 'warm_start_stage3.png'))
        plt.close()

        with open(self.report_file, 'a+') as fileID:
            print('======================================= Stage 3 Results ===================================\n', file=fileID)
            self.kinetic_cell.log_status(x_out, fileID)
            print('===================================== End Stage 3 Results =================================\n\n', file=fileID)

        print('Step 3 complete! Computed reaction:')
        self.kinetic_cell.print_reaction(x_out)

        self.kinetic_cell.clean_up()


        return x_out, bnds_out

    
    def get_warm_start_cost(self, cost_fun_type=None, points=None, stage='', log_dir=None):
        '''
        Gets cost for matching points to each other

        Input:
            points = [data point, simulation point]
        '''

        if cost_fun_type is None:
            raise Exception('Invalid cost function value entered.')
        

        def cost_fun(x):
            '''

            Note: when this function is called, the number of function evaluations is incremented, 
                the data loss value is recorded, and create the overlay plot. The status of the 
                parameters is not recorded.

            '''
            self.function_evals += 1

            # Plot experiment
            plt.figure()

            hr = min(self.data_container.heating_rates)
            data_dict = self.data_container.heating_data[hr]

            Oil_con_init = self.compute_init_oil_sat(x, data_dict)

            # Plot experimental data
            plt.plot(data_dict['Time'], 100*data_dict['O2'], 'b-', label=str(hr))

            legend_list = ['Data']

            try:  
            # if True: # use this line for debugging the try-block
                heating_data = {'Time': data_dict['Time'], 'Temp': data_dict['Temp']}
                IC = {'Temp': data_dict['Temp'][0], 'O2': data_dict['O2_con_in'], 'Oil': Oil_con_init} #self.data_container.Oil_con_init}
                y_dict = self.kinetic_cell.get_rto_data(x, heating_data, IC) 
                plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')
                legend_list.append('Simulation')

                if cost_fun_type == 'points':
                    
                    # experimental data point
                    if points[0] == 'peak':
                        O2_data_point = data_dict['Time'][np.argmax(data_dict['O2'])]
                    elif points[0] == 'start':
                        O2_data_point = np.amin(data_dict['Time'][data_dict['O2']>0.05*np.amax(data_dict['O2'])])
                    elif points[0] == 'end':
                        O2_data_point = np.amax(data_dict['Time'][data_dict['O2']>0.05*np.amax(data_dict['O2'])])
                    
                    # simulation data point
                    if points[1] == 'peak':
                        O2_sim_point = y_dict['Time'][np.argmax(y_dict['O2'])]
                    elif points[1] == 'start':
                        O2_sim_point = np.amin(y_dict['Time'][y_dict['O2']>0.05*np.amax(y_dict['O2'])])
                    elif points[1] == 'end':
                        O2_sim_point = np.amax(y_dict['Time'][y_dict['O2']>0.05*np.amax(y_dict['O2'])])

                    loss = (O2_data_point - O2_sim_point)**2
                

                elif cost_fun_type =='effluence':
                    O2_consump_sim = np.trapz(y_dict['O2'],x=y_dict['Time'])
                    O2_consump_data = np.trapz(data_dict['O2'],x=data_dict['Time'])
                    
                    CO2_consump_sim = np.trapz(y_dict['CO2'],x=y_dict['Time'])
                    CO2_consump_data = np.trapz(data_dict['CO2'],x=data_dict['Time'])

                    loss = (O2_consump_sim - O2_consump_data)**2 + (CO2_consump_sim - CO2_consump_data)**2

                elif cost_fun_type == 'peaks':

                    # Endpoints position loss
                    s_e_prop = 1e-2
                    O2_sim_start = np.amin(y_dict['Time'][y_dict['O2']>s_e_prop*np.amax(y_dict['O2'])])
                    O2_sim_start_ind = np.amin(np.where(y_dict['O2']>s_e_prop*np.amax(y_dict['O2'])))
                    O2_sim_end = np.amax(y_dict['Time'][y_dict['O2']>s_e_prop*np.amax(y_dict['O2'])])
                    O2_sim_end_ind = np.amax(np.where(y_dict['O2']>s_e_prop*np.amax(y_dict['O2'])))

                    O2_data_start = np.amin(data_dict['Time'][data_dict['O2']>s_e_prop*np.amax(data_dict['O2'])])
                    O2_data_start_ind = np.amin(np.where(data_dict['O2']>s_e_prop*np.amax(data_dict['O2'])))
                    O2_data_end = np.amax(data_dict['Time'][data_dict['O2']>s_e_prop*np.amax(data_dict['O2'])])
                    O2_data_end_ind = np.amax(np.where(data_dict['O2']>s_e_prop*np.amax(data_dict['O2'])))

                    endpts_loss = ((O2_sim_start - O2_data_start))**2 + ((O2_sim_end - O2_data_end))**2

                    plt.scatter([O2_data_start, O2_data_end], [5*np.amax(data_dict['O2']), 5*np.amax(data_dict['O2'])],c='g', marker='o')
                    plt.scatter([O2_sim_start, O2_sim_end], [5*np.amax(data_dict['O2']), 5*np.amax(data_dict['O2'])],c='g', marker='X')
                    legend_list.append('Data Endpoints')
                    legend_list.append('Simulation Endpoints')


                    # Difference in peak positions/values loss
                    peak_inds_O2_data = np.sort(find_peaks_cwt(data_dict['O2'], np.arange(1,np.round(data_dict['O2'].shape[0]/6),step=1), noise_perc=2))
                    peak_inds_O2_data = np.delete(peak_inds_O2_data, np.where(peak_inds_O2_data < O2_data_start_ind)) # throw out early peaks
                    peak_inds_O2_data = np.delete(peak_inds_O2_data, np.where(peak_inds_O2_data > O2_data_end_ind)) # throw out late peaks
                    
                    
                    peak_inds_O2_sim = np.sort(find_peaks_cwt(y_dict['O2'], np.arange(1,np.round(y_dict['O2'].shape[0]/6),step=1), noise_perc=2))
                    peak_inds_O2_sim = np.delete(peak_inds_O2_sim, np.where(peak_inds_O2_sim < O2_sim_start_ind))
                    peak_inds_O2_sim = np.delete(peak_inds_O2_sim, np.where(peak_inds_O2_sim > O2_sim_end_ind))
                    
                    plt.scatter(data_dict['Time'][peak_inds_O2_data], 100*data_dict['O2'][peak_inds_O2_data],c='r', marker='o')
                    plt.scatter(y_dict['Time'][peak_inds_O2_sim], 100*y_dict['O2'][peak_inds_O2_sim],c='r', marker='X')
                    legend_list.append('Data Peaks')
                    legend_list.append('Simulation Peaks')

                    # add zeros to the end if less simulation peaks than data peaks
                    num_peaks_diff = peak_inds_O2_data.shape[0] - peak_inds_O2_sim.shape[0]
                    if num_peaks_diff > 0:
                        peak_inds_O2_sim = np.concatenate((peak_inds_O2_sim, [0]*num_peaks_diff))

                    # Iterate over peaks and collect squared error of times and heights
                    diff_loss = 0
                    for i in range(peak_inds_O2_data.shape[0]):
                        # data_idx = (np.abs(data_dict['Time'][peak_inds_O2_data] - y_dict['Time'][peak_inds_O2_sim[i]])).argmin()
                        diff_loss += (y_dict['Time'][peak_inds_O2_sim[i]] - data_dict['Time'][peak_inds_O2_data[i]])**2
                        
                    for i in range(peak_inds_O2_data.shape[0]):
                        # sim_idx = (np.abs(y_dict['Time'][peak_inds_O2_sim] - data_dict['Time'][peak_inds_O2_data[i]])).argmin()
                        diff_loss += 100*(y_dict['Time'][peak_inds_O2_sim[i]] - data_dict['Time'][peak_inds_O2_data[i]])**2
                        
                    loss = diff_loss + endpts_loss 

                elif cost_fun_type == 'mse':
                    O2_MSE = np.mean(np.power(data_dict['O2'] - np.interp(data_dict['Time'], y_dict['Time'], y_dict['O2']),2))
                    CO2_MSE = np.mean(np.power(data_dict['CO2'] - np.interp(data_dict['Time'], y_dict['Time'], y_dict['CO2']),2))
                    loss = O2_MSE + CO2_MSE

                elif cost_fun_type == 'l1':
                    O2_L1 = np.mean(np.abs(data_dict['O2'] - np.interp(data_dict['Time'], y_dict['Time'], y_dict['O2'])))
                    CO2_L1 = np.mean(np.abs(data_dict['CO2'] - np.interp(data_dict['Time'], y_dict['Time'], y_dict['CO2'])))
                    loss = O2_L1 + CO2_L1

            except:
                loss = 1e6

            plt.xlabel('Time')
            plt.ylabel(r'$O_2$ consumption [% mol]')
            plt.title(r'Warm Start $O_2$ consumption (Stage {})'.format(stage))
            plt.legend(legend_list)
            
            # Print figure with O2 consumption data
            plt.savefig(os.path.join(self.kinetic_cell.results_dir, 'figures', log_dir, '{}.png'.format(time.time())))
            plt.close()

            # Save loss value
            self.loss_values.append(loss)
            
            return loss

        return cost_fun

    
    def compute_init_oil_sat(self, x, data_dict):
        '''
        Compute initial oil saturation for a given stoichiometry. 

        Input:
            x - parameter vector defining reaction

        Returns:
            oil_sat - initial oil saturation required 

        '''
        reac_coeff, prod_coeff, _, _, _, _ = self.kinetic_cell.map_parameters(x)

        coeff = reac_coeff - prod_coeff

        fuels = self.kinetic_cell.pseudo_fuel_comps+['Oil']
        fuel_inds = [self.kinetic_cell.comp_names.index(f) for f in fuels]
        
        A = np.zeros((len(fuels), len(fuels)))
        O2_vec = np.zeros((len(fuels)))
        b = np.zeros((len(fuels)))
        
        for i, f in enumerate(fuels):
            f_rxns = [k for k, r in enumerate(self.kinetic_cell.reac_names) if f in r]
            
            coeff_temp = np.sum(coeff[f_rxns,:], axis=0)
            O2_vec[i] = coeff_temp[self.kinetic_cell.comp_names.index('O2')]
            A[:,i] = coeff_temp[fuel_inds]

            if f == 'Oil':
                b[i] = 1

        x = np.linalg.solve(A, b)
        nu = np.dot(np.squeeze(x), O2_vec)
        P = 1e5 # pressure in pascals
        R = 8.3145
        Flow = 100 * 1e-6 # m^3 / min
        O2_conversion = P/R*np.trapz(Flow*data_dict['O2']/(25.0+273.15), x=data_dict['Time'])
        Oil_conversion = O2_conversion / nu # mole
        Oil_mass = 1e3 * self.kinetic_cell.material_dict['M']['Oil'] * Oil_conversion  # in g
        Vol = 1.32*1.32*3 # kinetic cell volume in cm^3
        void_V = 0.36*Vol # void volume [Porosity x Volume]
        rho = self.kinetic_cell.material_dict['rho']['Oil'] # 0.965 # density of oil in g/cm^3

        oil_sat = Oil_mass / rho / void_V # Calculate final oil saturation
        oil_sat = np.maximum(np.minimum(oil_sat, 0.99), 1e-4)

        return oil_sat


    ########################################################################
    # UNCERTANITY ANALYSIS
    ########################################################################

    def analyze_uncertainty(self):
        '''
        Perform sensitivity analysis and uncertainty quantification on parameters
        '''

        print("Starting uncertainty analysis...")
        if not os.path.exists(os.path.join(self.kinetic_cell.results_dir, 'uncertainty_analysis')):
            os.mkdir(os.path.join(self.kinetic_cell.results_dir, 'uncertainty_analysis'))

        ###### MCMC CONFIDENCE INTERVALS
        os.environ["OMP_NUM_THREADS"] = "1"
        initial = np.expand_dims(self.sol,0) + 5e-2*np.random.randn(8*len(self.kinetic_cell.param_types), len(self.kinetic_cell.param_types))
        nwalkers, ndim = initial.shape
        nsteps = 10

        def log_prob(x):
            if np.any(np.greater(np.abs(self.kinetic_cell.compute_residuals(x).flatten()), 1e-3)):
                out = np.inf
            else:
                out = self.base_cost(x)
            print('call log prob')
            return out

        # with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        sampler.run_mcmc(initial, nsteps)


        ###### SENSITIVITY ANALYSIS
        param_names = []
        bnds = []
        for i, p in enumerate(self.kinetic_cell.param_types):
            if p[0] == 'preexp':
                param_names.append('A rxn {}'.format(str(p[1]+1)))
                bnds.append([self.sol[i]-0.1, self.sol[i]+0.1])
            elif p[0] == 'acteng':
                param_names.append('Ea rxn {}'.format(str(p[1]+1)))
                bnds.append([self.sol[i]-0.1, self.sol[i]+0.1])
            elif p[0] == 'stoic':
                param_names.append('{} rxn {}'.format(p[2], str(p[1]+1)))
                bnds.append([np.maximum(self.sol[i]-0.5,0.01), np.minimum(self.sol[i]+0.5,40.0)])

        problem = {
            'num_vars': len(self.kinetic_cell.param_types),
            'names': param_names,
            'bounds': bnds
        }

        # if os.path.exists(os.path.join(self.load_dir, 'sensitivity_responses.npy')):
        #     param_values = np.load(os.path.join(self.load_dir, 'sensitivity_inputs.npy'))
        #     Y = np.load(os.path.join(self.load_dir, 'sensitivity_responses.npy'))
        
        # else:
        #     param_values = saltelli.sample(problem, 500)
        #     Y = np.zeros([param_values.shape[0]])

        #     for i, X in enumerate(param_values):
        #         Y[i] = self.base_cost(np.array(X))

        #     np.save(os.path.join(self.load_dir, 'sensitivity_inputs.npy'), param_values)
        #     np.save(os.path.join(self.load_dir, 'sensitivity_responses.npy'), Y)

        Si = sobol.analyze(problem, Y, parallel=True, n_processors=4)

        pickle.dump(problem, os.path.join(self.kinetic_cell.results_dir, 'uncertainty_analysis', 'si_problem.pkl'))
        pickle.dump(Si, os.path.join(self.kinetic_cell.results_dir, 'uncertainty_analysis', 'si.pkl'))

        # Create tornado plot
        sort_inds = sorted(range(len(problem['names'])), key=Si['ST'].__getitem__, reverse=True)

        plt.rcdefaults()
        fig, ax = plt.subplots()

        # Example data
        variables = [problem['names'][i] for i in sort_inds]
        y_pos = np.arange(len(variables))
        sensitivity = [Si['ST'][i] for i in sort_inds]
        error = [Si['ST_conf'][i] for i in sort_inds]

        ax.barh(y_pos, sensitivity, xerr=error, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variables)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Sensitivity')
        ax.tick_params(labelsize=8)
        ax.set_title('Sensitivity plot of variables')

        fig.savefig(os.path.join(self.kinetic_cell.results_dir,'sensitivity_plot.png'))


    @staticmethod
    def sum_squared_error_loss(O2_exp, O2_sim):
        return np.sum((O2_sim - O2_exp)**2)
    
    @staticmethod
    def sum_abs_error_loss(O2_exp, O2_sim):
        return np.sum(np.abs(O2_sim - O2_exp))

    @staticmethod
    def peaks_loss(O2_exp, O2_sim):
        raise Exception('Peak position and height loss not implemented.')

    @staticmethod
    def start_end_loss(O2_exp, O2_sim):
        raise Exception('Start/end loss not implemented.')

