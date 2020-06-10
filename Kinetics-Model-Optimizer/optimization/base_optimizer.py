import time
import pickle
import os
import shutil
import copy
import argparse
from random import shuffle
from abc import ABC, abstractmethod

# import autograd.numpy as np
import numpy as np
# from autograd import grad
import matplotlib.pyplot as plt
import scipy as sp
import networkx as nx

from networkx.algorithms.simple_paths import all_simple_paths
from scipy.optimize import minimize, differential_evolution, brute, NonlinearConstraint
from pyswarm import pso
from scipy.signal import find_peaks_cwt
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal

from utils.utils import mkdirs, numerical_hessian

from SALib.sample import saltelli
from SALib.analyze import sobol


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
                            os.path.join(self.figs_dir, 'warm_start_3a'), os.path.join(self.figs_dir, 'warm_start_3b'),
                            os.path.join(self.figs_dir, 'warm_start_3c'), os.path.join(self.figs_dir, 'warm_start_3d'),
                            os.path.join(self.figs_dir, 'warm_start_4'), os.path.join(self.figs_dir, 'warm_start_5'), 
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

        # if opts.output_loss == 'mse':
        #     output_loss = self.sum_squared_error_loss
        
        # elif opts.output_loss == 'mean_abs_error':
        #     output_loss = self.sum_squared_error_loss


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
                # L = cholesky(M, lower=True)

                # # Referenced https://stats.stackexchange.com/questions/186307/
                # #                  efficient-stable-inverse-of-a-patterned-covariance-matrix-for-gridded-data
                # alpha = np.linalg.solve(L.T, np.linalg.solve(L, x)) # np.linalg.solve(M, x)
                # loss = 0.5*np.dot(alpha, x) + N*np.log(2*np.pi) + np.sum(np.log(np.diagonal(L))) # np.linalg.det(M)

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

        if opts.output_loss not in ['gaussian', 'exponential']:
            raise Exception('Autodiff optimizer not supported for selected output loss.')
            
        if opts.param_loss not in ['uniform']:
            raise Exception('Autodiff optimizer not support for selected parameter loss.')
        

        def base_fun(x, save_filename=None):
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
            if save_filename is None:
                plt.savefig(os.path.join(self.figs_dir, 'optimization', 'O2_overlay_{}.png'.format(self.function_evals)))
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
        x_out = np.copy(x0)
        # x_out = np.array([np.log(13.13), np.log(28162.12), 2.5, 2.31, np.log(4.36), np.log(40893.75), 6.35, 2.64,  np.log(6.08), np.log(41223.16), 1.35, 4.01, 4.43])

        #### STEP 1: Optimize peaks to match in time treating model as single reaction
        bnds_all = self.data_container.compute_bounds(self.kinetic_cell.param_types)
        
        preexp_inds = [i for i,p in enumerate(param_types) if p[0]=='preexp']
        acteng_inds = [i for i,p in enumerate(param_types) if p[0]=='acteng']
        stoic_inds = [i for i, p in enumerate(param_types) if p[0]=='stoic']
        
        bnd_preexp = (np.amin([bnds_all[i][0] for i in preexp_inds]), np.amax([bnds_all[i][1] for i in preexp_inds]))
        bnd_acteng = (np.amin([bnds_all[i][0] for i in acteng_inds]), np.amax([bnds_all[i][1] for i in acteng_inds]))

        # ##### TESTING ONLY
        # x_out[preexp_inds] = np.log(13.34)
        # x_out[acteng_inds] = np.log(32482.33)

        
        warm_start_cost1 = self.get_warm_start_cost(cost_fun_type='points', points=['peak', 'peak'], stage='1', log_dir='warm_start_1')

        def fun1(x_in):
            # Setup parameter vector
            x = np.copy(x_out)
            x[preexp_inds] = np.log(5e4)# x_in[0]
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

        # lb = [np.log(5e4), bnd_acteng[0]-0.7]
        # ub = [np.log(5e4), bnd_acteng[1]+2.3]
        lb = [bnd_acteng[0]-0.7]
        ub = [bnd_acteng[1]+2.3]
        
        x_opt, _ = pso(fun1, lb, ub, swarmsize=10, maxiter=25, phip=0.25, phig=0.85, minfunc=1e0, minstep=1e0)
        x_out[preexp_inds] = np.log(5e4) # np.copy(x_opt[0])*np.ones_like(preexp_inds)
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
        
        ''' 
        #### STEP 2: Optimize coefficients to match conversion
        warm_start_cost2 = self.get_warm_start_cost(cost_fun_type='effluence', stage='2', log_dir='warm_start_2')

        # def constraint_fun(x_in):
        #     x = np.copy(x_out)
        #     x[stoic_inds] = x_in
        #     out1 = (np.array(self.kinetic_cell.compute_residuals(x))).flatten() + 1e-4
        #     out2 = 1e-4 - np.copy(out1)
        #     out = np.concatenate((out1, out2))
        #     return out

        lb_proj, ub_proj = np.zeros_like(lb_out), np.zeros_like(ub_out)
        lb_proj[stoic_inds] = 1e-2
        ub_proj[stoic_inds] = 50.0
        lb_proj[preexp_inds] = x_out[preexp_inds[0]]
        ub_proj[preexp_inds] = x_out[preexp_inds[0]]
        lb_proj[acteng_inds] = x_out[acteng_inds[0]]
        ub_proj[acteng_inds] = x_out[acteng_inds[0]]

        res_fun = lambda z: np.sum(np.abs((np.array(self.kinetic_cell.compute_residuals(z))).flatten()))
        def fun2(x_in):
            # Setup parameter vector
            x = np.copy(x_out)
            x[stoic_inds] = x_in
            # proj_result = minimize(res_fun, x, bounds=[(lb_proj[i], ub_proj[i]) for i in range(lb_out.shape[0])])
            x_proj = x # proj_result.x
            res_cost = res_fun(x_proj) # np.sum(np.abs((np.array(self.kinetic_cell.compute_residuals(x))).flatten()))

            # Compute cost
            base_cost = warm_start_cost2(x_proj)
            cost = base_cost # + 1e4*res_cost
            
            # Log optimization status
            with open(self.log_file, 'a+') as fileID:
                print('========================================== Status at Warm Start (Stage 2) - Iteration {} ========================================='.format(str(self.function_evals)), file=fileID)
                self.kinetic_cell.log_status(x_proj, fileID)
                print('Cost: {} (effluence: {}, balance: {})'.format(cost, base_cost, res_cost), file=fileID)
                print('========================================= End Status at Warm Start (Stage 2) - Iteration {} ======================================\n\n'.format(str(self.function_evals)), file=fileID)
            
            return cost

        lb, ub = [np.maximum(x_out[i]-10.0,1e-2) for i in stoic_inds], [np.minimum(x_out[i]+10.0,50.0) for i in stoic_inds]
        # swarmsize = 2*len(stoic_inds)
        # x_opt, _ = pso(fun2, lb, ub, swarmsize=swarmsize, maxiter=50, omega=0.5, 
        #                 phip=0.25, phig=0.85, minfunc=1e-3, minstep=1e-3)

        def constraint_fun(x_in):
            x = np.copy(x_out)
            x[stoic_inds] = x_in
            out = (np.array(self.kinetic_cell.compute_residuals(x))).flatten()
            return out

        res_constraint = NonlinearConstraint(constraint_fun, -1e-4, 1e-4)
        bnds = [(lb[i], ub[i]) for i in range(len(stoic_inds))]
        popsize = 3*len(stoic_inds)
        init = np.random.rand(popsize, len(stoic_inds)) # initialize population and project to feasible set
        for i in range(popsize):
            x = np.copy(x_out)
            x[stoic_inds] = init[i,:]
            result = minimize(res_fun, x, bounds=[(lb_proj[i], ub_proj[i]) for i in range(x_out.shape[0])])
            init[i,:] = np.copy(result.x[stoic_inds])

        result = differential_evolution(fun2, bnds, maxiter=100, popsize=popsize, init=init, constraints=(res_constraint))
        # result = minimize(fun2, np.array([x_out[i] for i in stoic_inds]), bounds=bnds, constraints=(res_constraint))
        x_opt = result.x
        
        x_out[stoic_inds] = np.copy(x_opt)

        ### TEST PROJECTION
        result = minimize(res_fun, x_out)
        x_out = np.copy(result.x)


        # Plot and record stage 2 results
        plt.figure()
        plt.plot(data_dict['Time'], 100*data_dict['O2'], 'b-', label=str(hr))
        y_dict = self.kinetic_cell.get_rto_data(x_out, heating_data, IC) 
        plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')
        plt.legend(['Observed', 'Predicted'])
        plt.xlabel('Time')
        plt.ylabel(r'$O_2$ consumption [% mol]')
        plt.title(r'Warm Start $O_2$ Stage 2')
        plt.savefig(os.path.join(self.figs_dir,'final_results', 'warm_start_stage2.png'))
        plt.close()

        with open(self.report_file, 'a+') as fileID:
            print('======================================= Stage 2 Results ===================================\n', file=fileID)
            self.kinetic_cell.log_status(x_out, fileID)
            print('===================================== End Stage 2 Results =================================\n\n', file=fileID)
        '''
        '''
        #### STEP 3: Optimize A and Ea to match beginning and end times + peak position
        warm_start_cost3 = self.get_warm_start_cost(cost_fun_type='peaks', stage='3', log_dir='warm_start_3a')
        def fun3(x_in):
            # Setup parameter vector
            x = np.copy(x_out)
            x[preexp_inds] = x_in[0]*np.ones_like(preexp_inds)
            x[acteng_inds] = x_in[1]*np.ones_like(acteng_inds)

            # Compute cost
            cost = warm_start_cost3(x)

            # Log optimization status
            with open(self.log_file, 'a+') as fileID:
                print('========================================== Status at Warm Start (Stage 3) - Iteration {} ========================================='.format(str(self.function_evals)), file=fileID)
                self.kinetic_cell.log_status(x, fileID)
                print('Cost: {}'.format(cost), file=fileID)
                print('======================================== End Status at Warm Start (Stage 3) - Iteration {} =======================================\n\n'.format(str(self.function_evals)), file=fileID)

            return cost
        lb = [np.log(1e-2), x_out[acteng_inds[0]]-2]
        ub = [np.log(1e4), x_out[acteng_inds[0]]+2]
        x_opt, _ = pso(fun3, lb, ub, swarmsize=4*len(preexp_inds), maxiter=50, phip=0.25, phig=0.85, minfunc=1e0, minstep=1e0)
        x_out[preexp_inds] = np.copy(x_opt[0])*np.ones_like(preexp_inds)
        x_out[acteng_inds] = np.copy(x_opt[1])*np.ones_like(acteng_inds)
        '''

        # Assemble final bounds to use
        lb_out[preexp_inds] = np.log(1e-2) # Bound within 1 order of magnitude
        lb_out[acteng_inds] = np.copy(x_out[acteng_inds]) - 4.6 #   of ballpark value
        ub_out[preexp_inds] = np.log(1e5)
        ub_out[acteng_inds] = np.copy(x_out[acteng_inds]) + 4.6
        lb_out[stoic_inds] = 1e-2
        ub_out[stoic_inds] = 50.0

        '''
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
        plt.savefig(os.path.join(self.figs_dir,'final_results', 'warm_start_stage3.png'))
        plt.close()

        with open(self.report_file, 'a+') as fileID:
            print('======================================= Stage 3 Results ===================================\n', file=fileID)
            self.kinetic_cell.log_status(x_out, fileID)
            print('===================================== End Stage 3 Results =================================\n\n', file=fileID)

        #### OLD STEP 3: Match simulation peak to beginning/end to get bound for A and Ea
        # Iterations: pre-exp lower bound, pre-exp upper bound, Ea lower bound, Ea upper bound
        points_data = [['end','start'], ['start','end'], ['start','end'], ['end','start']]
        lb_list = [[bnd_preexp[0]], [x_out[preexp_inds[0]]], [bnd_acteng[0]], [x_out[acteng_inds[0]]]]
        ub_list = [[x_out[preexp_inds[0]]], [bnd_preexp[1]], [x_out[acteng_inds[0]]], [bnd_acteng[1]]]
        messages = ['3 - A lb','3 - A ub','3 - Ea lb','3 - Ea ub']
        log_dirs = ['warm_start_3a','warm_start_3b','warm_start_3c','warm_start_3d']

        for i, pts in enumerate(points_data):
            warm_start_cost = self.get_warm_start_cost(cost_fun_type='points', points=pts, stage=messages[i], log_dir = log_dirs[i])
            
            def fun3(x_in):
                # Setup parameter vector
                x = np.copy(x_out)

                if i in [0,1]:
                    x[preexp_inds] = x_in*np.ones_like(preexp_inds)
                else:
                    x[acteng_inds] = x_in*np.ones_like(acteng_inds)

                # Compute cost
                cost = warm_start_cost(x)

                # Log optimization status
                with open(self.log_file, 'a+') as fileID:
                    print('========================================== Status at Warm Start (Stage {}) - Iteration {} ========================================='.format(messages[i], str(self.function_evals)), file=fileID)
                    self.kinetic_cell.log_status(x, fileID)
                    print('Cost: {}'.format(cost), file=fileID)
                    print('======================================== End Status at Warm Start (Stage {}) - Iteration {} =======================================\n\n'.format(messages[i], str(self.function_evals)), file=fileID)

                return cost
            
            x_opt, _ = pso(fun3, lb_list[i], ub_list[i], swarmsize=10, maxiter=25, 
                            phip=0.25, phig=0.85, minfunc=1e0, minstep=1e0)
            
            # Parse and record output
            x_temp = np.copy(x_out)
            if i==0:
                lb_out[preexp_inds] = np.copy(x_opt)*np.ones_like(preexp_inds)
                x_temp[preexp_inds] = np.copy(x_opt)*np.ones_like(preexp_inds)
                save_filename = os.path.join(self.figs_dir,'final_results', 'warm_start_stage3a.png')
                message = r'$A$ lower bound'

            elif i==1:
                ub_out[preexp_inds] = np.copy(x_opt)*np.ones_like(preexp_inds)
                x_temp[preexp_inds] = np.copy(x_opt)*np.ones_like(preexp_inds)
                save_filename = os.path.join(self.figs_dir,'final_results', 'warm_start_stage3b.png')
                message = r'$A$ upper bound'

            elif i==2:
                lb_out[acteng_inds] = np.copy(x_opt)*np.ones_like(acteng_inds)
                x_temp[acteng_inds] = np.copy(x_opt)*np.ones_like(acteng_inds)
                save_filename = os.path.join(self.figs_dir,'final_results', 'warm_start_stage3c.png')
                message = r'$E_a$ lower bound'

            elif i==3:
                ub_out[acteng_inds] = np.copy(x_opt)*np.ones_like(acteng_inds)
                x_temp[acteng_inds] = np.copy(x_opt)*np.ones_like(acteng_inds)
                save_filename = os.path.join(self.figs_dir,'final_results', 'warm_start_stage3d.png')
                message = r'$E_a$ upper bound'

            plt.figure()
            plt.plot(data_dict['Time'], 100*data_dict['O2'], 'b-', label=str(hr))
            y_dict = self.kinetic_cell.get_rto_data(x_temp, heating_data, IC) 
            plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')
            plt.legend(['Observed', 'Predicted'])
            plt.xlabel('Time')
            plt.ylabel(r'$O_2$ consumption [% mol]')
            plt.title(r'Warm Start $O_2$ Stage 3 - {}'.format(message))
            plt.savefig(save_filename)
            plt.close()

        lb_out[stoic_inds] = np.maximum(x_out[stoic_inds]-5.0, 1e-2) # Set bounds for stoichiometric parameters
        ub_out[stoic_inds] = np.minimum(x_out[stoic_inds]+5.0, 50)   # These are the final sets of bounds that will be used in the optimization

        with open(self.report_file, 'a+') as fileID:
            print('======================================= Stage 3 Results ===================================\n', file=fileID)
            
            print('| -------- Variable -------- | -- Lower Bound -- | -- Upper Bound -- |', file=fileID)
            print('{:<30}{:<20}{:<20}'.format('Pre-exponential Factor:', np.round(np.exp(lb_out[preexp_inds[0]]),decimals=2), np.round(np.exp(ub_out[preexp_inds[0]]),decimals=2)), file=fileID)
            print('{:<30}{:<20}{:<20}'.format('Activation Energy:', np.round(np.exp(lb_out[acteng_inds[0]]),decimals=2), np.round(np.exp(ub_out[acteng_inds[0]]),decimals=2)), file=fileID)

            print('===================================== End Stage 3 Results =================================\n\n', file=fileID)
        '''
        
        #### STEP 4: Optimize A and Ea to match peaks in effluence data
        warm_start_cost4 = self.get_warm_start_cost(cost_fun_type='peaks', stage='4', log_dir='warm_start_4')

        def fun4(x_in):
            # Setup parameter vector
            x = x_in # np.copy(x_out)
            # x[preexp_inds] = x_in[:len(preexp_inds)]
            # x[acteng_inds] = x_in[len(preexp_inds):]

            # Compute cost
            if np.sum(np.abs(self.kinetic_cell.compute_residuals(x)))>1e-4:
                cost = 1e6
            else:
                cost = warm_start_cost4(x)

            # Log optimization status
            with open(self.log_file, 'a+') as fileID:
                print('========================================== Status at Warm Start (Stage 4) - Iteration {} ========================================='.format(str(self.function_evals)), file=fileID)
                self.kinetic_cell.log_status(x, fileID)
                print('Cost: {}'.format(cost), file=fileID)
                print('======================================== End Status at Warm Start (Stage 4) - Iteration {} =======================================\n\n'.format(str(self.function_evals)), file=fileID)

            return cost

        # # Form constraints
        # fuel_names_all = self.kinetic_cell.fuel_names + self.kinetic_cell.pseudo_fuel_comps
        # fuel_dict = {}
        # for i in range(self.kinetic_cell.num_rxns):
        #     for s in self.kinetic_cell.reac_names[i]:
        #         if s in fuel_names_all:
        #             if s not in fuel_dict.keys():
        #                 fuel_dict[s] = []
        #             fuel_dict[s].append(i)
        
        # A = np.zeros((1,self.kinetic_cell.num_rxns))
        # for i in range(self.kinetic_cell.num_rxns):
        #     a_reac = np.zeros((1,self.kinetic_cell.num_rxns))
        #     a_reac[0,i] = -1

        #     for s in self.kinetic_cell.prod_names[i]:
        #         if s in fuel_names_all:
        #             for i in fuel_dict[s]:
        #                 a_prod = np.copy(a_reac)
        #                 a_prod[0,i] = 1
        #                 A = np.concatenate((A,a_prod))
        # A = A[1:,:]

        # def constraint_fun4(x):
        #     Ea = x[len(preexp_inds):]
        #     y = np.squeeze(np.dot(A, Ea))
        #     return y
        
        # x_out = np.array([np.log(13.34), np.log(32483.33), 2.55, 0.46, np.log(13.34), np.log(32482.33),
        #                     5.99, 2.42, np.log(13.34), np.log(32482.33), 1.36, 3.77, 4.49])
        popsize = x_out.shape[0] # 4*len(preexp_inds)
        # init = np.expand_dims(np.concatenate((x_out[preexp_inds],x_out[acteng_inds])),0) + 0.25*np.random.randn(popsize, 6)
        init = np.expand_dims(x_out, 0) + 0.25*np.random.randn(popsize, x_out.shape[0])
        # acteng_constraint = NonlinearConstraint(lambda x: constraint_fun4(x), 0, np.inf)
        # bnds = [(np.log(2.56)-0.7,np.log(18.43)+0.7), (np.log(2.56),np.log(18.43)), (np.log(2.56),np.log(18.43)), \
        #         (np.log(25400.7),np.log(48830.24)), (np.log(25400.7),np.log(48830.24)), (np.log(25400.7),np.log(48830.24))]
        # bnds = [(-4.6, 9.21), (-4.6, 9.21), (-4.6, 9.21), (np.log(25400.7),np.log(48830.24)+0.7), (np.log(25400.7),np.log(48830.24)+0.7), (np.log(25400.7),np.log(48830.24)+0.7)]
        # result = differential_evolution(fun4, bnds, init=init, constraints=acteng_constraint, maxiter=100, popsize=popsize, polish=False)
        bnds = [(lb_out[i], ub_out[i]) for i in range(x_out.shape[0])] # [(lb_out[i], ub_out[i]) for i in preexp_inds] + [(lb_out[i], ub_out[i]) for i in acteng_inds]
        result = differential_evolution(fun4, bnds, init=init, maxiter=100, popsize=popsize, polish=False)
        x_opt = result.x

        x_out = np.copy(x_opt)
        # x_out[preexp_inds] = np.copy(x_opt[:len(preexp_inds)])
        # x_out[acteng_inds] = np.copy(x_opt[len(preexp_inds):])

        # Plot and record stage 4 results
        plt.figure()
        plt.plot(data_dict['Time'], 100*data_dict['O2'], 'b-', label=str(hr))
        y_dict = self.kinetic_cell.get_rto_data(x_out, heating_data, IC)
        IC['Oil'] = self.compute_init_oil_sat(x_out, data_dict)
        plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')
        plt.legend(['Observed', 'Predicted'])
        plt.xlabel('Time')
        plt.ylabel(r'$O_2$ consumption [% mol]')
        plt.title(r'Warm Start $O_2$ Stage 4')
        plt.savefig(os.path.join(self.figs_dir, 'final_results', 'warm_start_stage4.png'))
        plt.close()

        with open(self.report_file, 'a+') as fileID:
            print('======================================= Stage 4 Results ===================================\n', file=fileID)
            self.kinetic_cell.log_status(x_out, fileID)
            print('===================================== End Stage 4 Results =================================\n\n', file=fileID)
        
        ####### FOR TESTING ONLY ########
        # x_out = np.array([np.log(13.34), np.log(33077.16), 2.55, 0.46, np.log(10.12), np.log(59900.39),
        #                     5.99, 2.42, np.log(11.66), np.log(31075.35), 1.36, 3.77, 4.49])
        # ub_out[preexp_inds] = 9.2*np.ones_like(preexp_inds)
        # lb_out[preexp_inds] = -4.6*np.ones_like(preexp_inds)
        
        # lb_out[acteng_inds] = np.log(25400.7)*np.ones_like(acteng_inds)
        # ub_out[acteng_inds] = np.log(48830.24)*np.ones_like(acteng_inds)+0.7

        # lb_out[stoic_inds] = np.maximum(x_out[stoic_inds]-5.0, 1e-2) # Set bounds for stoichiometric parameters
        # ub_out[stoic_inds] = np.minimum(x_out[stoic_inds]+5.0, 50)   # These are the final sets of bounds that will be used in the optimization
        
        #### STEP 5: Optimize all parameters to fit lowest heating rate
        warm_start_cost5 = self.get_warm_start_cost(cost_fun_type='mse', stage='5', log_dir='warm_start_5')
        
        def fun5(x):
            # Compute cost
            if np.sum(np.abs(self.kinetic_cell.compute_residuals(x)))>1e-4:
                cost = 1e6
            else:
                cost = warm_start_cost5(x)

            # Log optimization status
            with open(self.log_file, 'a+') as fileID:
                print('========================================== Status at Warm Start (Stage 5) - Iteration {} ========================================='.format(str(self.function_evals)), file=fileID)
                self.kinetic_cell.log_status(x, fileID)
                print('Cost: {}'.format(cost), file=fileID)
                print('======================================== End Status at Warm Start (Stage 5) - Iteration {} =======================================\n\n'.format(str(self.function_evals)), file=fileID)

            return cost
        
        bnds_out = [(lb_out[i], ub_out[i]) for i in range(ub_out.shape[0])]
        
        popsize = x_out.shape[0]
        maxiter = 100
        res_constraint = NonlinearConstraint(lambda x: (np.array(self.kinetic_cell.compute_residuals(x))).flatten(), -1e-5, 1e-5)
        init = np.expand_dims(x_out, 0) + 0.05*np.random.randn(popsize, x_out.shape[0])
        
        result = differential_evolution(fun5, bnds_out, popsize=popsize, init=init, polish=False, maxiter=maxiter,constraints=(res_constraint))
        # result = minimize(fun5, x_out, bounds=bnds_out, constraints=(res_constraint))
        x_out = np.copy(result.x)

        # Plot and record stage 5 results
        plt.figure()
        plt.plot(data_dict['Time'], 100*data_dict['O2'], 'b-', label=str(hr))
        IC['Oil'] = self.compute_init_oil_sat(x_out, data_dict)
        y_dict = self.kinetic_cell.get_rto_data(x_out, heating_data, IC) 
        plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')
        plt.legend(['Observed', 'Predicted'])
        plt.xlabel('Time')
        plt.ylabel(r'$O_2$ consumption [% mol]')
        plt.title(r'Warm Start $O_2$ Stage 5')
        plt.savefig(os.path.join(self.figs_dir, 'final_results', 'warm_start_stage5.png'))
        plt.close()

        with open(self.report_file, 'a+') as fileID:
            print('======================================= Stage 5 Results ===================================\n', file=fileID)
            self.kinetic_cell.log_status(x_out, fileID)
            print('===================================== End Stage 5 Results =================================\n\n', file=fileID)

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
                heating_data = {'Time': data_dict['Time'], 'Temp': data_dict['Temp']}
                IC = {'Temp': data_dict['Temp'][0], 'O2': data_dict['O2_con_in'], 'Oil': Oil_con_init} #self.data_container.Oil_con_init}
                y_dict = self.kinetic_cell.get_rto_data(x, heating_data, IC) 
                plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')
                legend_list.append('Simulation')

                if cost_fun_type == 'points':
                    
                    # experimental data point
                    if points[0] == 'peak':
                        O2_data_point = data_dict['Time'][np.argmax(data_dict['O2'])]
                        # CO2_data_point = data_dict['Time'][np.argmax(data_dict['CO2'])]
                    elif points[0] == 'start':
                        O2_data_point = np.amin(data_dict['Time'][data_dict['O2']>0.05*np.amax(data_dict['O2'])])
                        # CO2_data_point = np.amin(data_dict['Time'][data_dict['CO2']>0.05*np.amax(data_dict['CO2'])])
                    elif points[0] == 'end':
                        O2_data_point = np.amax(data_dict['Time'][data_dict['O2']>0.05*np.amax(data_dict['O2'])])
                        # CO2_data_point = np.amax(data_dict['Time'][data_dict['CO2']>0.05*np.amax(data_dict['CO2'])])
                    
                    # simulation data point
                    if points[1] == 'peak':
                        O2_sim_point = y_dict['Time'][np.argmax(y_dict['O2'])]
                        # CO2_sim_point = y_dict['Time'][np.argmax(y_dict['CO2'])]
                    elif points[1] == 'start':
                        O2_sim_point = np.amin(y_dict['Time'][y_dict['O2']>0.05*np.amax(y_dict['O2'])])
                        # CO2_sim_point = np.amin(y_dict['Time'][y_dict['CO2']>0.05*np.amax(y_dict['CO2'])])
                    elif points[1] == 'end':
                        O2_sim_point = np.amax(y_dict['Time'][y_dict['O2']>0.05*np.amax(y_dict['O2'])])
                        # CO2_sim_point = np.amax(y_dict['Time'][y_dict['CO2']>0.05*np.amax(y_dict['CO2'])])

                    loss = (O2_data_point - O2_sim_point)**2 #+ (CO2_data_point - CO2_sim_point)**2
                

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
                    peak_inds_O2_data = find_peaks_cwt(data_dict['O2'], np.arange(1,np.round(data_dict['O2'].shape[0]/6),step=1), noise_perc=2)
                    peak_inds_O2_data = np.delete(peak_inds_O2_data, np.where(peak_inds_O2_data > O2_data_end_ind)) # throw out late peaks
                    peak_inds_O2_data = np.delete(peak_inds_O2_data, np.where(peak_inds_O2_data < O2_data_start_ind)) # throw out early peaks
                    
                    peak_inds_O2_sim = find_peaks_cwt(y_dict['O2'], np.arange(1,np.round(y_dict['O2'].shape[0]/6),step=1), noise_perc=2)
                    peak_inds_O2_sim = np.delete(peak_inds_O2_sim, np.where(peak_inds_O2_sim > O2_sim_end_ind))
                    peak_inds_O2_sim = np.delete(peak_inds_O2_sim, np.where(peak_inds_O2_sim < O2_sim_start_ind))

                    # if peak_inds_O2_sim.shape[0] < peak_inds_O2_data.shape[0]:
                    #     peak_inds_O2_sim = np.concatenate((peak_inds_O2_sim, [O2_sim_end_ind]*(peak_inds_O2_data.shape[0] - peak_inds_O2_sim.shape[0])))

                    # peak_inds_CO2_data = find_peaks_cwt(data_dict['CO2'], np.arange(1,np.round(data_dict['CO2'].shape[0]/4),step=1))
                    # peak_inds_CO2_sim = find_peaks_cwt(y_dict['CO2'], np.arange(1,np.round(y_dict['CO2'].shape[0]/4),step=1))
                    # if peak_inds_CO2_sim.shape[0] < peak_inds_CO2_data.shape[0]:
                    #     peak_inds_CO2_sim = np.concatenate((peak_inds_CO2_sim, [y_dict['CO2'].shape[0]-1]*(peak_inds_CO2_data.shape[0] - peak_inds_CO2_sim.shape[0])))

                    plt.scatter(data_dict['Time'][peak_inds_O2_data], 100*data_dict['O2'][peak_inds_O2_data],c='r', marker='o')
                    plt.scatter(y_dict['Time'][peak_inds_O2_sim], 100*y_dict['O2'][peak_inds_O2_sim],c='r', marker='X')
                    legend_list.append('Data Peaks')
                    legend_list.append('Simulation Peaks')

                    # # Number of peaks loss
                    # num_loss = np.abs(peak_inds_O2_data.shape[0] - peak_inds_O2_sim.shape[0]) + \
                    #             np.abs(peak_inds_CO2_data.shape[0] - peak_inds_CO2_sim.shape[0])

                    num_peaks_diff = np.abs(peak_inds_O2_data.shape[0] - peak_inds_O2_sim.shape[0]) + 1
                    diff_loss = 0
                    for i in range(peak_inds_O2_sim.shape[0]):
                        data_idx = (np.abs(data_dict['Time'][peak_inds_O2_data] - y_dict['Time'][peak_inds_O2_sim[i]])).argmin()
                        diff_loss += (data_dict['Time'][data_idx] - y_dict['Time'][peak_inds_O2_sim[i]])**2
                        # diff_loss += ((data_dict['Time'][data_idx] - y_dict['Time'][peak_inds_O2_sim[i]])/data_dict['Time'].max())**2
                        # diff_loss += ((data_dict['O2'][data_idx] - y_dict['O2'][peak_inds_O2_sim[i]])/data_dict['O2'].max())**2

                    for i in range(peak_inds_O2_data.shape[0]):
                        sim_idx = (np.abs(y_dict['Time'][peak_inds_O2_sim] - data_dict['Time'][peak_inds_O2_data[i]])).argmin()
                        # diff_loss += ((y_dict['Time'][sim_idx] - data_dict['Time'][peak_inds_O2_data[i]])/data_dict['Time'].max())**2
                        diff_loss += (y_dict['Time'][sim_idx] - data_dict['Time'][peak_inds_O2_data[i]])**2
                        # diff_loss += ((y_dict['O2'][sim_idx] - data_dict['O2'][peak_inds_O2_data[i]])/data_dict['O2'].max())**2

                    # for i in range(peak_inds_CO2_sim.shape[0]):
                    #     data_idx = (np.abs(data_dict['Time'][peak_inds_CO2_data] - y_dict['Time'][peak_inds_CO2_sim[i]])).argmin()
                    #     # diff_loss += ((data_dict['Time'][data_idx] - y_dict['Time'][peak_inds_CO2_sim[i]])/data_dict['Time'].max())**2
                    #     diff_loss += (data_dict['Time'][data_idx] - y_dict['Time'][peak_inds_CO2_sim[i]])**2
                    #     # diff_loss += ((data_dict['CO2'][data_idx] - y_dict['CO2'][peak_inds_CO2_sim[i]])/data_dict['CO2'].max())**2

                    # for i in range(peak_inds_CO2_data.shape[0]):
                    #     sim_idx = (np.abs(y_dict['Time'][peak_inds_CO2_sim] - data_dict['Time'][peak_inds_CO2_data[i]])).argmin()
                    #     # diff_loss += ((y_dict['Time'][sim_idx] - data_dict['Time'][peak_inds_O2_data[i]])/data_dict['Time'].max())**2
                    #     diff_loss += ((y_dict['Time'][sim_idx] - data_dict['Time'][peak_inds_O2_data[i]]))**2
                    #     # diff_loss += ((y_dict['CO2'][sim_idx] - data_dict['CO2'][peak_inds_O2_data[i]])/data_dict['CO2'].max())**2

                    loss = 1e-3*num_peaks_diff*diff_loss + endpts_loss # + 100*num_loss

                elif cost_fun_type == 'mse':
                    O2_MSE = np.mean(np.power(data_dict['O2'] - np.interp(data_dict['Time'], y_dict['Time'], y_dict['O2']),2))
                    CO2_MSE = np.mean(np.power(data_dict['CO2'] - np.interp(data_dict['Time'], y_dict['Time'], y_dict['CO2']),2))
                    loss = O2_MSE + CO2_MSE

                elif cost_fun_type == 'l1':
                    O2_L1 = np.mean(np.abs(data_dict['O2'] - np.interp(data_dict['Time'], y_dict['Time'], y_dict['O2'])))
                    CO2_L1 = np.mean(np.abs(data_dict['CO2'] - np.interp(data_dict['Time'], y_dict['Time'], y_dict['CO2'])))
                    loss = O2_L1 + CO2_L1

            except:
                loss = 1e5

            plt.xlabel('Time')
            plt.ylabel(r'$O_2$ consumption [% mol]')
            plt.title(r'Warm Start $O_2$ consumption (Stage {})'.format(stage))
            plt.legend(legend_list)
            
            # Print figure with O2 consumption data
            plt.savefig(os.path.join(self.kinetic_cell.results_dir, 'figures', log_dir, 'O2_overlay_{}.png'.format(self.function_evals)))
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

        G = nx.Graph() 
        V = self.kinetic_cell.pseudo_fuel_comps+['Oil','O2'] # Create notes for every fuel species and O2
        G.add_nodes_from(V)

        for i in range(self.kinetic_cell.num_rxns):
            reac_fuel = [c for c in self.kinetic_cell.reac_names[i] if c is not 'O2'][0]

            if 'O2' in self.kinetic_cell.reac_names[i]:
                G.add_edge('O2', reac_fuel, weight=reac_coeff[i,self.kinetic_cell.comp_names.index('O2')])

            for c in self.kinetic_cell.prod_names[i]:
                if c in self.kinetic_cell.pseudo_fuel_comps:
                    G.add_edge(reac_fuel, c, weight=prod_coeff[i,self.kinetic_cell.comp_names.index(c)])
        
        P = all_simple_paths(G, 'O2', 'Oil')
        nu = 0
        for path in map(nx.utils.pairwise, P):
            Ws = [G.edges[u,v]['weight'] for u,v in list(path)]
            nu += np.product(Ws)
        
        P = 1e5 # pressure in pascals
        R = 8.3145
        Flow = 100 * 1e-6 # m^3 / min
        O2_conversion = P/R*np.trapz(Flow*data_dict['O2']/(25.0+273.15), x=data_dict['Time'])
        Oil_conversion = O2_conversion / nu # mole
        Oil_mass = 1e3 * self.kinetic_cell.material_dict['M']['Oil'] * Oil_conversion  # in g
        Vol = 1.32*1.32*3 # kinetic cell volume in cm^3
        void_V = 0.36*Vol # void volume [Porosity x Volume]
        rho = 0.965 # density of oil in g/cm^3

        oil_sat = Oil_mass / rho / void_V # Calculate final oil saturation
        oil_sat = np.maximum(np.minimum(oil_sat, 0.99), 1e-4)

        return oil_sat


    def analyze_uncertainty(self):
        '''
        Perform sensitivity analysis and uncertainty quantification on parameters
        '''

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

        if os.path.exists(os.path.join(self.load_dir, 'sensitivity_responses.npy')):
            param_values = np.load(os.path.join(self.load_dir, 'sensitivity_inputs.npy'))
            Y = np.load(os.path.join(self.load_dir, 'sensitivity_responses.npy'))
        
        else:
            param_values = saltelli.sample(problem, 50)
            Y = np.zeros([param_values.shape[0]])

            for i, X in enumerate(param_values):
                Y[i] = self.base_cost(np.array(X))

            np.save(os.path.join(self.load_dir, 'sensitivity_inputs.npy'), param_values)
            np.save(os.path.join(self.load_dir, 'sensitivity_responses.npy'), Y)

        Si = sobol.analyze(problem, Y)

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


    
    def compute_likelihood_intervals(self, x):
        '''
        Compute the likelihood interval at a given parameter value. 

        Inputs:
            x - parameters around which to compute the interval

        Prints the likelihood interval to the log file.

        '''

        H = numerical_hessian(x, self.base_cost)
        I = np.linalg.inv(H)
        rv = multivariate_normal(mean=x, cov=I)


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

