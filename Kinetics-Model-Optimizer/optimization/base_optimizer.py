import time
import pickle as pkl
import os
import shutil
import copy
import argparse
from random import shuffle
from abc import ABC, abstractmethod

import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import minimize
from scipy.signal import find_peaks_cwt
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal

from utils.utils import mkdirs, numerical_hessian


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
        parser.add_argument('--O2_sigma', type=float, default=1e-2, help='sigma for O2 covariance')
        parser.add_argument('--CO2_sigma', type=float, default=1e-2, help='sigma for CO2 covariance')
        parser.add_argument('--loss_sigma', type=float, default=1e-2, help='sigma for noise in signal')
        parser.add_argument('--CO2_O2_sigma', type=float, default=1e-3, help='sigma for CO2/O2 covariance')

        parser.set_defaults(log_params=True)

        return parser


    def __init__(self, kinetic_cell, data_container, opts):
        
        self.autodiff_enable = opts.autodiff_enable
        self.data_container = data_container
        self.kinetic_cell = kinetic_cell
        self.base_cost = self.get_base_cost_fun(kinetic_cell, data_container, opts)
        
        # Logging information
        self.function_evals = 0
        self.loss_values = []
        self.log_file = os.path.join(self.kinetic_cell.results_dir, opts.log_file)
        with open(self.log_file,'w') as fileID: # clear any existing log file
            pass
        
        self.figs_dir = os.path.join(self.kinetic_cell.results_dir, 'figures')
        if os.path.exists(self.figs_dir):
            shutil.rmtree(self.figs_dir)
        mkdirs([self.figs_dir])

        self.data_container.print_curves(save_path=os.path.join(self.figs_dir, 'rto_curves.png'))

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
                time_diffs = np.expand_dims(Time, 0) - np.expand_dims(Time, 1)
                
                # O2 only case
                if 'O2' in opts.output_loss_inputs and 'CO2' not in opts.output_loss_inputs:
                    x_data = data_dict['O2']
                    x_sim = y_dict['O2']
                    M = opts.O2_sigma*np.exp(-np.power(time_diffs,2) / 2 / opts.O2_sigma**2) + opts.loss_sigma*np.eye(N)

                # CO2 only case
                elif 'O2' not in opts.output_loss_inputs and 'CO2' in opts.output_loss_inputs:
                    x_data = data_dict['CO2']
                    x_sim = y_dict['CO2']
                    M = opts.O2_sigma*np.exp(-np.power(time_diffs,2) / 2 / opts.CO2_sigma**2) + opts.loss_sigma*np.eye(N)

                elif 'O2' in opts.output_loss_inputs and 'CO2' in opts.output_loss_inputs:
                    x_data = np.concatenate([data_dict['O2'], data_dict['CO2']])
                    x_sim = np.concatenate([y_dict['O2'], y_dict['CO2']])

                    A = opts.O2_sigma*np.exp(-np.power(time_diffs,2)/ 2 / opts.O2_sigma**2) + opts.loss_sigma*np.eye(N)
                    B = 1e-1*np.exp(-np.power(time_diffs,2) / 2 / opts.CO2_O2_sigma**2)
                    C = opts.O2_sigma*np.exp(-np.power(time_diffs,2)/ 2 / opts.CO2_sigma**2) + opts.loss_sigma*np.eye(N)
                    M = np.block([[A, B], [B, C]])
                
                x = x_data - x_sim
                L = cholesky(M, lower=True)

                # Referenced https://stats.stackexchange.com/questions/186307/
                #                  efficient-stable-inverse-of-a-patterned-covariance-matrix-for-gridded-data
                alpha = np.solve(L.T, np.solve(L, x))
                loss = 0.5*np.dot(alpha, x) + N*np.log(2*np.pi) + np.trace(np.log(L))

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
        

        def base_fun(x):
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

                # Plot experimental data
                plt.plot(data_dict['Time'], data_dict['O2'], colors[i]+'-', label=str(hr))
                            
                try:
                    heating_data = {'Time': data_dict['Time'], 'Temp': data_dict['Temp']}
                    IC = {'Temp': data_dict['Temp'][0], 'O2': data_dict['O2_con_in'], 'Oil': self.data_container.Oil_con_init}
                    y_dict = self.kinetic_cell.get_rto_data(x, heating_data, IC) 

                    loss += self.output_loss(data_dict, y_dict)
                    plt.plot(Time, O2_sim_out, colors[i]+'--')
                
                except:
                    loss += 1e4
            
            plt.xlabel('Time')
            plt.ylabel(r'$O_2$ consumption')
            plt.title(r'$O_2$ consumption')
            plt.legend()
            
            # Print figure with O2 consumption data
            plt.savefig(os.path.join(self.kinetic_cell.results_dir, 'figures', 'O2_overlay_{}.png'.format(self.function_evals)))
            plt.close()

            # Save loss value
            self.loss_values.append(loss)

            return loss

        return base_fun

    
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

