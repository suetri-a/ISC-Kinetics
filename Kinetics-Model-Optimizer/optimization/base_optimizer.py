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

from utils.utils import mkdirs


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
            output_loss = self.gaussian_likelihood


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
                Time, Temp, O2 = self.data_container.heating_data[hr]

                # Plot experimental data
                plt.plot(Time, O2, colors[i]+'-', label=str(hr))
                            
                try:
                    heating_data = {'Time': Time, 'Temp': Temp}
                    IC = {'Temp': Temp[0], 'O2': self.data_container.O2_con_in[i], 'Oil': self.data_container.Oil_con_init}
                    t, O2_sim = self.kinetic_cell.get_O2_consumption(x, heating_data, IC) 

                    O2_sim_out = np.interp(Time, t, O2_sim)
                    loss += self.output_loss(O2, O2_sim_out)
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

