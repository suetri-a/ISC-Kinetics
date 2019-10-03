import time
import pickle as pkl
import copy
import argparse
from random import shuffle

import autograd.numpy as np
from autograd import grad
import scipy as sp
from scipy.optimize import minimize
from scipy.signal import find_peaks_cwt


class BaseOptimizer():

    @staticmethod
    def modify_cmd_options(parser):
        
        parser.add_argument('--optim_write_file', type=str, default='optim_write_file.pkl', help='file to output optimization message')
        parser.add_argument('--optim_verbose', type=eval, default=False, help='output messages during optimization')
        parser.add_argument('--optim_write_output', type=eval, default=True, help='write results of optimization to disk')
        parser.add_argument('--num_peaks', type=int, nargs='+', action='append', default=[2, 2, 2, 2, 2, 2], 
            help='number of peaks for each RTO experiment')
        parser.add_argument('--peak_weight_vec', type=float, nargs='+', action='append', default=[75.0, 75.0, 250.0, 250.0], 
            help='weight for each part of O2 peaks cost funciton')
        parser.add_argument('--reac_tol', type=float, default=1e-4, help='tolerance for start and end of reaction')

        parser.set_defaults(log_params=True)

        return parser


    def __init__(self, kinetic_cell, data_cell, opts):
        
        self.autodiff_enable = opts.autodiff_enable
        self.data_cell = data_cell
        self.kinetic_cell = kinetic_cell
        self.base_cost = self.get_base_cost_fun(kinetic_cell, data_cell, opts)


    def get_base_cost_fun(self, kinetic_cell, data_cell, opts):
        '''
        Generate log-likelihood cost function for the optimization
        '''

        if opts.output_prior not in ['gaussian', 'exponential']:
            raise Exception('Autodiff optimizer not supported for non-Gaussian output prior.')
            
        if opts.param_prior not in ['uniform']:
            raise Exception('Autodiff optimizer not support for non-uniform parameter prior.')

        O2_data = data_cell.get_O2_data()
        

        def base_fun(x):   
            if opts.param_prior == 'uniform':
                param_cost = 0
            
            O2_sim = kinetic_cell.get_O2_consumption(x)

            if opts.output_prior in ['ISO_peak', 'O2_peak']:

                if opts.output_prior == 'ISO_peak':
                    '''
                    Calculate cost function for isoconversional cost.
                    '''
                    pass 
                    # activation_energy = kinetic_cell.acteng_from_params(x)
                    # activation_energy[np.isnan(activation_energy)] = 0
                    # diff_v = np.real(activation_energy)-np.real(activation_energy)
                    # diff_v = diff_v[np.where(np.isnan(diff_v)!=1 and np.isinf(diff_v)!=1)]
                    # data_cost = np.sum(diff_v**2)
                
                elif opts.output_prior == 'O2_peak':
                    pass

                    # def find_peak_info(O2_consumption):
                    #     '''
                    #     Find O2 consumption peaks for ground truth data
                    #     '''
                
                    #     # Initialize lists for peak time/values
                    #     peak_ind, peak_val = [], [] # simulation peak times and values
                    #     if peak_widths is None:
                    #         peak_widths = np.arange(np.round(kinetic_cell.time_line.shape[0]/50), 
                    #             np.round(kinetic_cell.time_line.shape[0]/20), 10)
                        
                    #     for i in range(kinetic_cell.num_heats):                
                    #         # Find consumption peaks
                    #         peak_inds = find_peaks_cwt(O2_consumption[:,i], peak_widths)
                
                    #         if peak_inds.shape[0] < opts.num_peaks[i]:
                    #             peak_inds = peak_inds[0]*np.ones((opts.num_peaks[i],1))
                    #         else:
                    #             peak_inds = peak_inds[:opts.num_peaks[i]]
                            
                    #         peak_ind.append(peak_inds.astype(int))
                    #         peak_val.append(O2_consumption[peak_inds.astype(int)])
                            
                    #     return peak_ind, peak_val
                
                
                    # def find_start_end_info(O2_consumption):
                    #     '''
                    #     Find start and end times of a combustion reaction sequence.
                    #     '''
                    #     start_inds = list(np.amin(np.where(O2_consumption >= opts.reac_tol), axis=0))
                    #     end_inds = list(np.amax(np.where(O2_consumption >= opts.reac_tol), axis=0))
                    #     return start_inds, end_inds
                    
                    # # Calculate cost function for O2 consumption.
                    # peak_ind_sim, peak_val_sim = find_peak_info(O2_sim)
                    # peak_ind_gt, peak_val_gt = find_peak_info(O2_data)
                    
                    # start_ind_sim, end_ind_sim = find_start_end_info(O2_sim)
                    # start_ind_gt, end_ind_gt = find_start_end_info(O2_data)
            
                    # # Calculate distance cost function
                    # data_cost = opts.peak_weight_vec[0]*np.sum(((peak_ind_gt - peak_ind_sim) / opts.num_sim_steps)**2) + \
                    #     opts.peak_weight_vec[1]*np.sum(((peak_val_gt - peak_val_sim) / (opts.O2_con_sim-peak_val_gt))**2) + \
                    #     opts.peak_weight_vec[2]*np.sum(((start_ind_sim - start_ind_gt) / opts.num_sim_steps)**2) + \
                    #     opts.peak_weight_vec[3]*np.sum(((end_ind_sim - end_ind_gt) / opts.num_sim_steps)**2)
            
            elif opts.output_prior == 'gaussian':
                data_cost = np.mean(np.power(O2_sim - O2_data, 2))
            
            elif opts.output_prior == 'exponential':
                data_cost = np.mean(np.abs(O2_sim - O2_data))
            
            cost = data_cost + param_cost

            return cost

        return base_fun