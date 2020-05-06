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
from scipy.optimize import minimize, differential_evolution, brute
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
            if os.path.exists(self.figs_dir):
                shutil.rmtree(self.figs_dir)
            mkdirs([self.figs_dir])

            self.warm_start_complete = False
            self.optim_complete = False
            with open(os.path.join(self.load_dir,'warm_start_complete.pkl'),'wb') as fp:
                pickle.dump(self.warm_start_complete, fp)
            with open(os.path.join(self.load_dir,'optim_complete.pkl'),'wb') as fp:
                pickle.dump(self.optim_complete, fp)
            

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
                L = cholesky(M, lower=True)

                # Referenced https://stats.stackexchange.com/questions/186307/
                #                  efficient-stable-inverse-of-a-patterned-covariance-matrix-for-gridded-data
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, x)) # np.linalg.solve(M, x)
                loss = 0.5*np.dot(alpha, x) + N*np.log(2*np.pi) + np.sum(np.log(np.diagonal(L))) # np.linalg.det(M)

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
                    plt.plot(y_dict['Time'], y_dict['O2'], colors[i]+'--')
                
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

    
    def warm_start(self, x0, param_types, return_bounds = False):
        '''
        Get objective function used to warm start optimization by optimizing activation energies

        '''

        if return_bounds:
            lb_out = np.zeros((len(param_types)))
            ub_out = np.zeros((len(param_types)))

        ##### Begin stage 1 of warm start
        bnds_all = self.data_container.compute_bounds(self.kinetic_cell.param_types)
        
        preexp_inds = [i for i,p in enumerate(param_types) if p[0]=='preexp']
        acteng_inds = [i for i,p in enumerate(param_types) if p[0]=='acteng']
        
        bnd_preexp = (np.amin([bnds_all[i][0] for i in preexp_inds]), np.amax([bnds_all[i][1] for i in preexp_inds]))
        bnd_acteng = (np.amin([bnds_all[i][0] for i in acteng_inds]), np.amax([bnds_all[i][1] for i in acteng_inds]))

        if return_bounds:
            cost_types = [['peak', 'peak'], ['start', 'end'], ['end', 'start']]
        else:
            cost_types = [['peak', 'peak']]

        
        for ct in cost_types:
            warm_start_cost = self.get_warm_start_points_cost(points=ct)
            
            def fun1(x_in):
                # Setup parameter vector
                x = np.copy(x0)
                x[preexp_inds] = x_in[0]
                x[acteng_inds] = x_in[1]

                # Compute cost
                cost = warm_start_cost(x)

                # Log optimization status
                with open(self.log_file, 'a+') as fileID:
                    print('========================================== Status at Warm Start (Stage 1) - Iteration {} ========================================='.format(str(self.function_evals)), file=fileID)
                    self.kinetic_cell.log_status(x, fileID)
                    print('Base cost: {}'.format(cost), file=fileID)
                    print('======================================== End Status at Warm Start (Stage 1) - Iteration {} =======================================\n\n'.format(str(self.function_evals)), file=fileID)

                return cost

            x_opt = brute(fun1, (bnd_preexp, bnd_acteng), Ns=25, finish=None)
            
            # Parse output 
            if ct ==['peak','peak']:
                x_out = np.copy(x0)
                x_out[preexp_inds] = np.copy(x_opt[0])*np.ones_like(preexp_inds)
                x_out[acteng_inds] = np.copy(x_opt[1])*np.ones_like(acteng_inds)
            elif ct ==['end','start']:
                ub_out[preexp_inds] = np.copy(x_opt[0])*np.ones_like(preexp_inds)
                ub_out[acteng_inds] = np.copy(x_opt[1])*np.ones_like(acteng_inds)
            elif ct == ['start', 'end']:
                lb_out[preexp_inds] = np.copy(x_opt[0])*np.ones_like(preexp_inds)
                lb_out[acteng_inds] = np.copy(x_opt[1])*np.ones_like(acteng_inds)

        # Post process first stage results
        if return_bounds:
            lb_out = np.minimum(lb_out, ub_out)
            ub_out = np.maximum(lb_out, ub_out)

        x0 = np.copy(x_out) # Reassign with updated frequency factors and activation energies


        ##### Begin stage 2 of warm starting
        stoic_inds = [i for i, p in enumerate(param_types) if p[0]=='stoic']
        bnds = [bnds_all[i] for i in stoic_inds]
        warm_start_cost = self.get_warm_start_effluence_cost()

        def fun2(x_in):
            # Setup parameter vector
            x = np.copy(x0)
            x[stoic_inds] = x_in

            # Compute cost
            cost = warm_start_cost(x)

            # Log optimization status
            with open(self.log_file, 'a+') as fileID:
                print('========================================== Status at Warm Start (Stage 2) - Iteration {} ========================================='.format(str(self.function_evals)), file=fileID)
                self.kinetic_cell.log_status(x, fileID)
                print('Base cost: {}'.format(cost), file=fileID)
                print('======================================== End Status at Warm Start (Stage 2) - Iteration {} =======================================\n\n'.format(str(self.function_evals)), file=fileID)

            return cost

        lb, ub = [l for l,u in bnds], [ u for l,u in bnds ]
        x_opt, _ = pso(fun2, lb, ub, swarmsize=40, maxiter=50, phip=0.5, phig=0.75)

        x_out[stoic_inds] = np.copy(x_opt)
        if return_bounds:
            lb_out[stoic_inds] = np.maximum(x_out[stoic_inds] - 2.0, 1e-2)
            ub_out[stoic_inds] = x_out[stoic_inds] + 2.0


        # Return warm started parameters
        if return_bounds:
            bnds_out = [(l, u) for l, u in zip(lb_out.tolist(), ub_out.tolist())]
            return x_out, bnds_out
        else:
            return x_out

    
    def get_warm_start_points_cost(self, points=['peak','peak']):

        def cost_fun(x):
            '''

            Note: when this function is called, the number of function evaluations is incremented, 
                the data loss value is recorded, and create the overlay plot. The status of the 
                parameters is not recorded.

            '''
            self.function_evals += 1

            # Plot experimet
            plt.figure()

            hr = min(self.data_container.heating_rates)
            data_dict = self.data_container.heating_data[hr]

            # Plot experimental data
            plt.plot(data_dict['Time'], 100*data_dict['O2'], 'b-', label=str(hr))


            try:   
                heating_data = {'Time': data_dict['Time'], 'Temp': data_dict['Temp']}
                IC = {'Temp': data_dict['Temp'][0], 'O2': data_dict['O2_con_in'], 'Oil': self.data_container.Oil_con_init}
                y_dict = self.kinetic_cell.get_rto_data(x, heating_data, IC) 
                plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')

                if points[0] == 'peak':
                    O2_data_point = data_dict['Time'][np.argmax(data_dict['O2'])]
                    CO2_data_point = data_dict['Time'][np.argmax(data_dict['CO2'])]
                elif points[0] == 'start':
                    O2_data_point = np.amin(data_dict['Time'][data_dict['O2']>0.05*np.amax(data_dict['O2'])])
                    CO2_data_point = np.amin(data_dict['Time'][data_dict['CO2']>0.05*np.amax(data_dict['CO2'])])
                elif points[0] == 'end':
                    O2_data_point = np.amax(data_dict['Time'][data_dict['O2']>0.05*np.amax(data_dict['O2'])])
                    CO2_data_point = np.amax(data_dict['Time'][data_dict['CO2']>0.05*np.amax(data_dict['CO2'])])

                if points[1] == 'peak':
                    O2_sim_point = y_dict['Time'][np.argmax(y_dict['O2'])]
                    CO2_sim_point = y_dict['Time'][np.argmax(y_dict['CO2'])]
                elif points[1] == 'start':
                    O2_sim_point = np.amin(y_dict['Time'][y_dict['O2']>0.05*np.amax(y_dict['O2'])])
                    CO2_sim_point = np.amin(y_dict['Time'][y_dict['CO2']>0.05*np.amax(y_dict['CO2'])])
                elif points[1] == 'end':
                    O2_sim_point = np.amax(y_dict['Time'][y_dict['O2']>0.05*np.amax(y_dict['O2'])])
                    CO2_sim_point = np.amax(y_dict['Time'][y_dict['CO2']>0.05*np.amax(y_dict['CO2'])])

                peak_pos_loss = (O2_data_point - O2_sim_point)**2 + (CO2_data_point - CO2_sim_point)**2

            except:
                peak_pos_loss = 1e4

            plt.xlabel('Time')
            plt.ylabel(r'$O_2$ consumption [% mol]')
            plt.title(r'Warm Start $O_2$ consumption (Stage 1 - {}, {})'.format(points[0], points[1]))
            
            # Print figure with O2 consumption data
            plt.savefig(os.path.join(self.kinetic_cell.results_dir, 'figures', 'O2_overlay_{}.png'.format(self.function_evals)))
            plt.close()

            # Final loss value
            loss = peak_pos_loss

            # Save loss value
            self.loss_values.append(loss)

            return loss

        return cost_fun

    
    def get_warm_start_effluence_cost(self):

        def cost_fun(x):
            '''

            Note: when this function is called, the number of function evaluations is incremented, 
                the data loss value is recorded, and create the overlay plot. The status of the 
                parameters is not recorded.

            '''
            self.function_evals += 1

            # Plot experimet
            plt.figure()

            hr = min(self.data_container.heating_rates)
            data_dict = self.data_container.heating_data[hr]

            # Plot experimental data
            plt.plot(data_dict['Time'], 100*data_dict['O2'], 'b-', label=str(hr))


            try:   
                heating_data = {'Time': data_dict['Time'], 'Temp': data_dict['Temp']}
                IC = {'Temp': data_dict['Temp'][0], 'O2': data_dict['O2_con_in'], 'Oil': self.data_container.Oil_con_init}
                y_dict = self.kinetic_cell.get_rto_data(x, heating_data, IC) 
                plt.plot(y_dict['Time'], 100*y_dict['O2'], 'b--')

                O2_consump_sim = np.trapz(y_dict['O2'],x=y_dict['Time'])
                O2_consump_data = np.trapz(data_dict['O2'],x=data_dict['Time'])
                CO2_consump_sim = np.trapz(y_dict['CO2'],x=y_dict['Time'])
                CO2_consump_data = np.trapz(data_dict['CO2'],x=data_dict['Time'])

                effluence_loss = (O2_consump_sim - O2_consump_data)**2 + (CO2_consump_sim - CO2_consump_data)**2

            except:
                effluence_loss = 1e4

            plt.xlabel('Time')
            plt.ylabel(r'$O_2$ consumption [% mol]')
            plt.title(r'Warm Start $O_2$ consumption (Stage 2)')
            
            # Print figure with O2 consumption data
            plt.savefig(os.path.join(self.kinetic_cell.results_dir, 'figures', 'O2_overlay_{}.png'.format(self.function_evals)))
            plt.close()

            # Final loss value
            loss = effluence_loss

            # Save loss value
            self.loss_values.append(loss)

            return loss

        return cost_fun


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

