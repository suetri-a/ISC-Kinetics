import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
from .base_optimizer import BaseOptimizer


class ConstrainedParticleSwarmOptimizer(BaseOptimizer):

    def __init__(self, kinetic_cell, kc_ground_truth, opts):
        super().__init__(kinetic_cell, kc_ground_truth, opts)
        self.constraint_loss_values = []
        
    def cost_fun(self, x):
        
        cost = self.base_cost(x) # base cost set from options

        # Log optimization status
        with open(self.log_file, 'a+') as fileID:
            print('==================================================== Status at Iteration {} ===================================================='.format(str(self.function_evals)), file=fileID)
            self.kinetic_cell.log_status(x, fileID)
            print('Loss value: {}'.format(cost), file=fileID)
            print('================================================== End Status at Iteration {} ==================================================\n\n'.format(str(self.function_evals)), file=fileID)

        return cost

    
    def optimize_cell(self):

        # Warm start initial guess
        if self.warm_start_complete:
            x0 = np.load(os.path.join(self.load_dir,'warm_start.npy'))
            with open(os.path.join(self.load_dir, 'warn_start_bounds.pkl'), 'rb') as fp:
                bnds = pickle.load(fp)
            print('Warm start loaded!')

        else:
            def res_fun(x): return np.sum(np.power(self.kinetic_cell.compute_residuals(x),2))
            x0 = self.data_container.compute_initial_guess(self.kinetic_cell.reac_names, self.kinetic_cell.prod_names,
                                                            res_fun, self.kinetic_cell.param_types)
            x0, bnds = self.warm_start(x0, self.kinetic_cell.param_types)
            np.save(os.path.join(self.load_dir,'warm_start.npy'), x0)
            np.save(os.path.join(self.load_dir, 'total_loss.npy'), np.array(self.loss_values))
            np.save(os.path.join(self.load_dir,'function_evals.npy'), self.function_evals)
            
            self.warm_start_complete = True
            with open(os.path.join(self.load_dir, 'warm_start_complete.pkl'),'wb') as fp:
                pickle.dump(self.warm_start_complete, fp)
            
            with open(os.path.join(self.load_dir, 'warm_start_bounds.pkl'),'wb') as fp:
                pickle.dump(bnds, fp)


        # Optimize parameters
        if self.optim_complete:
            self.sol = np.load(os.path.join(self.load_dir,'optimal_params.npy'))
            print('Optimized parameters loaded!')
        
        else:
            cost_warm_start = self.cost_fun(x0)
            with open(self.log_file,'a+') as fileID:
                print('Warm start completed. Cost: {}'.format(str(cost_warm_start)), file=fileID)

            # Form constraint
            fuel_names_all = self.kinetic_cell.fuel_names + self.kinetic_cell.pseudo_fuel_comps
            fuel_dict = {}
            for i in range(self.kinetic_cell.num_rxns):
                for s in self.kinetic_cell.reac_names[i]:
                    if s in fuel_names_all:
                        if s not in fuel_dict.keys():
                            fuel_dict[s] = []
                        fuel_dict[s].append(i)

            A = np.zeros((1,self.kinetic_cell.num_rxns))
            for i in range(self.kinetic_cell.num_rxns):
                a_reac = np.zeros((1,self.kinetic_cell.num_rxns))
                a_reac[0,i] = -1

                for s in self.kinetic_cell.prod_names[i]:
                    if s in fuel_names_all:
                        for i in fuel_dict[s]:
                            a_prod = np.copy(a_reac)
                            a_prod[0,i] = 1
                            A = np.concatenate((A,a_prod))
            A = A[1:,:]
            acteng_inds = [i for i,p in enumerate(self.kinetic_cell.param_types) if p[0]=='acteng']

            def constraint_fun(x):
                out1 = (np.array(self.kinetic_cell.compute_residuals(x))).flatten()
                out2 = -1*np.copy(out1)
                out3 = np.squeeze(np.dot(A, x[acteng_inds]))
                out = np.concatenate((out1, out2, out3))
                return out  
                        
            # Run optimization
            lb, ub = [l for l,u in bnds], [ u for l,u in bnds ]
            lb = np.minimum(lb, ub)
            ub = np.maximum(lb, ub)
            self.sol, _ = pso(self.cost_fun, lb, ub, swarmsize=20, maxiter=50, phip=0.25, phig=0.75, 
                                f_ieqcons=constraint_fun)
            
            cost_final = self.base_cost(self.sol, save_filename=os.path.join(self.figs_dir, 'final_results', 'final_O2_overlay.png'))
            with open(self.log_file, 'a+') as fileID:
                print('Optimization completed. Final cost: {}.'.format(str(cost_final)), file=fileID)

            with open(self.report_file, 'a+') as fileID:
                print('================================== Optimization Logging ==================================\n\n', file=fileID)
                self.kinetic_cell.log_status(self.sol, fileID)
                print('Final cost: {}'.format(str(cost_final)), file=fileID)
            
            # Save data
            np.save(os.path.join(self.load_dir, 'total_loss.npy'), np.array(self.loss_values))
            np.save(os.path.join(self.load_dir, 'optimal_params.npy'), self.sol)
            np.save(os.path.join(self.load_dir, 'function_evals.npy'), self.function_evals)

            self.optim_complete = True
            with open(os.path.join(self.load_dir,'optim_complete.pkl'),'wb') as fp:
                pickle.dump(self.optim_complete, fp)


        # Print convergence plot
        plt.figure()
        plt.semilogy(np.array(self.loss_values))
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Total Loss value')
        plt.title('Optimization Convergence Plot')
        plt.savefig(os.path.join(self.kinetic_cell.results_dir, 'convergence_plot_total.png'))