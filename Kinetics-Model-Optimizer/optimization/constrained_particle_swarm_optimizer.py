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
            print('Warm start loaded!')

        else:
            def res_fun(x): return np.sum(np.power(self.kinetic_cell.compute_residuals(x),2))
            x0 = self.data_container.compute_initial_guess(self.kinetic_cell.reac_names, self.kinetic_cell.prod_names,
                                                            res_fun, self.kinetic_cell.param_types)
            x0 = self.warm_start(x0, self.kinetic_cell.param_types)
            np.save(os.path.join(self.load_dir,'warm_start.npy'), x0)
            np.save(os.path.join(self.load_dir, 'total_loss.npy'), np.array(self.loss_values))
            np.save(os.path.join(self.load_dir,'function_evals.npy'), self.function_evals)
            
            self.warm_start_complete = True
            with open(os.path.join(self.load_dir,'warm_start_complete.pkl'),'wb') as fp:
                pickle.dump(self.warm_start_complete, fp)


        # Optimize parameters
        if self.optim_complete:
            self.sol = np.load(os.path.join(self.load_dir,'optimal_params.npy'))
            print('Optimized parameters loaded!')
        
        else:
            cost_warm_start = self.cost_fun(x0)
            with open(self.log_file,'a+') as fileID:
                print('Warm start completed. Cost: {}'.format(str(cost_warm_start)), file=fileID)

            # Form constraint
            def constraint_fun1(x):
                return np.sum(np.power(self.kinetic_cell.compute_residuals(x),2))

            def constraint_fun2(x):
                return -1*np.sum(np.power(self.kinetic_cell.compute_residuals(x),2))
                        
            # Run optimization
            lb, ub = [], []
            for i, p in enumerate(self.kinetic_cell.param_types):
                if p[0]=='acteng':
                    lb.append(x0[i]-1.0)
                    ub.append(x0[i]+1.0)
                if p[0]=='preexp':
                    lb.append(x0[i]-1.0)
                    ub.append(x0[i]+1.0)
                if p[0]=='stoic':
                    lb.append(np.maximum(x0[i]-4.0,1e-2))
                    ub.append(np.minimum(x0[i]+4.0,40.0))
            
            self.sol, _ = pso(self.cost_fun, lb, ub, swarmsize=50, maxiter=50, phip=0.5, phig=0.75, 
                                ieqcons=[constraint_fun1, constraint_fun2])
            
            cost_final = self.cost_fun(self.sol)
            with open(self.log_file, 'a+') as fileID:
                print('Optimization completed. Final cost: {}.'.format(str(cost_final)), file=fileID)
            
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