import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from .base_optimizer import BaseOptimizer


# Wrapper function for using joblib with optimizers and UQ frameworks
def joblib_map(func, iter):
    return Parallel(n_jobs=-1)(delayed(func)(x) for x in iter)


class DifferentialEvolutionOptimizer(BaseOptimizer):

    def __init__(self, kinetic_cell, kc_ground_truth, opts):
        super().__init__(kinetic_cell, kc_ground_truth, opts)
        self.constraint_loss_values = []
        
    def cost_fun(self, x):
        if np.any(np.greater(np.abs(self.kinetic_cell.compute_residuals(x)), 1e-3)):
            cost = 1e6
        else:
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
            with open(os.path.join(self.load_dir, 'warm_start_bounds.pkl'), 'rb') as fp:
                bnds = pickle.load(fp)
            print('Warm start loaded!')

        else:
            x0, bnds = self.warm_start(None, self.kinetic_cell.param_types)
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

            popsize = x0.shape[0]
            maxiter = 100
            init = np.expand_dims(x0, 0) + 0.01*np.random.randn(popsize, x0.shape[0])
            
            # Run optimization
            result = differential_evolution(self.cost_fun, bnds, 
                                            popsize=popsize, 
                                            init=init, 
                                            polish=True, 
                                            maxiter=maxiter, 
                                            workers=joblib_map)
            self.sol = result.x
            
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

        # Print output message
        print('Final optimized reaction:')
        self.kinetic_cell.print_reaction(self.sol)