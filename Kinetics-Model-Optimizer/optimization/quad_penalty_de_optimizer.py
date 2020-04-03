import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from .base_optimizer import BaseOptimizer


class QuadPenaltyDEOptimizer(BaseOptimizer):

    def __init__(self, kinetic_cell, kc_ground_truth, opts):
        super().__init__(kinetic_cell, kc_ground_truth, opts)
        self.constraint_loss_values = []
        

    def get_objective_fun(self, penalty_fac):

        def fun(x):
            
            base_cost = self.base_cost(x) # base cost set from options

            res = self.kinetic_cell.compute_residuals(x) # get residuals from parameter vector
            constraint_cost = penalty_fac*np.sum(np.power(res,2)) # cost from quadratic penalty methods
            self.constraint_loss_values.append(constraint_cost)
            
            # Comput total cost
            cost = base_cost + constraint_cost

            # Log optimization status
            with open(self.log_file, 'a+') as fileID:
                print('==================================================== Status at Iteration {} ===================================================='.format(str(self.function_evals)), file=fileID)
                self.kinetic_cell.log_status(x, fileID)
                print('Loss value: {} ({} base cost, {} constraint cost)'.format(cost, base_cost, constraint_cost), file=fileID)
                print('================================================== End Status at Iteration {} ==================================================\n\n'.format(str(self.function_evals)), file=fileID)

            

            return cost
        
        return fun

    
    def optimize_cell(self):
        l = 1e-3
        bnds = self.kinetic_cell.get_bounds()
        def res_fun(x): return np.sum(np.power(self.kinetic_cell.compute_residuals(x),2))
        x0 = self.data_container.compute_initial_guess(self.kinetic_cell.reac_names, self.kinetic_cell.prod_names,
                                                        self.kinetic_cell.comp_names, res_fun, self.kinetic_cell.param_types)
        popsizes = [5, 2, 1, 1, 1]
        init_pop = np.tile(x0,(1,10)) + 0.1*np.random.randn(popsizes[0]*x0.shape[0], x0.shape[0])

        for i in range(5):
            
            # Define cost function with new Lambda
            cost = self.get_objective_fun(l)

            # Run optimization
            maxiter = int(400/len(bnds)/popsizes[i])
            result = differential_evolution(cost, bnds, init=init_pop, popsize=popsizes[i], maxiter=maxiter, polish=False)
            x = result.x
            
            # Perform updates
            l*=10
            init_pop = np.tile(x,(1,10)) + 0.1*np.random.randn(popsizes[i+1]*x.shape[0], x.shape[0])

        self.sol = x

        # Print convergence plot
        plt.figure()
        plt.semilogy(np.array(self.loss_values) + np.array(self.constraint_loss_values))
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Total loss value')
        plt.title('Optimization Convergence Plot - Total Loss')
        plt.savefig(os.path.join(self.kinetic_cell.results_dir, 'convergence_plot_total.png'))

        # Print convergence plot
        plt.figure()
        plt.semilogy(self.loss_values)
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Data loss value')
        plt.title('Optimization Convergence Plot - Data')
        plt.savefig(os.path.join(self.kinetic_cell.results_dir, 'convergence_plot_base.png'))

        # Print convergence plot
        plt.figure()
        plt.semilogy(self.constraint_loss_values)
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Constraint loss value')
        plt.title('Optimization Convergence Plot - Constraint Loss')
        plt.savefig(os.path.join(self.kinetic_cell.results_dir, 'convergence_plot_constraint.png'))