import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from .base_optimizer import BaseOptimizer


class QuadPenaltySLSQPOptimizer(BaseOptimizer):

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
        l = 1.0
        bnds = self.kinetic_cell.get_bounds()
        def res_fun(x): return np.sum(np.power(self.kinetic_cell.compute_residuals(x),2))
        x0 = self.data_container.compute_initial_guess(self.kinetic_cell.reac_names, self.kinetic_cell.prod_names,
                                                        res_fun, self.kinetic_cell.param_types)
        x0 = self.warm_start_act_engs(x0, self.kinetic_cell.param_types)

        opt_start_ind = self.function_evals

        for i in range(5):
            
            # Define cost function with new Lambda
            cost = self.get_objective_fun(l)

            # Run optimization
            result = minimize(cost, x0, bounds=bnds, method='SLSQP')
            
            if i < 4:
                # Perform updates
                x0 = result.x
                l*=10

        self.sol = result.x


        # Print convergence plot
        plt.figure()
        plt.semilogy(np.array(self.loss_values[opt_start_ind:]) + np.array(self.constraint_loss_values))
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Total loss value')
        plt.title('Optimization Convergence Plot - Total Loss')
        plt.savefig(os.path.join(self.kinetic_cell.results_dir, 'convergence_plot_total.png'))

        # Print convergence plot
        plt.figure()
        plt.semilogy(self.loss_values[:opt_start_ind])
        plt.semilogy(self.loss_values[opt_start_ind:])
        plt.legend(['Warm Start', 'Optimization'])
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

        # Save data
        np.save(os.path.join(self.kinetic_cell.results_dir, 'total_loss.npy'), np.array(self.loss_values) + np.array(self.constraint_loss_values))
        np.save(os.path.join(self.kinetic_cell.results_dir, 'constraint_loss.npy'), np.array(self.constraint_loss_values))
        np.save(os.path.join(self.kinetic_cell.results_dir, 'data_loss.npy'), np.array(self.loss_values))
        np.save(os.path.join(self.kinetic_cell.results_dir, 'optimal_params.npy'), self.sol)

        # self.compute_likelihood_intervals(self.sol)