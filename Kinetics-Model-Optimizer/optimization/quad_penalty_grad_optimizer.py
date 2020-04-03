import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize, differential_evolution
from .base_optimizer import BaseOptimizer

class QuadPenaltyGradOptimizer(BaseOptimizer):

    def __init__(self, kinetic_cell, kc_ground_truth, opts):
        super().__init__(kinetic_cell, kc_ground_truth, opts)
        pass

    def get_objective_fun(self, penalty_fac):

        def fun(x):
            base_cost = self.base_cost(x)
            res = self.kinetic_cell.compute_residuals(x)
            cost = base_cost + penalty_fac*np.sum(np.power(res,2))
            return cost
        
        return fun

    
    def optimize_cell(self):
        l = 1
        x = self.kinetic_cell.params

        for _ in range(5):
            
            # Define cost function with new Lambda
            cost = self.get_objective_fun(l)

            if self.autodiff_enable:
                jac = grad(cost)
            else:
                jac = None

            # result = sp.optimize.differential_evolution(cost, bnds)
            result = minimize(cost, x, jac=jac, callback=self.callbackfun)
            x = result.x
            l*=10

        self.sol = x