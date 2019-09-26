import autograd.numpy as np
from .base_optimizer import BaseOptimizer

class QuadPenaltyOptimizer(BaseOptimizer):

    def __init__(self, kinetic_cell, kc_ground_truth, opts):
        super().__init__(kinetic_cell, kc_ground_truth, opts)
        pass

    def objfun(self):

        def fun(x):
            cost = self.base_cost(x) + 0
        
        return fun
    
    def optimize_cell(self):
        l = 10
        p = self.kinetic_cell.params

        for _ in range(5):
            
            # Define cost function with new Lambda
            def cost(x):
                base_cost = self.base_fun(x) 
                res = self.kinetic_cell.residuals_from_params(x)
                penalty_cost = l*np.sum(res**2)
                return base_cost + penalty_cost

            if opts.autodiff_enable:
                jac = grad(cost)
            else:
                jac = None

            # result = sp.optimize.differential_evolution(cost, bnds)
            result = minimize(cost, p, jac=jac, tol=1e-3, bounds=self.bnds)
            p = result.x
            l*=10

        self.sol = p