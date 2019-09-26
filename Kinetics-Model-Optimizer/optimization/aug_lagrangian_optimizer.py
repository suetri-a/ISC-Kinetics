import autograd.numpy as np
from .base_optimizer import BaseOptimizer

class AugLagrangianOptimizer(BaseOptimizer):

    def __init__(self, kinetic_cell, kc_ground_truth, opts):
        super().__init__(kinetic_cell, kc_ground_truth, opts)
        pass
    
    def optimize_cell(self):
        jac = self.get_gradient(self.opts.autodiff_enable)
        sol = minimize(self.objfun(), self.kinetic_cell.params, jac=jac, bounds=self.bnds)
        self.sol = sol.x

        p = self.kinetic_cell.params
        l = np.ones((self.kinetic_cell.reaction.num_rxns))
        mu = 10
        omega = 1 / mu
        eta = 1 / mu**0.1

        def proj_fun(x): return np.maximum(np.minimum(x, ub), lb)
        
        for _ in range(10):

            def cost(x): 
                data_cost = self.base_cost(x)
                res = self.kinetic_cell.residuals_from_params(x)
                cost_out = data_cost - np.dot(l, res) + mu/2*np.sum(res**2)
                return cost_out

            if self.opts.autodiff_enable:
                jac = grad(cost)
            else:
                jac = None
            
            result = minimize(cost, p, jac=jac, bounds=bnds)
            p = result.x
            
            residuals = self.kinetic_cell.residuals_from_params(p)

            if np.linalg.norm(residuals) < eta:
                if np.linalg.norm(p - proj_fun(p - result.jac)) < omega:
                    break
                l -= mu*residuals
                mu = mu
                eta = eta / mu**0.9
                omega = omega / mu
            else:
                l = l
                mu *= 1e2
                eta = 1 / mu**0.1
                omega = 1 / mu

        self.sol = p