from .base_optimizer import BaseOptimizer


class DiffEvolutionOptimizer(BaseOptimizer):

    def __init__(self, kinetic_cell, kc_ground_truth, opts):
        super().__init__(kinetic_cell, kc_ground_truth, opts)
        pass

    def objfun(self):
        return self.base_cost

    def optimize_cell(self):
        fun = self.objfun()
        jac = self.get_gradient(self.opts.autodiff_enable)
        sol = minimize(fun, self.kinetic_cell.params, jac=jac, bounds=self.bnds)
        self.sol = sol.x