import argparse
import os
import kinetics
import optim

from .base_options import KineticCellOptions

class OptimizationOptions(KineticCellOptions):


    def initialize(self, parser):
        parser = KineticCellOptions().initialize(parser)

        parser.set_defaults(name='optimization_example')

        # Optimization options
        parser.add_argument('--log_params', type=eval, default=True, help='use log of some parameters during optimization')
        parser.add_argument('--optimizer_type', type=str, default='quad_penalty', help='type of optimization to use [quad_penalty | aug_lagrangian]')
        parser.add_argument('--autodiff_enable', type=eval, default=False, help='use autodiff in optimizer')
        parser.add_argument('--param_prior', type=str, default='uniform', help='prior for use in parameter inversion [uniform]')
        parser.add_argument('--output_prior', type=str, default='gaussian', help='data prior for inversion [gaussian | exponential | ISO_peak | O2_peak]')
        parser.add_argument('--balances', type=str, nargs='+', action='append', default=['M', 'O'], help='balances to constrain optimization [M | O | C]')

        # Reaction arguments
        parser.add_argument('--reaction_model', type=str, default='Chen2', 
            help='pre-programmed reaction model to use [Cinar | CinarMod1 | CinarMod2 | CinarMod3 | Chen1 | Chen 2 | Dechelette1 | Dechelette2 | Crookston]')
        parser.add_argument('--reac_names', type=str, default=None, help='names of reactants')
        parser.add_argument('--prod_names', type=str, default=None, help='names of products')
        parser.add_argument('--rxn_constraints', type=str, default=None, help='constraints in the reaction')
        parser.add_argument('--init_coeff', type=str, default=None, help='initial coefficients in the reaction model')
        parser.add_argument('--Tspan', type=float, nargs='+', action='append', default=[0.0, 600.0], help='time span for running simulation')

        # Parameter constraints
        parser.add_argument('--pre_exp_lower', type=float, default=0.0, help='lower bound of log parameters for pre exponential factor')
        parser.add_argument('--pre_exp_upper', type=float, default=21.0, help='upper bound of log parameters for pre exponential factors')
        parser.add_argument('--act_eng_lower', type=float, default=0.0, help='lower bound of log parameters for activation energies')
        parser.add_argument('--act_eng_upper', type=float, default=21.0, help='upper bound of log parameters for activation energies')
        parser.add_argument('--rxn_order_lower', type=float, default=1e-2, help='lower bound of reaction orders')
        parser.add_argument('--rxn_order_upper', type=float, default=2.0, help='upper bound of reaction orders')
        parser.add_argument('--stoic_coeff_lower', type=float, default=1e-2, help='lower bound of stoichiometric coefficients')
        parser.add_argument('--stoic_coeff_upper', type=float, default=30.0, help='upper bound of stoichiometric coefficients')

        self.isOptimization = True

        return parser 