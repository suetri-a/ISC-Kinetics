
from .base_options import KineticCellOptions

class OptimizationOptions(KineticCellOptions):


    def initialize(self, parser):
        parser = KineticCellOptions().initialize(parser)

        parser.set_defaults(name='optimization_example', isOptimization=True)

        # Optimization options
        parser.add_argument('--autodiff_enable', type=eval, default=False, help='use autodiff in optimizer')
        parser.add_argument('--param_loss', type=str, default='uniform', help='prior for use in parameter inversion [uniform]')
        parser.add_argument('--output_loss', type=str, default='gaussian', help='data loss function for inversion [gaussian | exponential | ISO_peak | O2_peak]')
        parser.add_argument('--output_loss_inputs', type=str, default='[O2,CO2]', help='data to feed into the loss function depending on likelihood model')
        parser.add_argument('--balances', type=str, default='[M,O]', help='balances to constrain optimization [M | O | C]')

        # Reaction arguments
        parser.add_argument('--load_rxn', type=str, default=None, help='names of reactants')

        # Parameter constraints
        parser.add_argument('--pre_exp_lower', type=float, default=0.0, help='lower bound of log parameters for pre exponential factor')
        parser.add_argument('--pre_exp_upper', type=float, default=21.0, help='upper bound of log parameters for pre exponential factors')
        parser.add_argument('--act_eng_lower', type=float, default=0.0, help='lower bound of log parameters for activation energies')
        parser.add_argument('--act_eng_upper', type=float, default=21.0, help='upper bound of log parameters for activation energies')
        parser.add_argument('--rxn_order_lower', type=float, default=1e-2, help='lower bound of reaction orders')
        parser.add_argument('--rxn_order_upper', type=float, default=2.0, help='upper bound of reaction orders')
        parser.add_argument('--stoic_coeff_lower', type=float, default=1e-2, help='lower bound of stoichiometric coefficients')
        parser.add_argument('--stoic_coeff_upper', type=float, default=40.0, help='upper bound of stoichiometric coefficients')

        self.isOptimization = True

        return parser 