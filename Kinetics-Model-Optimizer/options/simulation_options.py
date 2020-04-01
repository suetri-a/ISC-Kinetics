import argparse
from .base_options import KineticCellOptions

class SimulationOptions(KineticCellOptions):


    def initialize(self, parser):
        parser = KineticCellOptions().initialize(parser)

        parser.set_defaults(name='simulation_example', isOptimization=False)

        # Simulation parameters
        parser.add_argument('--heating_rates', type=float, nargs='+', action='append', default=[5.0], help='heating rates from RTO experiments')
        parser.add_argument('--max_temp', type=float, default=750.0, help='maximum temperature during RTO experiment')
        parser.add_argument('--kinetics_model', type=str, default='stars', help='type of kinetics model to use [arrhenius | stars]')
        parser.add_argument('--load_rxn', type=eval, default=True, help='names of reactants')
        parser.add_argument('--phase', type=str, default='simulation', help='either simulation or optimization')
        parser.add_argument('--iso_resolution', type=int, default=100, help='number of time steps for isoconversional analysis interpolation')
        
        # Reaction arguments
        parser.add_argument('--reaction_model', type=str, default='Cinar', 
            help='pre-programmed reaction model to use [Cinar | CinarMod1 | CinarMod2 | CinarMod3 | Chen1 | Chen 2 | Dechelette1 | Dechelette2 | Crookston]')
        parser.add_argument('--reac_names', type=str, default=None, help='names of reactants')
        parser.add_argument('--prod_names', type=str, default=None, help='names of products')
        parser.add_argument('--rxn_constraints', type=str, default=None, help='constraints in the reaction')
        parser.add_argument('--init_coeff', type=str, default=None, help='initial coefficients in the reaction model')
        parser.add_argument('--O2_con_in', type=float, default=0.25, help='concentration of flow of oxygen into kinetic cell')
        parser.add_argument('--Tspan', type=float, nargs='+', action='append', default=[0.0, 500.0], help='time span for running simulation')
        parser.add_argument('--balances', type=str, nargs='+', action='append', default=['M', 'O'], help='balances to constrain optimization [M | O | C]')

        self.isOptimization = False

        return parser 