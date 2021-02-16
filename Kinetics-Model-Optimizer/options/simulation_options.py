import argparse, os
from .base_options import KineticCellOptions

class SimulationOptions(KineticCellOptions):


    def initialize(self, parser):
        parser = KineticCellOptions().initialize(parser)

        parser.set_defaults(name='simulation_example', isOptimization=False)

        # Simulation parameters
        parser.add_argument('--heating_rates', type=str, default='[5.0]', help='list of heating rates passed in as single string')
        parser.add_argument('--max_temp', type=float, default=750.0, help='maximum temperature during RTO experiment')
        parser.add_argument('--load_rxn', type=eval, default=True, help='names of reactants')
        parser.add_argument('--phase', type=str, default='simulation', help='either simulation or optimization')
        parser.add_argument('--iso_resolution', type=int, default=100, help='number of time steps for isoconversional analysis interpolation')
        
        # Reaction arguments
        parser.add_argument('--O2_con_in', type=float, default=0.21, help='concentration of flow of oxygen into kinetic cell')
        parser.add_argument('--Tspan', type=str, default='[0.0,300.0]', help='time span for running simulation')
        parser.add_argument('--balances', type=str, nargs='+', action='append', default=['M', 'O'], help='balances to constrain optimization [M | O | C]')

        parser.add_argument('--oil_sample', type=str, default='chichimene', help='type of oil for which to load the oil_data.json file')

        opts, _ = parser.parse_known_args()
        parser.add_argument('--data_dir', type=str, default='datasets', help='folder to store data for oil samples')
        parser.add_argument('--dataset', type=str, default=os.path.join('OilSamples', opts.oil_sample), help='helper argument to load oil properties')

        self.isOptimization = False

        return parser 