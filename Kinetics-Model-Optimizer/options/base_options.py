import argparse
import os
import json
import simulation
import optimization
import data

from utils import utils
from . import set_reaction_defaults, get_component_props, parse_rxn_info, parse_optimization_info

class KineticCellOptions():
    '''
    Container for default options for kinetic cell optimization.  

    '''

    def __init__(self):
        ''' Reset the class; indicates the class hasn't been initailized'''
        self.initialized = False


    def initialize(self, parser):
        ''' Define the common options used across all simulations and optimizations'''

        # Basic parameters
        parser.add_argument('--name', type=str, default='vkc_example', help='name of the numerical experiment')
        parser.add_argument('--results_dir', type=str, default='results', help='folder to contain results from the optimization')
        parser.add_argument('--isOptimization', type=eval, default=False, help='set if optimizing or simulating')
        parser.add_argument('--load_from_saved', action='store_true', help='load experiment with same name')

        # Kinetic cell parameters
        parser.add_argument('--R', type=float, default=8.314, help='universal gas constant')
        parser.add_argument('--T0', type=float, default=21.0, help='ambient temperature')
        parser.add_argument('--Ua', type=float, default=1.5e4, help='heat transfer coefficient')
        parser.add_argument('--oil_sat', type=float, default=0.04, help='initial oil saturation')
        parser.add_argument('--kc_V', type=float, default=2.895e-5, help='kinetic cell volume')
        parser.add_argument('--porosity', type=float, default=0.36, help='porosity of kinetic cell')
        parser.add_argument('--flow_rate', type=float, default=9.66667e-3, help='gas flow rate in kinetic cell')
        parser.add_argument('--O2_partial_pressure', type=float, default=6.89476e3, help='O2 partial pressure in kinetic cell')
        parser.add_argument('--material_dict', type=str, default=None, help='Dictionary of material properties')
        parser.add_argument('--comp_phase', type=str, default=None, help='phase of components')
        parser.add_argument('--fuel_comps', type=str, default=None, help='fuel components')
        parser.add_argument('--combustion_prods', type=str, default=None, help='components produced from combustion')
        parser.add_argument('--IC_dict', type=str, default=None, help='dictionary of initial conditions')

        # Reaction arguments
        parser.add_argument('--reac_names', type=str, default=None, help='names of reactants')
        parser.add_argument('--prod_names', type=str, default=None, help='names of products')
        parser.add_argument('--rxn_constraints', type=str, default=None, help='constraints in the reaction')
        parser.add_argument('--init_coeff', type=str, default=None, help='initial coefficients in the reaction model')
        parser.add_argument('--reaction_model', type=str, default=None, 
            help='pre-programmed reaction model to use [Cinar | CinarMod1 | CinarMod2 | CinarMod3 | Chen1 | Chen 2 | Dechelette1 | Dechelette2 | Crookston]')
        parser.add_argument('--kinetics_model', type=str, default='stars', help='type of kinetics model to use [arrhenius | stars]')

        # Other arguments
        parser.add_argument('--optimizer_type', type=str, default='quad_penalty_de', help='type of optimization to use [quad_penalty | aug_lagrangian]')
        parser.add_argument('--log_params', type=eval, default=True, help='use log of some parameters during optimization')
        parser.add_argument('--experiment_type', type=str, default='rto', help='type ot data to load [rto | ignition]')

        self.initialized = True

        return parser


    def gather_options(self):
        """
        Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # Get the basic options
        opt, _ = parser.parse_known_args()

        # Modify model-related parser options
        kinetic_cell_name = opt.kinetics_model
        kc_option_setter = simulation.get_option_setter(kinetic_cell_name)
        parser = kc_option_setter(parser)

        reaction_model_name = opt.reaction_model
        parser = set_reaction_defaults(parser, reaction_model_name)
        opt, _ = parser.parse_known_args()  # parse again with new defaults


        # Modify data- and optimizer-related parser options
        if self.isOptimization:
            data_option_setter = data.get_option_setter(opt.experiment_type)
            parser = data_option_setter(parser)
            optimizer_option_setter = optimization.get_option_setter(opt.optimizer_type)
            parser = optimizer_option_setter(parser)


        # Save parser
        self.parser = parser
        opts = parser.parse_args()

        # Load in reaction info if required
        opts = get_component_props(opts)

        # Parse the reaction info into form usable by optimizer/simulator modules
        opts = parse_rxn_info(opts)

        if self.isOptimization:
            opts = parse_optimization_info(opts)

        # Return final parser
        return opts


    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if default is None or v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # write options to text file
        expr_dir = os.path.join(opt.results_dir, opt.name)
        utils.mkdirs(expr_dir)
        type_of_exp = 'optimization' if self.isOptimization else 'simulation'
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(type_of_exp))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        # save options as json object for easy loading
        file_name = os.path.join(expr_dir, '{}_opt.json'.format(type_of_exp))
        with open(file_name, 'w') as fp:
            json.dump(opt.__dict__, fp)

    def parse(self):
        '''
        Parse input options, create checkpoints directory suffix.
        '''

        opts = self.gather_options()

        self.print_options(opts)
        self.opts = opts

        return self.opts