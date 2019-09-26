import numpy as np

'''
This file contains a holding constants and default parameters needed to run the kinetic cell 
    simulations and optimizations. We place these classes here to avoid any circular importing
    in the other files. When running the model fitting, this file will hold the default
    options for the command line arguments. The KineticCell class will import these as a 
    default when it is not being used with model fitting.

Note: We place these options in a file separate from the KineticCell class so it is easier to edit
    the defaults used in a set of numerical experiments. (Someone wishing to run this code 
    should not need to touch the other files for the code to work.)

'''

class KineticCellOptions:
    '''

    '''

    def __init__(self, data_type = 'synthetic', output_prior = 'gaussian', param_prior = 'uniform', 
        kinetics_model_type = 'Arrhenius',
        optimizer_type = 'penalty', optim_write_file = 'kc_opt_file.p', reaction_type = 'Chen2', max_temp = 750, 
        rate_heat = [1.54, 1.74, 1.92, 3.3]):
        ### CHEMICAL REACTION PARAMETERS ###
        ### Do NOT change these unless you wish to change the default parameters for all experiments ###
        self.R = 8.314 # Gas constant
        self.T0 = 293.15 # Ambient temperature

        # Kinetic cell parameters (from Bo's original model)
        self.Ua = 1.5e4 # overall heat transfer coefficient coeff times heat-exchange area [J/(m^3*s*K)] 
        # self.V = 4e-5 # void volume [m^3]
        # self.q = 2 / 60 # volume flow rate [L/sec]
        # O2_con_sim = 0.2094 # O2 concentration in the simulation
        # self.P = 15*6.89476e3 # O2 partial pressure 
        self.oil_sat = 0.04 # initial oil saturation

        # Actual kinetic cell parameters
        kc_V = 2.896e-5
        porosity = 0.36
        self.V = porosity * kc_V
        self.q = 9.66667e-3 # [580 cm^3/min]
        self.P = 6.89476e3 # O2 partial pressure 
        O2_con_sim = 0.2070

        self.O2_con_sim = O2_con_sim*self.P / self.R / self.T0
        
        # Chemical parameters and material properties
        # Note: we store these as dictionaries so we can use the species name as 
        #   key when creating the kinetic cell object.

        # Phase of each species:
        #   1 - gas, 2 - liquid, 3 - gas, 4 - solid
        self.comp_phase = {'H2O': 1,  'N2': 3, 'O2': 3, 'CO2': 3, 'CO': 3, 'Gas': 3, 
            'OIL-H': 2, 'OIL-L': 2, 
            'OIL': 2, 'Coke1': 4, 'Coke2': 4, 'Ci': 4}

        self.fuel_comps = ['OIL-H', 'OIL-L', 'OIL', 'Coke1', 'Coke2', 'Ci']
        self.combustion_prods = ['H2O', 'CO', 'CO2', 'Gas']

        # Store material properties in a dictionary of dictionaries
        self.material_dict = {} 
        
        # Molecular weights
        self.material_dict['M'] = {'H2O': 18, 'N2': 28, 'O2': 32, 'CO2': 44, 'CO': 28, 'Gas': 28.04, # Product species
            'OIL-H': 800, 'OIL-L': 400, # Parent oil species
            'OIL': np.nan, 'Coke1': np.nan, 'Coke2': np.nan, 'Ci': np.nan # Pseudocomponents
            } 

        # Oxygen counts
        self.material_dict['O'] = {'H2O': 1, 'N2': 0, 'O2': 2, 'CO2': 2, 'CO': 1, 'Gas': 1, 
            'OIL-H': 0, 'OIL-L': 0, 
            'OIL': np.nan, 'Coke1': np.nan, 'Coke2': np.nan, 'Ci': np.nan} 
        
        # Carbon counts
        self.material_dict['C'] = {'H2O': 0, 'N2': 0, 'O2': 2, 'CO2': 1, 'CO': 1, 'Gas': 1, 
            'OIL-H': 50, 'OIL-L': 30,
            'OIL': np.nan, 'Coke1': np.nan, 'Coke2': np.nan, 'Ci': np.nan} 

        # Heat capacity of each species [J/mole]
        self.material_dict['Cp'] = {'N2': 29.93161, 'O2': 30.47713, 'H2O': 34.49885, 'CO2': 21.08550, 
            'CO': 30.08316, 'Gas': 30.99, 
            'OIL-H': 225.658, 'OIL-L': 225.658,
            'OIL': 125.658, 'Coke1': 8.3908475, 'Coke2': 6.96015, 'Ci': 7.192155} 


        ### EXPERIMENTAL PARAMETERS ###
        ### These are initialized here and used as defaults for the command line arguments ###
        ### Note: these are the defaults used if running the KineticCell object in a Jupyter Notebook ###

        # Optimization options
        self.autodiff_enable = False
        self.data_type = data_type # 'synthetic' or 'experimental'
        self.param_prior = param_prior  # 'uniform', 'gaussian', 'beta'
        self.output_prior = output_prior  # 'gaussian', 'ISO_peak', or 'O2_peak'

        self.output_gaussian_weights = None

        self.param_gaussian_mu = None
        self.param_gaussian_sigma = None

        self.param_beta_alpha = None
        self.param_beta_beta = None
                                              
        self.peak_weight_vec = [75, 75, 250, 150]    # Objective hyperparameters 
                                                # [Peak Locations, Peak Values, O2 Start, O2 End]

        self.balances = ['M', 'O'] # Optionally include 'H' and 'C' for hydrogen and carbon

        self.optimizer_type = optimizer_type # 'evolutionary' or 'minimize' (BFGS, L-BFGS-B, or SLSQP)
        self.optim_write_file = optim_write_file
        self.optim_verbose = True
        self.optim_write_output = True
        self.optim_use_barrier_fun = False
        self.optim_res_tol = 1e-3
        self.log_params = True
        
        # Simulation parameters
        self.solver_type = 'BDF'       # type of numerical solver used with scipy solve_ivp
        self.num_sim_steps = 250       # [min]
        self.resol = 100               # Isoconversional Resolution

        self.rate_heat = rate_heat # Heating rates (C/min) Bo Chen rates (testing)
        self.max_temp = max_temp            # [C], maximum temperature value for the temperature
        self.exp_real = [1, 2, 3, 4, 5, 6]   # Combination Consistent Set
        self.num_peaks = [2, 2, 2, 2, 2, 2]  # Number of peaks at each heating rate
        self.reac_tol = 1e-4  # Tolerance for beginning and end of reaction

        self.kinetics_model_type = kinetics_model_type
        if kinetics_model_type == 'experimental':
            self.reaction_type = 'experimental'
        else:
            self.reaction_type = reaction_type       # Reaction type
            default_reaction_types = ['Cinar','CinarMod1','CinarMod2','CinarMod3','Chen1','Chen2',
                'Dechelette1', 'Dechelette2','Crookston']
            if self.reaction_type in default_reaction_types:
                self.get_predefined_rxn(self.reaction_type)

        self.IC_dict = {'H2O': 0, 'N2': 0, 'O2': None, 'CO2': 0, 'CO': 0, 'Gas': 0, 
                    'OIL-H': 2e4, 'OIL-L': 2e4,  # Dictionary of initial condition values
                    'OIL': 0, 'Coke1': 0, 'Coke2': 0, 'Ci': 0}
        self.params = None  # Parameter vector
        self.param_types = None # Type of each parameter

        self.IC = None  # Initialize to None to store IC later based on reaction scheme
        self.Tspan = [0, 600]  # Time span for the reactions (min)
        self.max_temp = 750 # In C

        self.data_file = None

        # STARS Simulation Parameters
        self.stars_base_file = 'BASE_CASE.dat'
        self.stars_sim_folder = './KCRunFiles/'


    def get_predefined_rxn(self, reaction_type):

        if reaction_type =='Cinar':
            '''
            Reaction scheme:
            (1)       OIL-H + O2 -> COKE1       + H2O
            (2)       COKE1 + O2 -> CO    + CO2 + H2O
            (3)       COKE1      -> COKE2
            (4)       COKE2 + O2 -> CO    + CO2 + H2O
            (5)       COKE1      -> CI
            (6)       CI    + O2 -> CO    + CO2

            '''
            
            self.reac_names = [['OIL-H','O2'], ['Coke1', 'O2'], ['Coke1'], ['Coke2', 'O2'], ['Coke1'], ['Ci', 'O2']] 
            self.prod_names = [['Coke1', 'H2O'], ['CO', 'CO2', 'H2O'], ['Coke2'], ['CO', 'CO2', 'H2O'], ['Ci'], ['CO', 'CO2']] 
            
            self.heat_reaction = np.array([0, -1e4, 0, -2e4, 0, -5e4])
            self.pre_exp_factors = np.array([1e1, 1e1, 1e-1, 1e-1, 1e-3, 1e-3])
            self.act_energies = np.array([1e5, 5e4, 6e4, 8e4, 8e4, 1e5])

            self.rxn_constraints = [[2, 'O2', 'CO2', 0.5], [2, 'CO2', 'CO', 0.4201], 
                [4, 'O2', 'CO2', 1/1.15], [4, 'CO2', 'CO', 0.3294]]
            self.init_coeff = [[1, 'O2',30], [1, 'Coke1', 10], [1, 'H2O', 2],
                [2, 'O2', 5], [2, 'CO', 0.3181], [2, 'CO2', 0.75], [2, 'H2O', 1.3815], 
                [3, 'Coke2', 1.3824], 
                [4, 'O2', 0.65], [4, 'CO', 0.1862], [4, 'CO2', 0.5652], [4, 'H2O', 0.2393],
                [5, 'Ci', 0.9135],
                [6, 'O2', 0.9], [6, 'CO', 0.2]]
        

        elif reaction_type == 'CinarMod1':
            '''
            Reaction scheme:
            (1)       OIL-H      -> COKE1 + CI
            (2)       COKE1 + O2 -> CO    + CO2 + H2O
            (3)       OIL-H      -> COKE2
            (4)       COKE2 + O2 -> CO    + CO2 + H2O
            (5)       CI    + O2 -> CO    + CO2 + H2O

            '''
            
            self.reac_names = [['OIL-H'], ['Coke1', 'O2'], ['OIL-H'], ['Coke2', 'O2'], ['Ci', 'O2']] 
            self.prod_names = [['Coke1', 'Ci'], ['CO', 'CO2', 'H2O'], ['Coke2'], ['CO', 'CO2', 'H2O'], ['CO', 'CO2', 'H2O']] 
            
            self.heat_reaction = np.array([0, -5e5, 0, -4e5, -4e5])
            self.pre_exp_factors = np.array([4.231e13, 3.265e8, 9.334e6, 5.621e9, 1.621e10])
            self.act_energies = np.array([8.62e4,  9.29e4,  8.0e4, 1.21e5, 1e5])

            self.rxn_constraints = [[2, 'O2', 'CO2', 1/1.15], [2, 'CO2', 'CO', 0.77], 
                [4, 'O2', 'CO2', 1/1.3], [4, 'CO2', 'CO', 0.41], 
                [5, 'O2', 'CO2', 1/1.5], [5, 'CO2', 'CO', 0.77]]
            self.init_coeff = [[1, 'Coke1', 5], [1, 'Ci', 5],
                [2, 'O2', 7.5], [2, 'CO', 0.5], [2, 'H2O', 2], 
                [3, 'Coke2', 5],
                [4, 'O2', 1.1], [4, 'CO', 0.4], [4, 'H2O', 2], 
                [5, 'O2', 7.5], [5, 'CO', 0.5], [5, 'H2O', 2]]
        

        elif reaction_type == 'CinarMod2':
            '''
            Reaction scheme:
            (1)       OIL-H      -> COKE1
            (2)       COKE1 + O2 -> CO    + CO2 + H2O
            (3)       OIL-H      -> COKE2
            (4)       COKE2 + O2 -> CO    + CO2 + H2O

            '''

            self.reac_names = [['OIL-H'], ['Coke1', 'O2'], ['OIL-H'], ['Coke2', 'O2']] 
            self.prod_names = [['Coke1'], ['CO', 'CO2', 'H2O'], ['Coke2'], ['CO', 'CO2', 'H2O']] 
            
            self.heat_reaction = np.array([0, -5e5, 0, -4e5])
            self.pre_exp_factors = np.array([4.231e13, 3.265e8, 9.334e6, 5.621e9])
            self.act_energies = np.array([8.62e4, 9.29e4, 8.0e4, 1.21e5])

            self.rxn_constraints = [[2, 'O2', 'CO2', 1/1.15], [2, 'CO2', 'CO', 0.77], 
                [4, 'O2', 'CO2', 1/1.3], [4, 'CO2', 'CO', 0.41]]
            self.init_coeff = [[1, 'Coke1', 7], 
                [2, 'O2', 7.5], [2, 'CO', 0.5], [2, 'H2O', 5], 
                [3, 'Coke2', 19], 
                [4, 'O2', 1.2], [4, 'CO', 0.4], [4, 'H2O', 5]]
        

        elif reaction_type == 'CinarMod3':
            '''
            (1)       OIL-H      -> COKE1
            (2)       COKE1 + O2 -> CO    + CO2 + H2O
            (3)       COKE1      -> COKE2
            (4)       COKE2 + O2 -> CO    + CO2 + H2O

            '''

            self.reac_names = [['OIL-H'], ['Coke1', 'O2'], ['Coke1'], ['Coke2', 'O2']] 
            self.prod_names = [['Coke1'], ['CO', 'CO2', 'H2O'], ['Coke2'], ['CO', 'CO2', 'H2O']] 
            
            self.heat_reaction = np.array([0, -5e5, 0, -4e5])
            self.pre_exp_factors = np.array([4.231e13, 3.265e8, 9.334e6, 5.621e9])
            self.act_energies = np.array([8.62e4, 9.29e4, 8.0e4, 1.21e5])

            self.rxn_constraints = [[2, 'O2', 'CO2', 1/1.15], [2, 'CO2', 'CO', 0.77], 
                [4, 'O2', 'CO2', 1/1.3], [4, 'CO2', 'CO', 0.41]]
            self.init_coeff = [[1, 'Coke1', 10], 
                [2, 'O2', 7.5], [2, 'CO', 0.5], [2, 'H2O', 24], 
                [3, 'Coke2', 1.033], 
                [4, 'O2', 1.2], [4, 'CO', 0.4], [4, 'H2O', 1]]
        

        elif reaction_type == 'Chen1':
            '''
            (1)       OIL-H     -> OIL
            (2)       OIL + O2  -> COKE + CO + CO2 + H2O + GAS
            (3)       COKE + O2 -> CO + CO2 + H2O

            '''

            self.reac_names = [['OIL-H'], ['OIL', 'O2'], ['Coke1', 'O2']] 
            self.prod_names = [['OIL'], ['Coke1', 'CO', 'CO2', 'H2O', 'Gas'], ['CO', 'CO2', 'H2O']] 
            
            self.heat_reaction = np.array([-1e3, -1e4, -4e4])
            self.pre_exp_factors = np.array([1e-1, 1e-2, 1e0])
            self.act_energies = np.array([5e2, 4e4, 9e4])

            self.rxn_constraints = [[2, 'O2', 'CO2', 5/6], [2, 'CO2', 'CO', 0.4], 
                [3, 'O2', 'CO2', 5/6], [3, 'CO2', 'CO', 0.4]]
            self.init_coeff = [[1, 'OIL', 20], [2, 'Coke1', 5], [2, 'O2', 8], [3, 'O2', 5], [3, 'H2O', 2]]
        

        elif reaction_type == 'Chen2':
            '''
            (1)       OIL-H + O2 -> OIL
            (2)       OIL   + O2 -> CO   + CO2 + H2O

            '''

            self.reac_names = [['OIL-H', 'O2'], ['OIL', 'O2']] 
            self.prod_names = [['OIL'], ['CO', 'CO2', 'H2O']] 
            
            self.heat_reaction = np.array([-1e2, -4e3])
            self.pre_exp_factors = np.array([1e0, 1e1])
            self.act_energies = np.array([3e4, 6.5e4])

            # self.rxn_constraints = [[2, 'O2', 'CO2', 1], [2, 'CO2', 'CO', 0.2]]
            self.rxn_constraints = [[2, 'CO2', 'CO', 0.2]]
            self.init_coeff = [[1, 'O2', 10], [1, 'OIL', 10], [2, 'O2', 5], [2, 'H2O', 10]]


        elif reaction_type == 'Dechelette1':
            '''
            (1)       OIL-H + O2 -> Coke1
            (2)       Coke1 + O2 -> CO2 + H2O
            
            '''

            self.reac_names = [['OIL-H', 'O2'], ['Coke1', 'O2']] 
            self.prod_names = [['Coke1', 'CO2'], ['CO2', 'H2O']] 
            
            self.heat_reaction = np.array([-1e4, -4e4])
            self.pre_exp_factors = np.array([1e-1, 1e-2])
            self.act_energies = np.array([5e4, 9e4])

            self.rxn_constraints = [[2, 'O2', 'CO2', 1]]
            self.init_coeff = [[1, 'O2', 10], [2, 'O2', 4], [2, 'CO2', 4], [2, 'H2O', 8]]


        elif reaction_type == 'Dechelette2':
            '''
            From Dechelette (2006)

            (1)       OIL-H + O2   -> Coke1
            (2)       Coke1 + O2 -> CO2 + CO + H2O + Coke2
            (3)       Coke2 + O2 -> CO2 + CO + H2O
            
            '''

            self.reac_names = [['OIL-H', 'O2'], ['Coke1', 'O2'], ['Coke2', 'O2']] 
            self.prod_names = [['Coke1'], ['CO', 'CO2', 'H2O', 'Coke2'], ['CO', 'CO2', 'H2O']] 
            
            self.heat_reaction = np.array([0, -4e4, -5e4])
            self.pre_exp_factors = np.array([1e-3, 25, 1])
            self.act_energies = np.array([3.5e4, 7.3e4, 8e4])

            self.rxn_constraints = [[2, 'CO2', 'CO', 1/5], [3, 'CO2', 'CO', 1/5]]
            self.init_coeff = [[1, 'O2', 4.24], [2, 'O2', 33.8], [2, 'CO2', 23.77], [2, 'CO', 4.75], [2, 'H2O', 22], 
                [3, 'O2', 12.29], [3, 'CO2', 8.56], [3, 'CO', 1.75], [3, 'H2O', 8]]


        elif reaction_type == 'Crookston':
            '''
            From Crookston (1979)
            (1)       OIL-H      -> OIL + Coke1 + Gas
            (2)       OIL   + O2 -> CO2 + H2O
            (3)       OIL-H + O2 -> CO2 + H2O
            (4)       Coke1 + O2 -> CO2 + H2O

            '''
            self.reac_names = [['OIL-H', 'O2'], ['OIL', 'O2'], ['OIL-H', 'O2'], ['Coke1', 'O2']]
            self.prod_names = [['OIL', 'Coke1', 'Gas'], ['CO2', 'H2O'], ['CO2', 'H2O'], ['CO2', 'H2O']]

            self.heat_reaction = [-4.6e4, -2.19e5, -8.12e6, -5.2e4]
            self.pre_exp_factors = [5.04e-3, 1.68e-2, 1.68e-2, 1.68e-2]
            self.act_energies = [5e3, 7.75e4, 1e5, 5.4e4]

            self.rxn_constraints = []
            self.init_coeff = [[1, 'O2', 10], [1, 'OIL', 2.0], [1, 'Coke1', 4.67], [1, 'Gas', 13.3], 
                [2, 'O2', 5.0], [2, 'CO2', 3.0], [2, 'H2O', 4.0], 
                [3, 'O2', 18.0], [3, 'CO2', 12.0], [3, 'H2O', 12.0],
                [4, 'O2', 1.25], [4, 'CO2', 1.0], [4, 'H2O', 0.5]]

        else: 
            raise Exception('Invalid reaction type entered.')
            
