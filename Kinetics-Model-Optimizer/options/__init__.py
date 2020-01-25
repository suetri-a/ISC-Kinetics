import numpy as np

def get_predefined_rxn(opts):

    if opts.reaction_model =='Cinar':
        '''
        Reaction scheme:
        (1)       OIL-H + O2 -> COKE1       + H2O
        (2)       COKE1 + O2 -> CO    + CO2 + H2O
        (3)       COKE1      -> COKE2
        (4)       COKE2 + O2 -> CO    + CO2 + H2O
        (5)       COKE1      -> CI
        (6)       CI    + O2 -> CO    + CO2
        '''
        
        reac_names = [['OIL-H','O2'], ['Coke1', 'O2'], ['Coke1'], ['Coke2', 'O2'], ['Coke1'], ['Ci', 'O2']] 
        prod_names = [['Coke1', 'H2O'], ['CO', 'CO2', 'H2O'], ['Coke2'], ['CO', 'CO2', 'H2O'], ['Ci'], ['CO', 'CO2']] 
        
        heat_reaction = np.array([0, -1e4, 0, -2e4, 0, -5e4])
        pre_exp_factors = np.array([1e1, 1e1, 1e-1, 1e-1, 1e-3, 1e-3])
        act_energies = np.array([1e5, 5e4, 6e4, 8e4, 8e4, 1e5])

        rxn_constraints = [[2, 'O2', 'CO2', 0.5], [2, 'O2', 'CO', 0.21], 
            [4, 'O2', 'CO2', 1/1.15], [4, 'O2', 'CO', 0.286]]
        init_coeff = [[1, 'O2',30], [1, 'Coke1', 10], [1, 'H2O', 2],
            [2, 'O2', 5], [2, 'CO', 0.3181], [2, 'CO2', 0.75], [2, 'H2O', 1.3815], 
            [3, 'Coke2', 1.3824], 
            [4, 'O2', 0.65], [4, 'CO', 0.1862], [4, 'CO2', 0.5652], [4, 'H2O', 0.2393],
            [5, 'Ci', 0.9135],
            [6, 'O2', 0.9], [6, 'CO', 0.2]]
    

    elif opts.reaction_model == 'CinarMod1':
        '''
        Reaction scheme:
        (1)       OIL-H      -> COKE1 + CI
        (2)       COKE1 + O2 -> CO    + CO2 + H2O
        (3)       OIL-H      -> COKE2
        (4)       COKE2 + O2 -> CO    + CO2 + H2O
        (5)       CI    + O2 -> CO    + CO2 + H2O
        '''
        
        reac_names = [['OIL-H'], ['Coke1', 'O2'], ['OIL-H'], ['Coke2', 'O2'], ['Ci', 'O2']] 
        prod_names = [['Coke1', 'Ci'], ['CO', 'CO2', 'H2O'], ['Coke2'], ['CO', 'CO2', 'H2O'], ['CO', 'CO2', 'H2O']] 
        
        heat_reaction = np.array([0, -5e5, 0, -4e5, -4e5])
        pre_exp_factors = np.array([4.231e13, 3.265e8, 9.334e6, 5.621e9, 1.621e10])
        act_energies = np.array([8.62e4,  9.29e4,  8.0e4, 1.21e5, 1e5])

        rxn_constraints = [[2, 'O2', 'CO2', 1/1.15], [2, 'CO2', 'CO', 0.77], 
            [4, 'O2', 'CO2', 1/1.3], [4, 'CO2', 'CO', 0.41], 
            [5, 'O2', 'CO2', 1/1.5], [5, 'CO2', 'CO', 0.77]]
        init_coeff = [[1, 'Coke1', 5], [1, 'Ci', 5],
            [2, 'O2', 7.5], [2, 'CO', 0.5], [2, 'H2O', 2], 
            [3, 'Coke2', 5],
            [4, 'O2', 1.1], [4, 'CO', 0.4], [4, 'H2O', 2], 
            [5, 'O2', 7.5], [5, 'CO', 0.5], [5, 'H2O', 2]]
    

    elif opts.reaction_model == 'CinarMod2':
        '''
        Reaction scheme:
        (1)       OIL-H      -> COKE1
        (2)       COKE1 + O2 -> CO    + CO2 + H2O
        (3)       OIL-H      -> COKE2
        (4)       COKE2 + O2 -> CO    + CO2 + H2O

        '''

        reac_names = [['OIL-H'], ['Coke1', 'O2'], ['OIL-H'], ['Coke2', 'O2']] 
        prod_names = [['Coke1'], ['CO', 'CO2', 'H2O'], ['Coke2'], ['CO', 'CO2', 'H2O']] 
        
        heat_reaction = np.array([0, -5e5, 0, -4e5])
        pre_exp_factors = np.array([4.231e13, 3.265e8, 9.334e6, 5.621e9])
        act_energies = np.array([8.62e4, 9.29e4, 8.0e4, 1.21e5])

        rxn_constraints = [[2, 'O2', 'CO2', 1/1.15], [2, 'CO2', 'CO', 0.77], 
            [4, 'O2', 'CO2', 1/1.3], [4, 'CO2', 'CO', 0.41]]
        init_coeff = [[1, 'Coke1', 7], 
            [2, 'O2', 7.5], [2, 'CO', 0.5], [2, 'H2O', 5], 
            [3, 'Coke2', 19], 
            [4, 'O2', 1.2], [4, 'CO', 0.4], [4, 'H2O', 5]]
    

    elif opts.reaction_model == 'CinarMod3':
        '''
        (1)       OIL-H      -> COKE1
        (2)       COKE1 + O2 -> CO    + CO2 + H2O
        (3)       COKE1      -> COKE2
        (4)       COKE2 + O2 -> CO    + CO2 + H2O

        '''

        reac_names = [['OIL-H'], ['Coke1', 'O2'], ['Coke1'], ['Coke2', 'O2']] 
        prod_names = [['Coke1'], ['CO', 'CO2', 'H2O'], ['Coke2'], ['CO', 'CO2', 'H2O']] 
        
        heat_reaction = np.array([0, -5e5, 0, -4e5])
        pre_exp_factors = np.array([4.231e13, 3.265e8, 9.334e6, 5.621e9])
        act_energies = np.array([8.62e4, 9.29e4, 8.0e4, 1.21e5])

        rxn_constraints = [[2, 'O2', 'CO2', 1/1.15], [2, 'CO2', 'CO', 0.77], 
            [4, 'O2', 'CO2', 1/1.3], [4, 'CO2', 'CO', 0.41]]
        init_coeff = [[1, 'Coke1', 10], 
            [2, 'O2', 7.5], [2, 'CO', 0.5], [2, 'H2O', 24], 
            [3, 'Coke2', 1.033], 
            [4, 'O2', 1.2], [4, 'CO', 0.4], [4, 'H2O', 1]]
    

    elif opts.reaction_model == 'Chen1':
        '''
        (1)       OIL-H     -> OIL
        (2)       OIL + O2  -> COKE + CO + CO2 + H2O + GAS
        (3)       COKE + O2 -> CO + CO2 + H2O

        '''

        reac_names = [['OIL-H'], ['OIL', 'O2'], ['Coke1', 'O2']] 
        prod_names = [['OIL'], ['Coke1', 'CO', 'CO2', 'H2O', 'Gas'], ['CO', 'CO2', 'H2O']] 
        
        heat_reaction = [-1e3, -1e4, -4e4]
        pre_exp_factors = [1e-1, 1e-2, 1e0]
        act_energies = [5e2, 4e4, 9e4]

        rxn_constraints = [[2, 'O2', 'CO2', 5/6], [2, 'CO2', 'CO', 0.4], 
            [3, 'O2', 'CO2', 5/6], [3, 'CO2', 'CO', 0.4]]
        init_coeff = [[1, 'OIL', 20], [2, 'Coke1', 5], [2, 'O2', 8], [3, 'O2', 5], [3, 'H2O', 2]]
    

    elif opts.reaction_model == 'Chen2':
        '''
        (1)       OIL-H + O2 -> OIL
        (2)       OIL   + O2 -> CO   + CO2 + H2O

        '''

        reac_names = [['OIL-H', 'O2'], ['OIL', 'O2']] 
        prod_names = [['OIL'], ['CO', 'CO2', 'H2O']] 
        
        heat_reaction = [-1e2, -4e3]
        pre_exp_factors = [1e0, 1e1]
        act_energies = [3e4, 6.5e4]

        # self.rxn_constraints = [[2, 'O2', 'CO2', 1], [2, 'CO2', 'CO', 0.2]]
        rxn_constraints = [[2, 'CO2', 'CO', 0.5]]
        init_coeff = [[1, 'O2', 5], [1, 'OIL', 10], [2, 'O2', 5], [2, 'H2O', 10], [2, 'CO2', 6], [2, 'CO',3]]


    elif opts.reaction_model == 'Dechelette1':
        '''
        (1)       OIL-H + O2 -> Coke1
        (2)       Coke1 + O2 -> CO2 + H2O
        
        '''

        reac_names = [['OIL-H', 'O2'], ['Coke1', 'O2']] 
        prod_names = [['Coke1', 'CO2'], ['CO2', 'H2O']] 
        
        heat_reaction = [-1e4, -4e4]
        pre_exp_factors = [1e-1, 1e-2]
        act_energies = [5e4, 9e4]

        rxn_constraints = [[2, 'O2', 'CO2', 1]]
        init_coeff = [[1, 'O2', 10], [2, 'O2', 4], [2, 'CO2', 4], [2, 'H2O', 8]]


    elif opts.reaction_model == 'Dechelette2':
        '''
        From Dechelette (2006)

        (1)       OIL-H + O2   -> Coke1
        (2)       Coke1 + O2 -> CO2 + CO + H2O + Coke2
        (3)       Coke2 + O2 -> CO2 + CO + H2O
        
        '''

        reac_names = [['OIL-H', 'O2'], ['Coke1', 'O2'], ['Coke2', 'O2']] 
        prod_names = [['Coke1'], ['CO', 'CO2', 'H2O', 'Coke2'], ['CO', 'CO2', 'H2O']] 
        
        heat_reaction = [0, -4e4, -5e4]
        pre_exp_factors = [1e-3, 25, 1]
        act_energies = [3.5e4, 7.3e4, 8e4]

        rxn_constraints = [[2, 'CO2', 'CO', 1/5], [3, 'CO2', 'CO', 1/5]]
        init_coeff = [[1, 'O2', 4.24], [2, 'O2', 33.8], [2, 'CO2', 23.77], [2, 'CO', 4.75], [2, 'H2O', 22], 
            [3, 'O2', 12.29], [3, 'CO2', 8.56], [3, 'CO', 1.75], [3, 'H2O', 8]]


    elif opts.reaction_model == 'Crookston':
        '''
        From Crookston (1979)
        (1)       OIL-H      -> OIL + Coke1 + Gas
        (2)       OIL   + O2 -> CO2 + H2O
        (3)       OIL-H + O2 -> CO2 + H2O
        (4)       Coke1 + O2 -> CO2 + H2O

        '''
        reac_names = [['OIL-H', 'O2'], ['OIL', 'O2'], ['OIL-H', 'O2'], ['Coke1', 'O2']]
        prod_names = [['OIL', 'Coke1', 'Gas'], ['CO2', 'H2O'], ['CO2', 'H2O'], ['CO2', 'H2O']]

        heat_reaction = [-4.6e4, -2.19e5, -8.12e6, -5.2e4]
        pre_exp_factors = [5.04e-3, 1.68e-2, 1.68e-2, 1.68e-2]
        act_energies = [5e3, 7.75e4, 1e5, 5.4e4]

        rxn_constraints = []
        init_coeff = [[1, 'O2', 10], [1, 'OIL', 2.0], [1, 'Coke1', 4.67], [1, 'Gas', 13.3], 
            [2, 'O2', 5.0], [2, 'CO2', 3.0], [2, 'H2O', 4.0], 
            [3, 'O2', 18.0], [3, 'CO2', 12.0], [3, 'H2O', 12.0],
            [4, 'O2', 1.25], [4, 'CO2', 1.0], [4, 'H2O', 0.5]]

    else: 
        raise Exception('Invalid reaction type entered.')

    if hasattr(opts,'reac_names'): opts.reac_names = reac_names
    if hasattr(opts, 'prod_names'): opts.prod_names = prod_names
    if hasattr(opts, 'pre_exp_fwd'): opts.pre_exp_fwd = pre_exp_factors
    if hasattr(opts,'act_eng_fwd'): opts.act_eng_fwd = act_energies
    if hasattr(opts, 'rxn_constraints'): opts.rxn_constraints = rxn_constraints
    if hasattr(opts, 'init_coeff'): opts.init_coeff = init_coeff
    if hasattr(opts, 'heat_reaction'): opts.heat_reaction = heat_reaction

    return opts


def get_component_props(opts):

    # Store material properties in a dictionary of dictionaries
    opts.material_dict = {} 

    # Phase of each species: 1 - gas, 2 - liquid, 3 - gas, 4 - solid
    opts.comp_phase = {'H2O': 1,  'N2': 3, 'O2': 3, 'CO2': 3, 'CO': 3, 'Gas': 3, 
        'OIL-H': 2, 'OIL-L': 2, 'OIL': 2, 'Coke1': 4, 'Coke2': 4, 'Ci': 4}

    opts.fuel_comps = ['OIL-H', 'OIL-L']
    opts.pseudo_fuel_comps = ['Coke1', 'Coke2', 'Ci', 'OIL']
    opts.combustion_prods = ['H2O', 'CO', 'CO2', 'Gas']

    opts.IC_dict = {'H2O': 0, 'N2': 0, 'O2': None, 'CO2': 0, 'CO': 0, 'Gas': 0, 
                'OIL-H': 2e4, 'OIL-L': 2e4,  # Dictionary of initial condition values
                'OIL': 0, 'Coke1': 0, 'Coke2': 0, 'Ci': 0}

    if opts.kinetics_model == 'arrhenius': 
        
        # Molecular weights
        opts.material_dict['M'] = {'H2O': 18, 'N2': 28, 'O2': 32, 'CO2': 44, 'CO': 28, 'Gas': 28.04, # Product species
            'OIL-H': 800, 'OIL-L': 400, # Parent oil species
            'OIL': np.nan, 'Coke1': np.nan, 'Coke2': np.nan, 'Ci': np.nan # Pseudocomponents
            } 

        # Oxygen counts
        opts.material_dict['O'] = {'H2O': 1, 'N2': 0, 'O2': 2, 'CO2': 2, 'CO': 1, 'Gas': 1, 
            'OIL-H': 0, 'OIL-L': 0, 'OIL': np.nan, 'Coke1': np.nan, 'Coke2': np.nan, 'Ci': np.nan} 
        
        # Carbon counts
        opts.material_dict['C'] = {'H2O': 0, 'N2': 0, 'O2': 2, 'CO2': 1, 'CO': 1, 'Gas': 1, 
            'OIL-H': 50, 'OIL-L': 30, 'OIL': np.nan, 'Coke1': np.nan, 'Coke2': np.nan, 'Ci': np.nan} 

        # Heat capacity of each species [J/mole]
        opts.material_dict['Cp'] = {'N2': 29.93161, 'O2': 30.47713, 'H2O': 34.49885, 'CO2': 21.08550, 
            'CO': 30.08316, 'Gas': 30.99, 'OIL-H': 225.658, 'OIL-L': 225.658,
            'OIL': 125.658, 'Coke1': 8.3908475, 'Coke2': 6.96015, 'Ci': 7.192155}
    
    elif opts.kinetics_model == 'stars':
        # Oxygen counts
        opts.material_dict['O'] = {'H2O': 1, 'N2': 0, 'O2': 2, 'CO2': 2, 'CO': 1, 'Gas': 1, 
            'OIL-H': 0, 'OIL-L': 0, 'OIL': np.nan, 'Coke1': np.nan, 'Coke2': np.nan, 'Ci': np.nan} 
        
        # Carbon counts
        opts.material_dict['C'] = {'H2O': 0, 'N2': 0, 'O2': 2, 'CO2': 1, 'CO': 1, 'Gas': 1, 
            'OIL-H': 50, 'OIL-L': 30, 'OIL': np.nan, 'Coke1': np.nan, 'Coke2': np.nan, 'Ci': np.nan} 

        if opts.stars_base_model =='cinar':
            # Molecular weights
            opts.material_dict['M'] = {'H2O': 1.8e-2, 'N2': 2.8e-2, 'O2': 3.2e-2, 'CO2': 4.4e-2, 'CO': 2.8e-2, 'Gas': 2.8e-2, # Product species
                'OIL-H': 4.73e-1, 'OIL-L': 2.36e-1, # Parent oil species
                'OIL': np.nan, 'Coke1': np.nan, 'Coke2': np.nan, 'Ci': np.nan # Pseudocomponents
                }
        
        elif opts.stars_base_model == 'kristensen':
            pass

        elif opts.stars_base_model == 'chen':
            pass


    return opts