# Utilities for interfacing with CMG-STARS

import os
from . import CMG_Python as cmg
from .CMG_Python import *
import numpy as np

BASE_MODEL_DICT = {'CINAR': 1, 'KRISTENSEN': 2, 'CHEN': 3}


def get_component_dict(comp_name, base_model):

    component_dict = {}

    if base_model == 'CINAR':
        component_dict["H2O"] = Component(NAME="H2O",CMM=[1.8e-2],PCRIT=[22083],TCRIT=[373.8],
                                        AVG=[0],BVG=[0],AVISC=[0],BVISC=[0],
                                        CPG1=[0],CPG2=[0],CPG3=[0],CPG4=[0],CPL1=[0],CPL2=[0],
                                        MASSDEN=[0.001],CP=[0],CT1=[0],CT2=[0],SOLID_DEN=[],SOLID_CP=[])
        component_dict["OIL"] = Component(NAME="OIL",CMM=[4.73e-1],PCRIT=[890],TCRIT=[171],
                                        AVG=[0.0001610891804],BVG=[0.7453161006],AVISC=[1.426417368e-11],BVISC=[10823.06574],
                                        CPG1=[26.804420692906],CPG2=[0.005649089963],CPG3=[0.000095012314],CPG4=[-0.000000054709],CPL1=[524.8821790],CPL2=[1.148635444845],
                                        MASSDEN=[0.000999798],CP=[7.25e-7],CT1=[0.00069242],CT2=[0])
        component_dict["N2"] = Component(NAME="N2",CMM=[2.8e-2],PCRIT=[3392],TCRIT=[-147],
                                        AVG=[0.0003500869287],BVG=[0.6927470725],AVISC=[],BVISC=[],
                                        CPG1=[30.956477056957],CPG2=[-0.012716023994],CPG3=[0.000025490143],CPG4=[-0.000000011065],CPL1=[0],CPL2=[0])
        component_dict["O2"] = Component(NAME="O2",CMM=[3.2e-2],PCRIT=[5033],TCRIT=[-118],
                                        AVG=[0.000362791571],BVG=[0.7120986013],AVISC=[],BVISC=[],
                                        CPG1=[28.600167325729],CPG2=[-0.003497011859],CPG3=[0.000024399453],CPG4=[-0.000000014928],CPL1=[0],CPL2=[0])
        component_dict["CO2"] = Component(NAME="CO2",CMM=[4.4e-2],PCRIT=[7377],TCRIT=[31],
                                        AVG=[0.0001865724378],BVG=[0.7754816784],AVISC=[],BVISC=[],
                                        CPG1=[19.474325955388],CPG2=[0.075654731286],CPG3=[-0.000060750197],CPG4=[0.000000020109],CPL1=[0],CPL2=[0])
        component_dict["CO"] = Component(NAME="CO",CMM=[2.8e-2],PCRIT=[3496],TCRIT=[-144],
                                        AVG=[0.0003315014585],BVG=[0.7037315714],AVISC=[],BVISC=[],
                                        CPG1=[30.990187019402],CPG2=[-0.01392019971],CPG3=[0.00003014996],CPG4=[-0.00000001415],CPL1=[0],CPL2=[0])
        component_dict["Coke1"] = Component(NAME="Coke1",CMM=[1.88e-2],SOLID_DEN=[0.0014,0,0],SOLID_CP=[8.3908475,0.0439425])
        component_dict["Coke2"] = Component(NAME="Coke2",CMM=[1.36e-2],SOLID_DEN=[0.0014,0,0],SOLID_CP=[6.96015,0.03645])
        component_dict["Ci"] = Component(NAME="Ci",CMM=[2.08e-2],SOLID_DEN=[0.0014,0,0],SOLID_CP=[7.192155,0.037665])


    elif base_model == 'KRISTENSEN':
        component_dict['H2O'] = Component(NAME="WATER",CMM=[0.018],PCRIT=[21754.478],TCRIT=[647.4],ACEN=[0.344],
                                                AVG=[1.7e-5],BVG=[1.116],AVISC=[7.52e-5],BVISC=[1384.86],
                                                CPG1=[7.701],CPG2=[4.595e-4],CPG3=[2.521e-6],CPG4=[-0.859e-9],
                                                MOLDEN=[55520],CP=[4.352331606e-7],CT1=[2.16e-4])
        component_dict['OIL2'] = Component(NAME="HEVY OIL",CMM=[0.675],PCRIT=[830.865],TCRIT=[887.6],ACEN=[1.589],
                                                AVG=[7.503e-6],BVG=[1.102],AVISC=[4.02e-4],BVISC=[3400.89],
                                                CPG1=[-34.081],CPG2=[4.137],CPG3=[-2.279e-3],CPG4=[4.835e-7],
                                                MOLDEN=[1464],CP=[7.25388601e-7],CT1=[2.68e-4])
        component_dict['OIL'] = Component(NAME="LITE OIL",CMM=[0.157],PCRIT=[2107.56],TCRIT=[617.4],ACEN=[0.449],
                                                AVG=[3.77e-6],BVG=[0.943],AVISC=[4.02e-4],BVISC=[3400.89],
                                                CPG1=[-7.913],CPG2=[0.961],CPG3=[-5.29e-4],CPG4=[1.123e-7],
                                                MOLDEN=[5118],CP=[7.25388601e-7],CT1=[5.11e-4])
        component_dict['GAS'] = Component(NAME="INRT GAS",CMM=[0.041],PCRIT=[3445.05],TCRIT=[126.5],ACEN=[0.04],
                                                AVG=[3.213e-4],BVG=[0.702],
                                                CPG1=[31.150],CPG2=[-1.357e-2],CPG3=[2.679e-5],CPG4=[-1.167e-8])
        component_dict['O2'] = Component(NAME="OXYGEN",CMM=[0.032],PCRIT=[5035.852],TCRIT=[154.8],ACEN=[0.022],
                                                AVG=[3.355e-4],BVG=[0.721],
                                                CPG1=[28.106],CPG2=[-3.68e-6],CPG3=[1.746e-5],CPG4=[-1.065e-8])
        component_dict['COKE'] = Component(NAME="COKE",CMM=[0.013],SOLID_DEN=[916.344, 0, 0],SOLID_CP=[16, 0])

    elif base_model == 'CHEN':
        component_dict["H2O"] = Component(NAME="H2O",CMM=[1.8e-2],PCRIT=[22083],TCRIT=[373.8],
                                        AVG=[0],BVG=[0],AVISC=[0],BVISC=[0],
                                        CPG1=[0],CPG2=[0],CPG3=[0],CPG4=[0],CPL1=[0],CPL2=[0],
                                        MASSDEN=[0.001],CP=[0],CT1=[0],CT2=[0],SOLID_DEN=[],SOLID_CP=[])
        component_dict["OIL"] = Component(NAME="OIL",CMM=[4.73e-1],PCRIT=[890],TCRIT=[171],
                                        AVG=[0.0001610891804],BVG=[0.7453161006],AVISC=[1.426417368e-11],BVISC=[10823.06574],
                                        CPG1=[26.804420692906],CPG2=[0.005649089963],CPG3=[0.000095012314],CPG4=[-0.000000054709],CPL1=[524.8821790],CPL2=[1.148635444845],
                                        MASSDEN=[0.000999798],CP=[7.25e-7],CT1=[0.00069242],CT2=[0])
        component_dict["N2"] = Component(NAME="N2",CMM=[2.8e-2],PCRIT=[3392],TCRIT=[-147],
                                        AVG=[0.0003500869287],BVG=[0.6927470725],AVISC=[],BVISC=[],
                                        CPG1=[30.956477056957],CPG2=[-0.012716023994],CPG3=[0.000025490143],CPG4=[-0.000000011065],CPL1=[0],CPL2=[0])
        component_dict["O2"] = Component(NAME="O2",CMM=[3.2e-2],PCRIT=[5033],TCRIT=[-118],
                                        AVG=[0.000362791571],BVG=[0.7120986013],AVISC=[],BVISC=[],
                                        CPG1=[28.600167325729],CPG2=[-0.003497011859],CPG3=[0.000024399453],CPG4=[-0.000000014928],CPL1=[0],CPL2=[0])
        component_dict["CO2"] = Component(NAME="CO2",CMM=[4.4e-2],PCRIT=[7377],TCRIT=[31],
                                        AVG=[0.0001865724378],BVG=[0.7754816784],AVISC=[],BVISC=[],
                                        CPG1=[19.474325955388],CPG2=[0.075654731286],CPG3=[-0.000060750197],CPG4=[0.000000020109],CPL1=[0],CPL2=[0])
        component_dict["CO"] = Component(NAME="CO",CMM=[2.8e-2],PCRIT=[3496],TCRIT=[-144],
                                        AVG=[0.0003315014585],BVG=[0.7037315714],AVISC=[],BVISC=[],
                                        CPG1=[30.990187019402],CPG2=[-0.01392019971],CPG3=[0.00003014996],CPG4=[-0.00000001415],CPL1=[0],CPL2=[0])
        component_dict["Coke1"] = Component(NAME="Coke1",CMM=[1.88e-2],SOLID_DEN=[0.0014,0,0],SOLID_CP=[8.3908475,0.0439425])
        component_dict["Coke2"] = Component(NAME="Coke2",CMM=[1.36e-2],SOLID_DEN=[0.0014,0,0],SOLID_CP=[6.96015,0.03645])
        component_dict["Ci"] = Component(NAME="Ci",CMM=[2.08e-2],SOLID_DEN=[0.0014,0,0],SOLID_CP=[7.192155,0.037665])

    return component_dict


def create_stars_runfile(filename, base_model, components, kinetics, heating_rate):
    '''
    Create runfile for specified reaction
    '''

    fluid_option = BASE_MODEL_DICT[base_model]

    fileID = open(filename, 'a')
    cmg.print_IO_control(fileID, fluid_option)
    cmg.print_grid(fileID, fluid_option)
    cmg.print_fluid(fileID, fluid_option, components, kinetics)
    cmg.K_value_table(fluid_option, fileID)
    cmg.print_ref_cond(fileID, fluid_option)
    cmg.print_rock_fluid(fileID, fluid_option)
    cmg.print_initial_cond(fileID, fluid_option)
    cmg.print_numerical(fileID, fluid_option)
    cmg.print_recurrent(fileID, fluid_option)
    cmg.print_heater(fileID, fluid_option)
    cmg.print_heating_rate(fileID, heating_rate, fluid_option)
    fileID.close()


def run_stars_runfile(filename, exe_path, cd_path):
    # Run CMG .dat file
    # Example: 
    #     exe_path = '"C:\\Program Files (x86)\\CMG\\STARS\\2017.10\\Win_x64\\EXE\\st201710.exe"'
    #     cd_path = 'cd C:\\Users\\yunanli\\OneDrive - Leland Stanford Junior University\\CMG_Python\\test1 '

    os.system('{}&{} -f {}.dat'.format(cd_path, exe_path, filename))


def read_stars_output(filename, base_model):

    '''
    Input:
        filename - base of file for the simulation

    '''

    # Open STARS output files
    FID = [line.decode('utf-8', errors='ignore') for line in open(filename + '.irf', 'rb')]
    FID_bin = open(filename + '.mrf', 'rb')

    # Select number of species
    #   NEED TO GENERALIZE ON FUTURE ITERATIONS
    if base_model == 'MURAT':
        numSPHIST = 22 # MURAT
    elif base_model == 'BO':
        numSPHIST = 18 # BO
    else:
        raise Exception('Invalid reaction type')
    num_grids = 22

    # Initialize variables/objects for parsing data
    i = 0
    UNIT = {} # Create units dictionary
    COMP = {} # Compositions dictionary
    TIME = {} # Time dictionary
    GRID = {} # Grid dictionary
    GRID['TEMP'] = []
    SP = {}
    SP['VAL'] = []
    SPHIST = {}

    TIME['VECT'] = []
    TIME['CHR'] = []
    TIME['CHR_UNIT'] = []

    REC = {}
    REC_list = ['WELL-REC', 'LAYER-REC', 'GROUP-REC', 'SECTOR-REC', 'RSTSPEC01-REC', 'RSTSPEC02-REC', 'RSTSPEC03-REC',
        'RSTSPEC04-REC', 'RSTSPEC05-REC', 'RSTSPEC06-REC', 'RSTSPEC07-REC', 'RSTSPEC08-REC', 'RSTSPEC09-REC',
        'RSTSPEC10-REC', 'RSTSPEC11-REC', 'RSTSPEC12-REC', 'RSTSPEC13-REC', 'RSTSPEC14-REC', 'RSTSPEC15-REC',
        'RSTSPEC16-REC', 'RSTSPEC17-REC', 'RSTSPEC18-REC', 'RSTSPEC19-REC', 'RSTSPEC20-REC', 'RSTSPEC21-REC',
        'RSTSPEC22-REC']
    
    skip_list = ['WELL-ARRAY', 'LAYER-ARRAY', 'GROUP-ARRAY']

    # Iterate over lines of file
    while i < len(FID):
        line_split = FID[i].split()
        
        # Parse case-by-case quantities
        if line_split[0] == 'INTERNAL-UNIT-TABLE':
            i+=1 
            UNIT['INT'] = [str(FID[i+j].split()[1]) for j in range(21)]
            UNIT['DESCP'] = [str(FID[i+j].split()[3]) for j in range(21)]
            i+=21 
            
        elif line_split[0] == 'OUTPUT-UNIT-TABLE':
            i+=1
            UNIT['OUT'] = [str(FID[i+j].split()[1]) for j in range(21)]
            i+=21

        elif line_split[0] == 'NCOMP':
            COMP['NUM'] = int(line_split[1])
            i+=1

        elif line_split[0] == 'COMPNAME':
            i+=1
            COMP['NAME'] = [str(FID[i+j].split()[1]).replace("'","") for j in range(COMP['NUM'])]
            i+=COMP['NUM']

        elif line_split[0] == 'COMP-PHASE-TEMPLATE':
            i+=1
            COMP['TEMPL'] = [str(FID[i+j].split()[2]) for j in range(COMP['NUM'])]
            i+=COMP['NUM']
            
        elif line_split[0] == 'TIME' and 'SPEC-HISTORY' in [FID[i-1].split()[0], FID[i-3].split()[0]]:
            TIME['VECT'].append(line_split[1:])
            i+=1
            
        elif line_split[0] == 'TIMCHR':
            TIME['CHR'].append(line_split[2])
            TIME['CHR_UNIT'].append(line_split[3].replace("'",""))
            i+=1
        

        # Parse grid values from binary file
        elif line_split[0] == 'GRID':
            item_num = int(FID[i].split()[2])
            for j in range(item_num):
                num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                _ = np.fromfile(FID_bin, np.int8, count=np.asscalar(num_bytes)).byteswap() 
            _, i = parse_nobin(FID, i)

        elif line_split[0] == 'GRID-VALUE':
            props_list, i = parse_nobin(FID, i)
            print('self.parse_nobin is', parse_nobin(FID, i))
            for prop in props_list[3:]:
                num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                print('num_bytes is', num_bytes)
                print('prop is', prop)
                print('np.asscalar(num_bytes) is', np.asscalar(num_bytes))
                if prop == 'TEMP':
                    grid_temp = np.fromfile(FID_bin, np.float64, count=num_grids).byteswap()
                    GRID['TEMP'].append(grid_temp)
                else:
                    _ = np.fromfile(FID_bin, np.int8, count=np.asscalar(num_bytes)).byteswap()


        # Parse species names and values from binary file
        elif line_split[0] == 'SPHIST-NAMES':
            SPHIST['NUM'] = []
            SPHIST['NAME'] = []
            i+=1
            for j in range(numSPHIST):
                line_split = FID[i+j].split()
                SPHIST['NUM'].append(line_split[0])
                SPHIST['NAME'].append(' '.join([line.replace("'","") for line in line_split[3:]]))
            i+=numSPHIST

        elif line_split[0] == 'SPEC-HISTORY':
            props_list, i = parse_nobin(FID, i)
            for prop in props_list[3:]:
                num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                # print(prop)
                # print(np.asscalar(num_bytes))
                if prop == 'SPVALS':
                    spvals_temp = np.fromfile(FID_bin, np.float64, count=numSPHIST).byteswap()
                    SP['VAL'].append(spvals_temp)
                else:
                    _ = np.fromfile(FID_bin, np.byte, count=np.asscalar(num_bytes)).byteswap()
        

        # Variables stored in binary but skipped in parsing
        elif line_split[0] in skip_list:
            item_num = int(line_split[2])
            for j in range(item_num):
                num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                _ = np.fromfile(FID_bin, np.byte, count=np.asscalar(num_bytes)).byteswap() 
            i+=1
        

        # Other variables parsed from text file
        elif line_split[0] in REC_list:
            list_temp = FID[i].split()
            while list_temp[-1] != '/':
                i+=1
                list_temp += FID[i].split()
            REC[line_split[0]] = list_temp[1:-1]
            i+=1

        else:
            i+=1
    
    FID_bin.close()

    # OUTPUT VARIABLES MURAT
    if base_model == 'CINAR':
        t = (np.asarray(TIME['VECT'])[:,1]).astype(np.float)
        TEMP_VALS = np.asarray(GRID['TEMP'])
        SPEC_VALS = np.asarray(SP['VAL'])
        N2_PROD   = SPEC_VALS[:,8]
        O2_PROD   = SPEC_VALS[:,9] 
        H2O_PROD  = SPEC_VALS[:,10]
        CO_PROD   = SPEC_VALS[:,11]
        CO2_PROD  = SPEC_VALS[:,12]
        lin_HR = np.asarray(TEMP_VALS)[:,8]
    elif base_model == 'CHEN':
        # OUTPUT VARIABLES BO
        TEMP_VALS = np.asarray(GRID['TEMP'])
        SPEC_VALS = np.asarray(SP['VAL'])
        N2_PROD = SPEC_VALS[6,:]
        O2_PROD = SPEC_VALS[7,:] 
        H2O_PROD = SPEC_VALS[8,:]
        CO_PROD = SPEC_VALS[9,:]
        CO2_PROD = SPEC_VALS[10,:]
        lin_HR = TEMP_VALS[8,:]

    ydict = {'O2': O2_PROD, 'N2': N2_PROD, 'H2O': H2O_PROD, 'CO': CO_PROD, 'CO2': CO2_PROD, 'Temp': lin_HR}

    return t, ydict
    

def parse_nobin(file_in, i):
    '''
    file_in - file to be parsed
    i - current line number
    '''
    
    list_temp = file_in[i].split()
    while list_temp[-1] != '/':
        i+=1
        list_temp += file_in[i].split()
    i+=1

    return list_temp[1:-1], i