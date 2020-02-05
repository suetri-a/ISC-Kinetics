import numpy as np 
import os
from abc import ABC, abstractmethod

from .reactions import Component, Kinetics
from .utils import clear_stars_files, copy_up_to_string

class RtoBase(ABC):

    @staticmethod
    def get_default_reaction():
        pass


    def __init__(self, folder_name = './', input_file_name = 'rto_experiment'):
        self.folder_name = folder_name
        self.input_file_name = input_file_name
        self.sim_completed = False
        self.parsed = False


    ##############################################################################################
    #
    # BEGIN METHODS FOR WRITING THE RUN FILE
    #
    ##############################################################################################

    def write_dat_file(self, COMPS=None, REACTS=None, IC_dict=None, HR=None, 
                        TFINAL=500, TEMP_MAX = 750, O2_con_in = 0.21):
        '''
        Inputs:
            COMPS - dictionary of component data for the reaction
            REACTS - list of reactions for the runfile
            IC_dict - dictionary of initial condition for Heavy Oil and Oxygen. All other components
                are assumed to have an initial concentration of 0.
            HR - heating rate
            TSPAN - two-element list containing the start and end time (in minutes)

        '''
        clear_stars_files(self.folder_name)
        newfile_path = self.folder_name + self.input_file_name + '.dat'
        fileID = open(newfile_path, 'a')
                
        self.print_IO_control(fileID)
        self.print_grid(fileID)
        self.print_fluid(fileID, COMPS)
        self.print_reactions(fileID, REACTS, COMPS)
        self.print_ref_cond(fileID)
        self.print_rock_fluid(fileID)
        self.print_initial_cond(fileID, IC_dict)
        self.print_numerical(fileID)
        self.print_recurrent(fileID, O2_con_in, COMPS)
        self.print_heater(fileID)
        self.print_heating_ramp(fileID, HR, TFINAL, TEMP_MAX)
        fileID.close()


    @abstractmethod
    def print_IO_control(self, fileID):
        pass

    @abstractmethod
    def print_grid(self, fileID):
        pass
        
    def print_attrs(self, components, attrslist, fileID):
        '''
        Print attributes across all components. Components must be ordered by 
            phase value so the attributes are printed in the correct order.

        '''
        comp_names = components.keys() # get component names
        phases = [components[c].phase for c in comp_names] # get phases
        comp_names = [c for _, c in sorted(zip(phases,comp_names))] # sort comp_names according to phase

        for attr in attrslist:
            print_str='*' + attr
            for comp in comp_names:
                if hasattr(components[comp], attr):
                    if attr =='COMPNAME':
                        print_str+= ' ' + '\'' + str(getattr(components[comp], attr)) + '\'' 
                    else:
                        print_str+= ' ' + str(*getattr(components[comp], attr)) 
            print(print_str, file=fileID)
    
    
    @abstractmethod
    def print_fluid(self, fileID, components):
        pass
    

    def print_reactions(self, fileID, kinetics, components):
        print('** ==============  REACTIONS  ======================', file = fileID)
        
        # Loop over reactions in list
        for r in kinetics:
            r.print_rxn(fileID)

        num_k_vals = len([c for c in components.keys() if components[c].phase in [1,2]])
        kvals = ['0']*num_k_vals
        
        # Print K value table
        print("""
*KV1 {kvals} 
*KV2 {kvals}
*KV3 {kvals}
*KV4 {kvals}
*KV5 {kvals}

                """.format(kvals=' '.join(kvals)), file = fileID)
            

    @abstractmethod
    def print_ref_cond(self, fileID):
        pass

    @abstractmethod
    def print_rock_fluid(self, fileID):
        pass

    @abstractmethod
    def print_initial_cond(self, fileID, IC_dict):
        pass

    @abstractmethod
    def print_numerical(self, fileID):
        pass

    @abstractmethod
    def print_recurrent(self, fileID, O2_con_in, components):
        pass

    @abstractmethod
    def print_heater(self, fileID):
        pass

    @abstractmethod
    def print_heating_ramp(self, fileID, heating_rate, tfinal, tempmax):
        pass


    ##############################################################################################
    #
    # BEGIN METHODS FOR EXECUTING THE RUN FILE
    #
    ##############################################################################################

    def run_dat_file(self, exe_path, cd_path):
        '''
        Example for definition of path
        exe_path='"C:\\Program Files (x86)\\CMG\\STARS\\2017.10\\Win_x64\\EXE\\st201710.exe"'
        cd_path='cd C:\\Users\\yunanli\\Desktop\\CMG\\VKC '
        '''
        os_system_line = 'cd ' + cd_path + ' & ' + exe_path + '  -f ' + f'"{self.input_file_name}"' + '.dat'
        os.system(os_system_line)
        self.sim_completed = True


    ##############################################################################################
    #
    # BEGIN METHODS FOR PARSING RESULTS
    #
    ##############################################################################################

    def parse_stars_output(self):

        # Open STARS output files
        fid_temp = open(self.folder_name + self.input_file_name + '.irf', 'rb')
        FID = [line.decode('utf-8', errors='ignore') for line in fid_temp]
        FID_bin = open(self.folder_name + self.input_file_name + '.mrf', 'rb')

        # Initialize variables/objects for parsing data
        i = 0
        self.UNIT = {} # Create units dictionary
        self.COMP = {} # Compositions dictionary
        self.TIME = {} # Time dictionary
        self.GRID = {} # Grid dictionary
        self.GRID['TEMP'] = []
        self.SP = {}
        self.SP['VAL'] = []
        self.SPHIST = {}

        self.TIME['VECT'] = []
        self.TIME['CHR'] = []
        self.TIME['CHR_UNIT'] = []

        self.REC = {}
        REC_list = ['WELL-REC', 'LAYER-REC', 'GROUP-REC', 'SECTOR-REC', 'RSTSPEC01-REC', 'RSTSPEC02-REC', 'RSTSPEC03-REC',
            'RSTSPEC04-REC', 'RSTSPEC05-REC', 'RSTSPEC06-REC', 'RSTSPEC07-REC', 'RSTSPEC08-REC', 'RSTSPEC09-REC',
            'RSTSPEC10-REC', 'RSTSPEC11-REC', 'RSTSPEC12-REC', 'RSTSPEC13-REC', 'RSTSPEC14-REC', 'RSTSPEC15-REC',
            'RSTSPEC16-REC', 'RSTSPEC17-REC', 'RSTSPEC18-REC', 'RSTSPEC19-REC', 'RSTSPEC20-REC', 'RSTSPEC21-REC',
            'RSTSPEC22-REC']

        # Iterate over lines of file
        while i < len(FID):
            line_split = FID[i].split()
            
            # Parse case-by-case quantities
            if line_split[0] == 'INTERNAL-UNIT-TABLE':
                i+=1 
                self.UNIT['INT'] = [str(FID[i+j].split()[1]) for j in range(21)]
                self.UNIT['DESCP'] = [str(FID[i+j].split()[3]) for j in range(21)]
                i+=21 

            elif line_split[0] == 'OUTPUT-UNIT-TABLE':
                i+=1
                self.UNIT['OUT'] = [str(FID[i+j].split()[1]) for j in range(21)]
                i+=21

            elif line_split[0] == 'NCOMP':
                self.COMP['NUM'] = int(line_split[1])
                i+=1

            elif line_split[0] == 'COMPNAME':
                i+=1
                self.COMP['NAME'] = [str(FID[i+j].split()[1]).replace("'","") for j in range(self.COMP['NUM'])]
                i+=self.COMP['NUM']

            elif line_split[0] == 'COMP-PHASE-TEMPLATE':
                i+=1
                self.COMP['TEMPL'] = [str(FID[i+j].split()[2]) for j in range(self.COMP['NUM'])]
                i+=self.COMP['NUM']

            elif line_split[0] == 'TIME' and 'SPEC-HISTORY' in [FID[i-1].split()[0], FID[i-3].split()[0]]:
                self.TIME['VECT'].append(line_split[1:])
                i+=1

            elif line_split[0] == 'TIMCHR':
                self.TIME['CHR'].append(line_split[2])
                self.TIME['CHR_UNIT'].append(line_split[3].replace("'",""))
                i+=1


            # Parse grid values from binary file
            elif line_split[0] == 'GRID':
                item_num = int(FID[i].split()[2])
                for j in range(item_num):
                    num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                    _ = np.fromfile(FID_bin, np.int8, count=np.asscalar(num_bytes)).byteswap() 
                _, i = self.parse_nobin(FID, i)

            elif line_split[0] == 'GRID-VALUE':
                props_list, i = self.parse_nobin(FID, i)
                for prop in props_list[3:]:
                    num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                    if prop == 'TEMP':
                        grid_temp = np.fromfile(FID_bin, np.float64, count=int(num_bytes/8)).byteswap()
                        self.GRID['TEMP'].append(grid_temp)
                    else:
                        _ = np.fromfile(FID_bin, np.int8, count=np.asscalar(num_bytes)).byteswap()


            # Parse species names and values from binary file
            elif line_split[0] == 'SPHIST-NAMES':
                self.SPHIST['NUM'] = []
                self.SPHIST['NAME'] = []
                i+=1
                for j in range(self.numSPHIST):
                    line_split = FID[i+j].split()
                    self.SPHIST['NUM'].append(line_split[0])
                    self.SPHIST['NAME'].append(' '.join([line.replace("'","") for line in line_split[3:]]))
                i+=self.numSPHIST

            elif line_split[0] == 'SPEC-HISTORY':
                props_list, i = self.parse_nobin(FID, i)
                for prop in props_list[3:]:
                    num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                    if prop == 'SPVALS':
                        spvals_temp = np.fromfile(FID_bin, np.float64, count=self.numSPHIST).byteswap()
                        self.SP['VAL'].append(spvals_temp)
                    else:
                        _ = np.fromfile(FID_bin, np.byte, count=np.asscalar(num_bytes)).byteswap()


            # Variables stored in binary but skipped in parsing
            elif len(line_split) >= 4:
                if line_split[1]=='(' and line_split[3]==')':
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
                self.REC[line_split[0]] = list_temp[1:-1]
                i+=1

            else:
                i+=1

        FID_bin.close()

        self.t = (np.asarray(self.TIME['VECT'])[:,1]).astype(np.float)
        self.SPEC_VALS = np.asarray(self.SP['VAL'])
        self.parsed = True

            
    @staticmethod
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

    @abstractmethod
    def get_reaction_dict(self):
        pass