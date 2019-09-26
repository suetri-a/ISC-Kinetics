import numpy as np 
import struct


def parse_runfile(self, input_file_base, reaction_type = 'MURAT'):

    # Open STARS output files
    FID = [line.decode('utf-8', errors='ignore') for line in open(input_file_base + '.irf', 'rb')]
    FID_bin = open(input_file_base + '.mrf', 'rb')

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
    
    skip_list = ['WELL-ARRAY', 'LAYER-ARRAY', 'GROUP-ARRAY']

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
                # print(prop)
                # print(np.asscalar(num_bytes))
                if prop == 'SPVALS':
                    spvals_temp = np.fromfile(FID_bin, np.float64, count=self.numSPHIST).byteswap()
                    self.SP['VAL'].append(spvals_temp)
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
            self.REC[line_split[0]] = list_temp[1:-1]
            i+=1

        else:
            i+=1
    
    FID_bin.close()

    # OUTPUT VARIABLES MURAT
    self.t = (np.asarray(self.TIME['VECT'])[:,1]).astype(np.float)
    TEMP_VALS = np.asarray(self.GRID['TEMP'])
    self.lin_HR = np.asarray(TEMP_VALS)[:,8]
    self.SPEC_VALS = np.asarray(self.SP['VAL'])
    
    if reaction_type == 'MURAT':
        self.N2_PROD   = self.SPEC_VALS[:,8]
        self.O2_PROD   = self.SPEC_VALS[:,9] 
        self.H2O_PROD  = self.SPEC_VALS[:,10]
        self.CO_PROD   = self.SPEC_VALS[:,11]
        self.CO2_PROD  = self.SPEC_VALS[:,12]
    elif reaction_type == 'BO':
        # OUTPUT VARIABLES BO
        self.N2_PROD = self.SPEC_VALS[:,6]
        self.O2_PROD = self.SPEC_VALS[:,7] 
        self.H2O_PROD = self.SPEC_VALS[:,8]
        self.CO_PROD = self.SPEC_VALS[:,9]
        self.CO2_PROD = self.SPEC_VALS[:,10]