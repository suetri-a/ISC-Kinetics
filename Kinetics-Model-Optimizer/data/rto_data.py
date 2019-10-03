import autograd.numpy as np 
import pandas as pd
import glob, os
from .base_data import BaseData

class RtoData(BaseData):

    @staticmethod
    def modify_cmd_options(parser):
        parser.set_defaults(data_dir='datasets/RTO/')
        parser.add_argument('--data_from_default', type=str, default='None', help='load data from simulating default reaction')
        parser.add_argument('--dataset', type=str, default='synthetic', help='name of dataset to load from \'datasets/RTO\' directory')
        return parser


    def __init__(self, opts):
        super().__init__(opts)


    def get_O2_data(self):
        '''
        Return O2 consumption data

        '''
        O2_out = np.asarray([self.O2_consumption_data[hr] for hr in self.heating_rates])
        return O2_out


    def get_T_data(self):
        '''
        Return temperature data
        '''
        T_out = np.asarray([self.temperature_data[hr] for hr in self.heating_rates])
        return T_out


    def data_load(self, opts):
        '''
        Parse the input data file for experimental data kinetic cell models

        '''

        raw_data = {}
        for file in glob.glob(self.dataset_dir + "*Cmin.xlsx"):
            raw_data[float(file[len(self.dataset_dir):-9])] = pd.read_excel(file)
        self.heating_rates = [*raw_data]
        self.heating_rates.sort()
        self.num_heats = len(self.heating_rates)

        self.time_line = {} # times
        self.temperature_data = {} # temperature curves
        self.O2_consumption_data = {} # O2 consumption curves
        self.O2_con = {} # O2 concentration
        self.max_temp = {}

        for hr in self.heating_rates:
            temp_last_valid = raw_data[hr]['CH0'].last_valid_index()
            temperature_time_temp = raw_data[hr]['Time (s)'].values[:temp_last_valid] - raw_data[hr]['Time (s)'].values[0]
            self.max_temp[hr] = raw_data[hr]['CH0'][temp_last_valid]

            O2_last_valid = raw_data[hr]['Oxygen'].last_valid_index()
            O2_time_temp = raw_data[hr]['Time (s)'].values[:O2_last_valid] - raw_data[hr]['Time (s)'].values[0]

            self.time_line[hr] = np.linspace(O2_time_temp.min(), O2_time_temp.max(), num=100)
            
            self.temperature_data[hr] = np.interp(self.time_line[hr], temperature_time_temp, 
                                                raw_data[hr]['CH0'].values[:temp_last_valid], right=self.max_temp[hr])

            self.O2_con[hr] = raw_data[hr]['Oxygen'][O2_last_valid]
            self.O2_consumption_data[hr] = np.interp(self.time_line[hr], O2_time_temp, 
                                                self.O2_con[hr] - raw_data[hr]['Oxygen'].values[:O2_last_valid], right=0)


    def get_initial_condition(self, species, hr):
        '''
        Query initial conditions based on the heating rate and the species. 

        '''
        
        if species == 'O2':
            return self.O2_con[hr]
        elif species == 'OIL-H':
            return 4.0
        elif species == 'T':
            return self.temperature_data[hr][0]
        else:
            return 0.0