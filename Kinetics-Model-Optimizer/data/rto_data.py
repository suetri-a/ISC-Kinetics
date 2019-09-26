import autograd.numpy as np 
import pandas as pd
from .base_data import BaseData

class RtoData(BaseData):

    @staticmethod
    def modify_cmd_options(parser):
        parser.set_defaults(data_dir='data/RTO/')
        parser.add_argument('--data_from_default', type=str, default='None', help='load data from simulating default reaction')
        parser.add_argument('--dataset_name', type=str, default='synthetic', help='name of dataset to load from \'datasets/RTO\' directory')
        return parser


    def __init__(self, opts):
        self.data_load(opts)


    def get_O2_data(self):
        '''
        Return O2 consumption data

        '''
        oxy_out = np.stack([self.y[i][self.O2_ind,:] for i in range(self.num_heats)])
        return oxy_out

    def get_T_data(self):
        '''
        Return temperature data
        '''
        temp_out = np.stack([self.y[i][-1,:] for i in range(self.num_heats)])
        return temp_out


    def data_load(self, opts):
        '''
        Parse the input data file for experimental data kinetic cell models

        '''

        data = pd.read_excel(opts.data_file)

        # Collect heating rates
        column_names = data.columns.values
        self.heating_rates = []
        for i in range(column_names.shape[0]):
            if column_names[i][:7] != 'Unnamed':
                self.heating_rates.append(float(column_names[i][:-6])/60) # convert to C/sec
        self.num_heats = len(self.heating_rates)
        self.O2_ind = 2

        # Assemble the CO, CO2, and O2 solutions from the data
        data_vals = data.values[2:,:]
        self.ICs = []
        self.t = []
        self.y = []

        t_min = 0
        t_max = 0
        max_temp = 0

        for i, r in enumerate(self.heating_rates):

            # CO data
            CO = np.array(data_vals[:, i*12+11], dtype=float)
            CO = CO[np.isfinite(CO)]

            # CO2 data
            CO2 = np.array(data_vals[:, i*12+10], dtype=float)
            CO2 = CO2[np.isfinite(CO2)]

            # Oxygen data
            O2 = np.array(data_vals[:, i*12+9], dtype=float)
            O2 = O2[np.isfinite(O2)]/100

            # Temperature data
            T = np.array(data_vals[:, i*12+2], dtype=float)
            T = T[np.isfinite(T)] + 273.15
            
            # Time points (in seconds)
            time = np.array(data_vals[:, i*12+6], dtype=float)
            time = time[np.isfinite(time)]

            # O2 concentration
            O2_con_sim = np.array(data_vals[:, i*12+8]/100, dtype=float)
            O2_con_sim = O2_con_sim[np.isfinite(O2_con_sim)]
            
            # Shift curve as necessary
            time_end_ind = np.argmax(time)
            if O2[time_end_ind] == 0:
                O2 = O2_con_sim - O2 # calculate concentration
            else:
                shift_tan_angle = O2[time_end_ind]/time[time_end_ind]
                first_non_zero = np.min(np.nonzero(O2))
                O2[first_non_zero:time_end_ind+1] -= shift_tan_angle*time[first_non_zero:time_end_ind+1]
                O2[time_end_ind+1:] *= 0
                O2 = O2_con_sim - O2
            
            # Concatenate species and temperature into solution vector
            ys = np.concatenate((np.expand_dims(CO,0), np.expand_dims(CO2,0), 
                np.expand_dims(O2,0), np.expand_dims(T,0)))

            # Add to solutions for the kinetic cell object
            self.t.append(time)
            self.y.append(ys)
            self.ICs.append(ys[:,0])

            t_min = np.minimum(t_min, time[0])
            t_max = np.maximum(t_max, time[-1])
            max_temp = np.maximum(max_temp, T[-1])
        
        self.O2_con_sim = O2_con_sim[0]
        self.Tspan = [t_min, t_max]
        self.opts.max_temp = max_temp - 273.1