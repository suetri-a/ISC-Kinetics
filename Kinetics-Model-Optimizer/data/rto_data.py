import numpy as np 
import pandas as pd
import networkx as nx
import glob, os, warnings
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.integrate import trapz, cumtrapz
from scipy.optimize import minimize
from sklearn.cluster import KMeans, SpectralClustering


from .base_data import BaseData
from utils.utils import isoconversional_analysis


def load_rto_data(data_path, clean_data = True, return_O2_con_in = False):

    df = pd.read_excel(data_path + '.xls')
    
    # Read in data
    Time = df.Time.values/60
    O2 = df.O2.values
    CO2 = df.CO2.values
    Temp = df.Temperature.values
    
    ind100C = np.amin(np.asarray(Temp > 100).nonzero())
    ind120C = np.amin(np.asarray(Temp > 120).nonzero())
    inds750 = np.asarray(Temp > 745).nonzero()[0]
    ind750C1 = inds750[np.round(0.75*inds750.shape[0]).astype(int)]
    ind750C2 = inds750[np.round(0.9*inds750.shape[0]).astype(int)]


    # Gather datapoints and perform linear regression correction
    correction_times = np.concatenate([Time[ind100C:ind120C+1], Time[ind750C1:ind750C2+1]])
    correction_O2s = np.concatenate([O2[ind100C:ind120C+1], O2[ind750C1:ind750C2+1]])
    slope, intercept, _, _, _ = linregress(correction_times, correction_O2s)
    O2_baseline = slope*Time + intercept
    O2_con_in = intercept


    # Calculate %O2 consumption and conversion
    O2_consumption = np.maximum(O2_baseline - O2, 0)
    
    start_ind_max, end_ind_max_O2 = find_start_end(O2_consumption)
    O2_consumption[:start_ind_max] = 0
    O2_consumption[end_ind_max_O2:] = 0
    

    # Process CO2 data
    correction_CO2s = np.concatenate([CO2[ind100C:ind120C+1], CO2[ind750C1:ind750C2+1]])
    slope, intercept, _, _, _ = linregress(correction_times, correction_CO2s)
    CO2_baseline = slope*Time + intercept
    CO2_production = np.maximum(CO2 - CO2_baseline, 0)
    
    start_ind_max, end_ind_max_CO2 = find_start_end(CO2_production)
    CO2_production[:start_ind_max] = 0
    CO2_production[end_ind_max_CO2:] = 0 

    
    # Preliminary correction
    global_max_ind = max(end_ind_max_O2, end_ind_max_CO2)
    Time = Time[:global_max_ind]
    Temp = Temp[:global_max_ind]
    O2_consumption = O2_consumption[:global_max_ind]
    CO2_production = CO2_production[:global_max_ind]
    
    ydict = {'Time': Time, 'Temp': Temp, 'O2_consumption': O2_consumption, 'O2_con_in': O2_con_in, 
             'CO2_production': CO2_production}
    
    return ydict
    

def find_start_end(x):
    '''
    Find start and end time stamps from array x
    
    '''
    
    start_ind = 0
    end_ind = 0
    start_ind_max = 0
    end_ind_max = x.shape[0]
    cumsum = 0.0
    max_cumsum = 0.0
    for i in range(x.shape[0]):
        if x[i] <= 0:
            if cumsum > max_cumsum:
                max_cumsum = cumsum
                start_ind_max = start_ind
                end_ind_max = end_ind
            
            cumsum = 0.0
            start_ind = i
            end_ind = i
                
        else:
            cumsum += x[i]
            end_ind += 1
    
    return start_ind_max, end_ind_max


class RtoData(BaseData):

    @staticmethod
    def modify_cmd_options(parser):
        parser.set_defaults(data_dir=os.path.join('datasets', 'RTO'))
        parser.add_argument('--data_from_default', type=str, default='None', help='load data from simulating default reaction')
        parser.add_argument('--dataset', type=str, default='synthetic', help='name of dataset to load from \'datasets/RTO\' directory')
        parser.add_argument('--Oil_con_init', type=float, default=0.04, help='initial concentration of oil in kinetic cell')
        parser.add_argument('--interp_num', type=int, default=200, help='number of points to interpolate experimental data onto')
        return parser


    def __init__(self, opts):
        super().__init__(opts)


    def data_load(self, opts):
        '''
        Parse the input data file for experimental data kinetic cell models

        '''

        self.Oil_con_init = opts.Oil_con_init

        self.heating_rates = [float(name[:-9]) for name in os.listdir(self.dataset_dir) if name[-9:].lower()=='c_min.xls']

        # Begin loading data
        self.heating_data = {}
        self.O2_con_in = []
        INTERPNUM = 200
        
        for hr in self.heating_rates:
            print('Loading heating rate {}...'.format(hr))
            # Read RTO data
            ydict = load_rto_data(os.path.join(self.dataset_dir, str(hr)+'C_min'))
            
            # Downsample and append
            time_downsampled = np.linspace(ydict['Time'].min(), ydict['Time'].max(), num=INTERPNUM)
            Temp_ds = np.interp(time_downsampled, ydict['Time'], ydict['Temp'])
            O2_consumption_ds = np.interp(time_downsampled, ydict['Time'], ydict['O2_consumption']/100)
            CO2_production_ds = np.interp(time_downsampled, ydict['Time'], ydict['CO2_production']/100)

            self.heating_data[hr] = {'Time': time_downsampled, 'Temp': Temp_ds, 'O2': O2_consumption_ds, 'CO2': CO2_production_ds, 'O2_con_in': ydict['O2_con_in']/100}

    
    def print_curves(self, save_path = None):

        plt.figure()
        for hr in sorted(self.heating_data.keys()):
            plt.plot(self.heating_data[hr]['Time'], self.heating_data[hr]['O2'])
        plt.xlabel('Time [min]')
        plt.ylabel('O2 consumption [% mol]')
        plt.title('O2 consumption for experiments')
        plt.legend([str(hr) for hr in sorted(self.heating_data.keys())])

        if isinstance(save_path, str):
            plt.savefig(save_path[:-4] + '_consumption' + save_path[-4:])
        else:
            plt.show()

        plt.figure()
        for hr in sorted(self.heating_data.keys()):
            plt.plot(self.heating_data[hr]['Time'], self.heating_data[hr]['Temp'])
        plt.xlabel('Time [min]')
        plt.ylabel('Temperature [C]')
        plt.title('Temperature profiles for experiments')
        plt.legend([str(hr) for hr in sorted(self.heating_data.keys())])

        if isinstance(save_path, str):
            plt.savefig(save_path[:-4] + '_temperature' + save_path[-4:])
        else:
            plt.show()


    def get_heating_data(self):
        
        '''
        Get dictionary of times and temperature data.

        '''
        hr_dict = {}
        for hr in self.heating_rates:
            Time, Temp, _ = self.heating_data[hr]
            hr_dict[hr] = {'Time': Time, 'Temp': Temp}
        
        return hr_dict


    def get_initial_condition(self, species, hr):
        '''
        Query initial conditions based on the heating rate and the species. 

        '''
        
        if species == 'O2':
            return self.O2_con_in[self.heating_rates.index(hr)]
        elif species == 'Oil':
            return self.Oil_con_init
        elif species == 'T':
            return self.heating_data[hr]['Temp'][0]
        else:
            return 0.0
    
            
    def print_isoconversional_curves(self, save_path=None, corrected = False):
        conv_grid, O2_eact, O2_rorder, O2_preexp = isoconversional_analysis(self.heating_data, corrected=corrected)
    
        plt.figure()
        plt.plot(conv_grid, O2_eact)
        plt.xlabel('O2 conversion [% mol]')
        plt.ylabel('Activation energy [J/mol]]')
        plt.title('O2 activation energy')

        if isinstance(save_path, str):
            plt.savefig(save_path[:-4] + '_O2_eact' + save_path[-4:])
        else:
            plt.show()
        
        plt.figure()
        plt.plot(conv_grid, O2_rorder)
        plt.xlabel('O2 conversion [% mol]')
        plt.ylabel('Reaction Order')
        plt.title('O2 conversion reaction order')

        if isinstance(save_path, str):
            plt.savefig(save_path[:-4] + '_O2_rorder' + save_path[-4:])
        else:
            plt.show()
            
        plt.figure()
        plt.plot(conv_grid, np.exp(O2_preexp))
        plt.xlabel('O2 conversion [% mol]')
        plt.ylabel('Pre-exponential factor')
        plt.title('O2 conversion pre-exponential factor')

        if isinstance(save_path, str):
            plt.savefig(save_path[:-4] + '_O2_preexp' + save_path[-4:])
        else:
            plt.show()
        

    def compute_kinetics_params(self, num_rxns, return_labels=False):
        
        conv_grid, O2_eact, _, O2_preexp = isoconversional_analysis(self.heating_data, corrected=True)
        
        conv_grid_fit = (conv_grid - np.mean(conv_grid)) / np.std(conv_grid)
        O2_eact_fit = (O2_eact - np.mean(O2_eact)) / np.std(O2_eact)
        
        labels = SpectralClustering(n_clusters=num_rxns, affinity='nearest_neighbors', n_neighbors=10).fit_predict(np.concatenate((np.expand_dims(conv_grid_fit,1), 
                                                                                    np.expand_dims(O2_eact_fit,1)),axis=1))
        
        # Transform labels to be ascending order of conversion value
        mean_convs = [np.mean(np.array(conv_grid)[labels==i]) for i in range(num_rxns)]
        label_sort_inds = sorted(range(num_rxns),key=mean_convs.__getitem__)
        labels = [label_sort_inds.index(l) for l in labels]
        
        act_eng = [np.mean(np.array(O2_eact)[np.equal(labels,i)]) for i in range(num_rxns)]
        pre_exp = [np.exp(np.mean(np.array(O2_preexp)[np.equal(labels,i)])) for i in range(num_rxns)]
        
        if return_labels:
            return pre_exp, act_eng, labels
        else:
            return pre_exp, act_eng
        
        
    def print_isoconversional_overlay(self, num_rxns=None, save_path=None):
        
        if num_rxns is None:
            raise Exception('Must enter number of oxygenated reactions.')
            
        conv_grid, O2_eact, _, _ = isoconversional_analysis(self.heating_data, corrected=True)
        _, e_acts, labels = self.compute_kinetics_params(num_rxns, return_labels=True)
        
        plt.figure()
        plt.plot(conv_grid, O2_eact)
        plt.xlabel('O2 conversion [% mol]')
        plt.ylabel('Activation energy [J/mol]]')
        plt.title('O2 activation energy')
        
        for i in range(num_rxns):
            convs = np.array(conv_grid)[np.equal(labels,i)]
            eacts = e_acts[i]*np.ones_like(convs)
            plt.scatter(convs, eacts)
        
        plt.legend(['Observed Activation Energy']+['Reaction {}'.format(i+1) for i in range(num_rxns)])

        if isinstance(save_path, str):
            plt.savefig(save_path[:-4] + '_isconversional_overlay' + save_path[-4:])
        
        plt.show()


    def compute_bounds(self, param_types, log_params=True):
        '''
        Compute bounds for each parameter based on the data

        '''
        _, O2_eact, _, O2_preexp = isoconversional_analysis(self.heating_data, corrected=True)

        eact_min = np.maximum(np.amin(O2_eact), 1e3)
        eact_max = np.minimum(np.amax(O2_eact), 1e6)

        preexp_min = np.maximum(np.amin(np.exp(O2_preexp)), 1e-2)
        preexp_max = np.minimum(np.amax(np.exp(O2_preexp)), 1e3)

        if log_params:
            eact_min, eact_max = np.log(eact_min)-1, np.log(eact_max)+1
            preexp_min, preexp_max = np.log(preexp_min), np.log(preexp_max)

        bnds = []

        for p in param_types:
            if p[0] == 'acteng':
                bnds.append((eact_min, eact_max))
            elif p[0] == 'preexp':
                bnds.append((preexp_min, preexp_max))
            elif p[0] == 'stoic':
                bnds.append((1e-2, 20))


        return bnds
            
    
    def compute_initial_guess(self, reac_names, prod_names, res, param_types, log_params=True):
        '''
        Inputs:
            reac_names - list of reactant names for every reaction
            prod_names - list of product names for every reaction
            res - function res(x) that computes the sum of squared residuals from an input parameter vector x
            param_types - list of parameter types 
            log_params - if log of pre-exponential factors and activation energy being used
            
        Returns: 
            x0 - initial guess for the parameters vector
        
        '''
        
        # Initialize parameter vector
        x0 = np.ones((len(param_types)))
        num_rxns = len(reac_names)
        
        # Begin building initial guess for pre-exponential factor and activation energies
        oxy_rxns= [i for i, r in enumerate(reac_names) if 'O2' in r]
        num_oxy_rxns = len(oxy_rxns)
        
        # Get guesses for oxygen-containing reactions
        pre_exp, act_engs = self.compute_kinetics_params(num_oxy_rxns)
        
        # Get distance scores for each reaction
        fuel_names = []
        comp_names = list(set([comp for reac_prod in reac_names+prod_names for comp in reac_prod]))
        for c in comp_names:
            if len(c)>=3:
                if c[:3]=='Oil':
                    fuel_names.append(c)
            if len(c)>=4:
                if c[:4]=='Coke':
                    fuel_names.append(c)
                
        # Build fuel graph
        G = nx.Graph()
        G.add_nodes_from(fuel_names)
        G.add_edge('Oil', 'Oil')

        fuel_set = set(fuel_names)
        for i in range(num_rxns):
            reac_fuels = list(fuel_set.intersection(set(reac_names[i])))
            prod_fuels = list(fuel_set.intersection(set(prod_names[i])))
            for f1 in reac_fuels:
                for f2 in prod_fuels:
                    G.add_edge(f1, f2)
        
        # Get shortest path between Oil and fuels
        shortest_fuel_paths = {}
        for f in fuel_names:
            shortest_fuel_paths[f] = nx.shortest_path_length(G, source='Oil', target=f)
            
        reaction_path_lengths = {}
        for i, r in enumerate(reac_names):
            for s in r: # Note: relies on only one fuel per reaction
                if s in fuel_names:
                    reaction_path_lengths[i] = shortest_fuel_paths[s]
        
        # Gather path lengths associated with each oxygen-containing reaction
        oxy_path_lengths = [reaction_path_lengths[i] for i in oxy_rxns]
        
        # Reaction index to nearest oxygen-containing reaction
        rounded_path_inds = [min(range(len(oxy_rxns)), key=lambda x: abs(reaction_path_lengths[i]-oxy_path_lengths[x])) for i in range(num_rxns)]
        
        # Compile pre-exponential factor and activation energy 
        pre_exps_all = [pre_exp[rounded_path_inds[i]] for i in range(num_rxns)]
        act_engs_all = [act_engs[rounded_path_inds[i]] for i in range(num_rxns)]
        
        if log_params:
            pre_exps_all = [np.log(A) for A in pre_exps_all]
            act_engs_all = [np.log(E) for E in act_engs_all]

        bnds=[]
        for i, p in enumerate(param_types):
            if p[0] =='preexp':
                x0[i] = pre_exps_all[p[1]]
                bnds.append((-np.inf, np.inf))
            elif p[0] == 'acteng':
                x0[i] = act_engs_all[p[1]]
                bnds.append((-np.inf, np.inf))
            else:
                bnds.append((0, np.inf))
                
        # Find coefficients that minimize the residual (i.e. physical reaction)
        sol = minimize(res, x0, bounds=bnds)
        x0 = sol.x # assign initial guess as vector that creates physical reaction
        
        return x0