'''
Run single kinetic cell simulation and save plot and/or data from the simulation

Forms kinetic cell to hold specified parameters (either default or from command line)
    and runs the simulation for the specified heating rates. 

'''
import time
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from options import str_to_list
from options.simulation_options import SimulationOptions
from simulation import create_kinetic_cell


if __name__ == '__main__':
    opts = SimulationOptions().parse()
    kinetic_cell = create_kinetic_cell(opts)

    # Get initial matrices
    reac_coeff, prod_coeff, reac_order, prod_order = kinetic_cell.init_reaction_matrices()
    pre_exp_fac = opts.pre_exp_factors
    e_act = opts.act_energies

    # Build parameters vector to map
    x = np.zeros(len(kinetic_cell.param_types))
    stoic_all = reac_coeff + prod_coeff
    for i, p in enumerate(kinetic_cell.param_types):
        if p[0] == 'preexp':
            x[i] = np.log(pre_exp_fac[p[1]])
        elif p[0] == 'acteng':
            x[i] = np.log(e_act[p[1]])
        elif p[0] == 'stoic':
            x[i] = stoic_all[p[1], kinetic_cell.comp_names.index(p[2])]

    # Map parameters to calcualte material properties
    reac_coeff, prod_coeff, reac_order, prod_order, e_act, pre_exp_fac = kinetic_cell.map_parameters(x)
    kinetic_cell.print_reaction(x)
    
    # Run simulations
    heating_rates = sorted([float(hr) for hr in str_to_list(opts.heating_rates)])

    colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']
    plt.figure()
    for i, hr in enumerate(heating_rates):
        print('Running heating rate {} C/min...'.format(hr))

        # Form initial condition and heating schedule dictionaries
        IC = {'O2': opts.O2_con_in, 
                'Temp': opts.T0,
                'Oil': opts.oil_sat}
        
        Tspan = [float(t) for t in str_to_list(opts.Tspan)]
        Time = np.arange(Tspan[0], Tspan[1])
        Temp = np.minimum(IC['Temp'] + hr*Time, opts.max_temp)
        heating_data = {'Time': Time, 'Temp': Temp}

        # Run simulation
        t, y_dict = kinetic_cell.run_RTO_simulation(REAC_COEFFS=reac_coeff, PROD_COEFFS=prod_coeff, 
                                                    REAC_ORDERS=reac_order, PROD_ORDERS=prod_order, 
                                                    ACT_ENERGY=e_act, PREEXP_FAC=pre_exp_fac,
                                                    HEATING_DATA=heating_data, IC=IC)
        y_dict = {'Time': t, 'O2': np.maximum(IC['O2'] - y_dict['O2'], 0), 'CO2': y_dict['CO2'], 'Temp': y_dict['Temp']}

        # Plot results
        plt.plot(y_dict['Time'], 100*y_dict['O2'], colors[i], label='{} C/min'.format(hr))

        # Save data
        df_out = pd.DataFrame()
        df_out['Time'] = 60*y_dict['Time']  # convert time to seconds
        df_out['O2'] = 100*(IC['O2'] - y_dict['O2'])  # convert back to %mol O2
        df_out['CO2'] = 100*y_dict['CO2']  # convert to %mol
        df_out['Temperature'] = y_dict['Temp']
        df_out.to_excel(os.path.join(kinetic_cell.results_dir, '{}C_min.xls'.format(hr)))

    plt.xlabel('Time [min]')
    plt.ylabel(r'$O_2$ Consumption [% mol]')
    plt.legend()
    plt.savefig(os.path.join(kinetic_cell.results_dir, 'O2_consumption.png'))