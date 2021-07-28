import numpy as np
import os, pickle
from joblib import Parallel, delayed
import tikzplotlib as tikz
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt

from options.optimization_options import OptimizationOptions
from simulation import create_kinetic_cell
from data import create_data_cell
from optimization import create_optimizer

from build_convergence_figures import get_model_losses, write_tex_file

def parse_params(name):
    '''
    Parse parameters from file
    
    To be used with the 'results_report.txt'

    '''
    
    with open(os.path.join('results', name, 'results_report.txt'), 'r') as f:
        lines = f.readlines()
    params_list = []

    for line in lines:
        line_split = line.split()
        
        if 'Pre-exp' in line_split or 'Activation' in line_split:
            params_list.append(np.log(float(line_split[2])))

        elif 'Coefficient' in line_split:
            params_list.append(float(line_split[1]))
    return params_list


if __name__ == '__main__':
    
    # Set random seed
    np.random.seed(999)
    alphas = [0.05, 0.1, 0.2, 0.4, 1.0]
    fig_dir = os.path.join('figures', 'warm_start')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # Load dataset, optimizer, and kinetic cell
    opts = OptimizationOptions().parse()
    opts.load_from_saved = True

    data_cell = create_data_cell(opts)
    kinetic_cell = create_kinetic_cell(opts)
    optimizer = create_optimizer(kinetic_cell, data_cell, opts)

    # Info for simulations
    hr = min(data_cell.heating_rates)
    data_dict = data_cell.heating_data[hr]
    heating_data = {'Time': data_dict['Time'], 'Temp': data_dict['Temp']}    

    # parse losses and parameters from log file
    if os.path.exists(os.path.join(fig_dir, 'parsed_log_file.pkl')):
        with open(os.path.join(fig_dir, 'parsed_log_file.pkl'), 'rb') as f:
            log_file = pickle.load(f)
            loss_dict, param_dict = log_file
    else:
        loss_dict, param_dict = get_model_losses(opts.name)
        with open(os.path.join(fig_dir, 'parsed_log_file.pkl'), 'wb') as f:
            pickle.dump((loss_dict, param_dict), f)
    
    final_params = np.reshape(np.array(parse_params(opts.name)), (-1, len(kinetic_cell.param_types))).T


    ########################################################################
    ############### WARM START 1 FIGURE - init stoichiometry and Ea est
    print("Creating figure for warm start stage 1...")
    # Check if data exists, and load if it does
    if os.path.exists(os.path.join(fig_dir, 'ws1_data.pkl')):
        with open(os.path.join(fig_dir, 'ws1_data.pkl'),'rb') as f:
            ws1_data = pickle.load(f)

    else:
        print("No simulation data found. Running stage 1 simulations...")
        ws1_params = np.reshape(np.array(param_dict['ws1']), (-1, len(kinetic_cell.param_types))).T  # reshape parameters
        
        ws1_data = []  # log simulation data for each parameters
        for alpha in alphas[:-1]:
            p_ind = int(alpha*ws1_params.shape[1])
            x = np.squeeze(ws1_params[:,p_ind])

            IC = {'Temp': data_dict['Temp'][0], 
                    'O2': data_dict['O2_con_in'], 
                    'Oil': optimizer.compute_init_oil_sat(x, data_dict)}
            
            y_dict = kinetic_cell.get_rto_data(x, heating_data, IC)
            ws1_data.append(y_dict)
        
        IC = {'Temp': data_dict['Temp'][0], 
                'O2': data_dict['O2_con_in'], 
                'Oil': optimizer.compute_init_oil_sat(np.squeeze(final_params[:,0]), data_dict)}
        y_dict = kinetic_cell.get_rto_data(final_params[:,0], heating_data, IC)
        ws1_data.append(y_dict)

        with open(os.path.join(fig_dir, 'ws1_data.pkl'),'wb') as f:
            pickle.dump(ws1_data, f)

    # make figure
    plt.figure()  
    plt.plot(data_dict['Time'], 100*data_dict['O2'], 
            'b', linewidth=2, label='Experimental Data')  # plot experimental data

    labels = [None, None, None, None, 'Stage 1']
    for i, alpha in enumerate(alphas):
        plt.plot(ws1_data[i]['Time'], 100*ws1_data[i]['O2'],
                'b--',
                linewidth=2,
                alpha=alpha,
                label=labels[i])

    plt.xlabel('Time')
    plt.ylabel(r'$O_2$ Consumption [% mol]')
    plt.legend(loc='upper left', bbox_to_anchor=(0.8, 0.97), framealpha=1.0)
    tikz_code = tikz.get_tikz_code()
    write_tex_file(tikz_code, os.path.join('figures', 'warm_start', 'stage1.tex'))
    plt.savefig(os.path.join('figures', 'warm_start', 'stage1.png'))


    ########################################################################
    ############### WARM START 2 FIGURE - start-peaks-end match

    print("Creating figure for warm start stage 2...")
    # Check if data exists, and load if it does
    if os.path.exists(os.path.join(fig_dir, 'ws2_data.pkl')):
        with open(os.path.join(fig_dir, 'ws2_data.pkl'),'rb') as f:
            ws2_data = pickle.load(f)

    else:
        print("No simulation data found. Running stage 2 simulations...")
        ws2_params = np.reshape(np.array(param_dict['ws2']), (-1, len(kinetic_cell.param_types))).T  # reshape parameters
        
        ws2_data = []  # log simulation data for each parameters
        for alpha in alphas[:-1]:
            p_ind = int(alpha*ws2_params.shape[1])
            x = np.squeeze(ws2_params[:,p_ind])

            IC = {'Temp': data_dict['Temp'][0], 
                    'O2': data_dict['O2_con_in'], 
                    'Oil': optimizer.compute_init_oil_sat(x, data_dict)}
            
            y_dict = kinetic_cell.get_rto_data(x, heating_data, IC)
            ws2_data.append(y_dict)
        
        IC = {'Temp': data_dict['Temp'][0], 
                'O2': data_dict['O2_con_in'], 
                'Oil': optimizer.compute_init_oil_sat(np.squeeze(final_params[:,1]), data_dict)}
        y_dict = kinetic_cell.get_rto_data(final_params[:,1], heating_data, IC)
        ws2_data.append(y_dict)

        with open(os.path.join(fig_dir, 'ws2_data.pkl'),'wb') as f:
            pickle.dump(ws2_data, f)

    # make figure
    plt.figure()  
    plt.plot(data_dict['Time'], 100*data_dict['O2'], 
            'b', linewidth=2, label='Experimental Data')  # plot experimental data
    
    s_e_prop = 1e-2
    
    # find and plot data start/end point
    O2_data_start = np.amin(data_dict['Time'][data_dict['O2']>s_e_prop*np.amax(data_dict['O2'])])
    O2_data_start_ind = np.amin(np.where(data_dict['O2']>s_e_prop*np.amax(data_dict['O2'])))
    O2_data_end = np.amax(data_dict['Time'][data_dict['O2']>s_e_prop*np.amax(data_dict['O2'])])
    O2_data_end_ind = np.amax(np.where(data_dict['O2']>s_e_prop*np.amax(data_dict['O2'])))
    plt.scatter([O2_data_start, O2_data_end], 
                [5*np.amax(data_dict['O2']), 5*np.amax(data_dict['O2'])],
                c='g', marker='o', label='Data Endpoints')

    # find and plot peaks in data
    peak_inds_O2_data = np.sort(find_peaks_cwt(data_dict['O2'], np.arange(1,np.round(data_dict['O2'].shape[0]/6),step=1), noise_perc=2))
    peak_inds_O2_data = np.delete(peak_inds_O2_data, np.where(peak_inds_O2_data < O2_data_start_ind)) # throw out early peaks
    peak_inds_O2_data = np.delete(peak_inds_O2_data, np.where(peak_inds_O2_data > O2_data_end_ind)) # throw out late peaks  
    plt.scatter(data_dict['Time'][peak_inds_O2_data], 
                100*data_dict['O2'][peak_inds_O2_data],
                c='r', marker='o', label='Data Peaks')

    labels = [None, None, None, None, 'Stage 2']
    labels_peaks = [None, None, None, None, 'Simulation Peaks']
    labels_ends = [None, None, None, None, 'Simulation Start/End']
    for i, alpha in enumerate(alphas):
        # plot effluence curve
        plt.plot(ws2_data[i]['Time'], 100*ws2_data[i]['O2'],
                'b--',
                linewidth=2,
                alpha=alpha,
                label=labels[i])

        y_dict = ws2_data[i]
        # plot peaks/ends
        O2_sim_start = np.amin(y_dict['Time'][y_dict['O2']>s_e_prop*np.amax(y_dict['O2'])])
        O2_sim_start_ind = np.amin(np.where(y_dict['O2']>s_e_prop*np.amax(y_dict['O2'])))
        O2_sim_end = np.amax(y_dict['Time'][y_dict['O2']>s_e_prop*np.amax(y_dict['O2'])])
        O2_sim_end_ind = np.amax(np.where(y_dict['O2']>s_e_prop*np.amax(y_dict['O2'])))
        
        peak_inds_O2_sim = np.sort(find_peaks_cwt(y_dict['O2'], np.arange(1,np.round(y_dict['O2'].shape[0]/6),step=1), noise_perc=2))
        peak_inds_O2_sim = np.delete(peak_inds_O2_sim, np.where(peak_inds_O2_sim < O2_sim_start_ind))
        peak_inds_O2_sim = np.delete(peak_inds_O2_sim, np.where(peak_inds_O2_sim > O2_sim_end_ind))

        plt.scatter(y_dict['Time'][peak_inds_O2_sim], 
                    100*y_dict['O2'][peak_inds_O2_sim],
                    c='r', marker='+', alpha=alpha,
                    label=labels_peaks[i])
        
        plt.scatter([O2_sim_start, O2_sim_end], 
                    [5*np.amax(data_dict['O2']), 5*np.amax(data_dict['O2'])], 
                    c='g', marker='+', alpha=alpha,
                    label=labels_ends[i])

    plt.xlabel('Time')
    plt.ylabel(r'$O_2$ Consumption [% mol]')
    plt.legend(loc='upper left', bbox_to_anchor=(0.9, 0.97), framealpha=1.0)
    tikz_code = tikz.get_tikz_code()
    write_tex_file(tikz_code, os.path.join('figures', 'warm_start', 'stage2.tex'))
    plt.savefig(os.path.join('figures', 'warm_start', 'stage2.png'))
    

    ########################################################################
    ############### WARM START 3 FIGURE - fitting to lowest heating rate

    print("Creating figure for warm start stage 3...")
    # Check if data exists, and load if it does
    if os.path.exists(os.path.join(fig_dir, 'ws3_data.pkl')):
        with open(os.path.join(fig_dir, 'ws3_data.pkl'),'rb') as f:
            ws3_data = pickle.load(f)

    else:
        print("No simulation data found. Running stage 3 simulations...")
        ws3_params = np.reshape(np.array(param_dict['ws3']), (-1, len(kinetic_cell.param_types))).T  # reshape parameters
        
        ws3_data = []  # log simulation data for each parameters
        for alpha in alphas[:-1]:
            p_ind = int(alpha*ws3_params.shape[1])
            x = np.squeeze(ws3_params[:,p_ind])

            IC = {'Temp': data_dict['Temp'][0], 
                    'O2': data_dict['O2_con_in'], 
                    'Oil': optimizer.compute_init_oil_sat(x, data_dict)}
            
            y_dict = kinetic_cell.get_rto_data(x, heating_data, IC)
            ws3_data.append(y_dict)
        
        IC = {'Temp': data_dict['Temp'][0], 
                'O2': data_dict['O2_con_in'], 
                'Oil': optimizer.compute_init_oil_sat(np.squeeze(final_params[:,2]), data_dict)}
        y_dict = kinetic_cell.get_rto_data(final_params[:,2], heating_data, IC)
        ws3_data.append(y_dict)

        with open(os.path.join(fig_dir, 'ws3_data.pkl'),'wb') as f:
            pickle.dump(ws3_data, f)

    # make figure
    plt.figure()  
    plt.plot(data_dict['Time'], 100*data_dict['O2'], 
            'b', linewidth=2, label='Experimental Data')  # plot experimental data

    labels = [None, None, None, None, 'Stage 3']
    for i, alpha in enumerate(alphas):
        plt.plot(ws3_data[i]['Time'], 100*ws3_data[i]['O2'],
                'b--',
                linewidth=2,
                alpha=alpha,
                label=labels[i])

    plt.xlabel('Time')
    plt.ylabel(r'$O_2$ Consumption [% mol]')
    plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.97), framealpha=1.0)
    tikz_code = tikz.get_tikz_code()
    write_tex_file(tikz_code, os.path.join('figures', 'warm_start', 'stage3.tex'))
    plt.savefig(os.path.join('figures', 'warm_start', 'stage3.png'))