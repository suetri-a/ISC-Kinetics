import numpy as np
import os
import pickle
from joblib import Parallel, delayed
import tikzplotlib as tikz
import matplotlib.pyplot as plt
import corner
from options.optimization_options import OptimizationOptions
from simulation import create_kinetic_cell
from data import create_data_cell
from optimization import create_optimizer
from SALib.analyze import delta

def write_tex_file(tikz_code, fname, tiny_text = False):
    
    # Fix legend font size
    if tiny_text:
        ind = tikz_code.find('tick align') - 3
        tikz_code = tikz_code[:ind] + ', font=\\footnotesize' + tikz_code[ind:]

    print('''\\documentclass[crop,tikz]{{standalone}}
    \\usepackage{{pgfplots}}
    \\begin{{document}}

     {}

     \\end{{document}}'''.format(tikz_code), file = open(fname, 'w'))


if __name__ == '__main__':
    
    # Set random seed
    np.random.seed(999)

    opts = OptimizationOptions().parse()
    opts.load_from_saved = True

    # Load data, kinetic cell, and optimizer
    data_cell = create_data_cell(opts)
    kinetic_cell = create_kinetic_cell(opts)
    optimizer = create_optimizer(kinetic_cell, data_cell, opts)

    # Build figures directory if it doesn't exist
    figs_dir = 'figures'
    if not os.path.exists(figs_dir):
        os.mkdir(figs_dir)
    
    # create dir to output figures for specific model
    if not os.path.exists(os.path.join(figs_dir, opts.name)):
        os.mkdir(os.path.join(figs_dir, opts.name))

    # Load optimal parameters vector
    x = np.load(os.path.join('results', opts.name, 'load_dir', 'optimal_params.npy'))

    # Load samples from uncertainty analysis
    if opts.run_uncertainty:
        samples = np.load(os.path.join('results', opts.name, 'uncertainty_analysis', 'samples.npy'))
        Y = np.load(os.path.join('results', opts.name, 'uncertainty_analysis', 'Y.npy'))

        samples = samples.reshape((-1, x.shape[0]))
        Y = Y.reshape((-1, 1))

    # #####################################################################################################
    # ####################################### CALIBRATION OVERLAY #########################################
    # #####################################################################################################
    
    # print("Running RTO overlay figure...")

    # colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']
    # plt.figure()

    # data_dicts = {}
    # y_dicts = {}

    # for i, hr in enumerate(sorted(data_cell.heating_rates)):
    #     data_dict = data_cell.heating_data[hr]
    #     Oil_con_init = optimizer.compute_init_oil_sat(x, data_dict)
    #     heating_data = {'Time': data_dict['Time'], 'Temp': data_dict['Temp']}
    #     IC = {'Temp': data_dict['Temp'][0], 'O2': data_dict['O2_con_in'], 'Oil': Oil_con_init}

    #     # Plot experimental data
    #     plt.plot(data_dict['Time'], 100 * data_dict['O2'], colors[i]+'-', label=str(hr)+' C/min')
    #     y_dict = kinetic_cell.get_rto_data(x, heating_data, IC) 
    #     plt.plot(y_dict['Time'], 100 * y_dict['O2'], colors[i]+'--')

    #     data_dicts[hr] = data_dict
    #     y_dicts[hr] = y_dict
    
    # plt.xlabel('Time [min]')
    # plt.ylabel(r'$O_2$ consumption [% mol]')
    # plt.legend()
    # tikz_code = tikz.get_tikz_code()
    # write_tex_file(tikz_code, os.path.join(figs_dir, opts.name, 'rto_overlay_{}_{}.tex'.format(opts.reaction_model, opts.dataset)))
    # plt.savefig(os.path.join(figs_dir, opts.name, 'rto_overlay_{}_{}.png'.format(opts.reaction_model, opts.dataset)))

    # # Save code from running overlay
    # with open(os.path.join(figs_dir, opts.name, 'data_{}_{}.pkl'.format(opts.reaction_model, opts.dataset)), 'wb') as f:
    #     pickle.dump(data_dicts, f)
    
    # with open(os.path.join(figs_dir, opts.name, 'y_{}_{}.pkl'.format(opts.reaction_model, opts.dataset)), 'wb') as f:
    #     pickle.dump(y_dicts, f)


    # #####################################################################################################
    # ####################################### SENSITIVITY ANALYSIS ########################################
    # #####################################################################################################
    
    # if opts.run_sensitivity:
    #     print("Running sensitivity analysis...")
    #     # # Marginals Plot
    #     # figure = corner.corner(samples)
    #     # tikz.save(os.path.join(figs_dir, opts.name, 'marginals.tex'))
    #     # figure.savefig(os.path.join(figs_dir, opts.name, 'marginals.png'))
    #     # plt.close()
        
    # if opts.run_sensitivity:
    #     param_names = []
    #     bnds = []
    #     for i, p in enumerate(kinetic_cell.param_types):
    #         if p[0] == 'preexp':
    #             param_names.append('A rxn {}'.format(str(p[1]+1)))
    #             bnds.append([x[i]-1e0, x[i]+1e0])
    #         elif p[0] == 'acteng':
    #             param_names.append('Ea rxn {}'.format(str(p[1]+1)))
    #             bnds.append([x[i]-1e0, x[i]+1e0])
    #         elif p[0] == 'stoic':
    #             param_names.append('{} rxn {}'.format(p[2], str(p[1]+1)))
    #             bnds.append([np.maximum(x[i]-0.5, 1e-3), np.minimum(x[i]+0.5,100.0)])

    #     problem = {
    #         'num_vars': len(kinetic_cell.param_types),
    #         'names': param_names,
    #         'bounds': bnds
    #     }

    #     # Throw out nan and inf values
    #     if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
    #         Y, samples = Y[~np.isnan(Y)], samples[np.where(~np.isnan(Y)),:]
    #         Y, samples = Y[~np.isinf(Y)], samples[np.where(~np.isinf(Y)),:]

    #     # Adjust dimension of samples
    #     D = len(kinetic_cell.param_types)
    #     if Y.shape[0] % (D+1) != 0:        
    #         ind_remainder = Y.shape[0] % (D+1)
    #         Y = Y[:-ind_remainder]
    #         samples = samples[:-ind_remainder]

    #     # Run sensitivity analysis and savel
    #     Si = delta.analyze(problem, samples, Y, print_to_console=True)

    #     # Create tornado plot of metrics
    #     plt.rcdefaults()
    #     fig, ax = plt.subplots()
    #     ax.barh(np.arange(len(problem['names'])), Si['delta'], xerr=Si['delta_conf'], align='center')
    #     ax.set_yticks(np.arange(len(problem['names'])))
    #     ax.set_yticklabels(problem['names'])
    #     ax.invert_yaxis() 
    #     ax.set_xlabel('Sensitivity')
    #     ax.tick_params(labelsize=8)
    #     ax.set_title('Sensitivity plot of variables')
    #     tikz.save(os.path.join(figs_dir, opts.name, 'delta_mim.tex'))
    #     fig.savefig(os.path.join(figs_dir, opts.name, 'delta_mim.png'))
    #     plt.close()



    ##################################################################################################
    ####################################### STOICHIOMETRY CI'S #######################################
    ##################################################################################################
    if opts.run_uncertainty:
        print("Running uncertainty quantification...")

    percentiles = [5, 95]
    if opts.run_uncertainty and any(not os.path.exists(os.path.join('results', opts.name, 'uncertainty_analysis', 'rxn_{}_percentile.txt'.format(p))) for p in percentiles):
        print("Calculating reaction confidence intervals...")
        rxn_params_all = Parallel()(delayed(kinetic_cell.map_parameters)(samples[i,:]) for i in range(samples.shape[0]))
        reac_coeffs_all, prod_coeffs_all, _, _, act_energy_all, preexp_fac_all = zip(*rxn_params_all)
        reac_coeffs_all = np.stack(reac_coeffs_all, axis=0) 
        prod_coeffs_all = np.stack(prod_coeffs_all, axis=0)

        np.save(os.path.join(kinetic_cell.results_dir, 'uncertainty_analysis', 'reac_coeffs.npy'), reac_coeffs_all)
        np.save(os.path.join(kinetic_cell.results_dir, 'uncertainty_analysis', 'prod_coeffs.npy'), prod_coeffs_all)
        np.save(os.path.join(kinetic_cell.results_dir, 'uncertainty_analysis', 'act_energy.npy'), act_energy_all)
        np.save(os.path.join(kinetic_cell.results_dir, 'uncertainty_analysis', 'pre_exp.npy'), preexp_fac_all)
        np.save(os.path.join(kinetic_cell.results_dir, 'uncertainty_analysis', 'comp_names.npy'), kinetic_cell.comp_names)

        act_energy_all = np.stack(act_energy_all, axis=0)
        preexp_fac_all = np.stack(preexp_fac_all, axis=0)

        
        for p in percentiles:
            reac_coeffs = np.percentile(reac_coeffs_all, p, axis=0)
            prod_coeffs = np.percentile(prod_coeffs_all, p, axis=0)
            act_energy = np.percentile(act_energy_all, p, axis=0)
            preexp_fac = np.percentile(preexp_fac_all, p, axis=0)

            message = '| ---------------------------- Reaction - {}% percentile ------------------------------------- | ----- A ---- | ----- E ---- |\n'.format(p)

            for i in range(kinetic_cell.num_rxns):
                reac_str = ''
                
                # Print reactants
                for j, c in enumerate(kinetic_cell.comp_names):
                    if c in kinetic_cell.reac_names[i]:
                        reac_str += str(np.around(reac_coeffs[i,j], decimals=3)) + ' '
                        reac_str += c + ' + '
                
                reac_str = reac_str[:-3] + ' -> '
                
                prod_str = ''
                
                # Print products
                for j, c in enumerate(kinetic_cell.comp_names):
                    if c in kinetic_cell.prod_names[i]:
                        prod_str += str(np.around(prod_coeffs[i,j], decimals=3)) + ' '
                        prod_str += c + ' + '
                prod_str = prod_str[:-3]

                message += '{:>30}{:<50}{:>15}{:>15}\n'.format(reac_str, prod_str, str(np.around(preexp_fac[i],decimals=3)), str(np.around(act_energy[i],decimals=3)))
            
            with open(os.path.join('results', opts.name, 'uncertainty_analysis', 'rxn_{}_percentile.txt'.format(p)), 'w') as f:
                print(message, file=f)
    
    exit()

    ########################################################################################################
    ####################################### LOWEST HEATING RATE CI'S #######################################
    ########################################################################################################

    if opts.run_uncertainty:
        print("Plotting O2 consumption intervals...")

        # Loop over heating rates and plot
        colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']
        fig, ax = plt.subplots()

        for i, hr in enumerate(sorted(data_cell.heating_rates)):
            print(hr)
            data_dict = data_cell.heating_data[hr]
            heating_data = {'Time': data_dict['Time'], 'Temp': data_dict['Temp']}

            if os.path.exists(os.path.join('results', opts.name, 'uncertainty_analysis', 'O2_CI_data_{}.npy'.format(hr))):
                print('Loading O2 CI Data...')
                O2_samples = np.load(os.path.join('results', opts.name, 'uncertainty_analysis', 'O2_CI_data_{}.npy'.format(hr)))

            else:
                # only consider last 100 samples
                print('No O2 CI data found. Running simulations...')
                N = 100
                samples_hr = samples[-N:,:]
                Y_hr = Y[-N:]

                def sim_fun(z):
                    IC = {'Temp': data_dict['Temp'][0], 
                            'O2': data_dict['O2_con_in'], 
                            'Oil': optimizer.compute_init_oil_sat(z, data_dict)}
                    y_dict = kinetic_cell.get_rto_data(z, heating_data, IC)
                    y_out = np.interp(data_dict['Time'], y_dict['Time'], 100*y_dict['O2'])
                    return y_out
                
                O2_samples = Parallel(n_jobs=4)(delayed(sim_fun)(np.squeeze(samples_hr[i,:])) for i in range(N))
                O2_samples = np.stack(O2_samples, axis=0)
                np.save(os.path.join('results', opts.name, 'uncertainty_analysis', 'O2_CI_data_{}.npy'.format(hr)), O2_samples)
            
            # Build and save plot
            print('Building O2 consumption CI plot...')
            IC = {'Temp': data_dict['Temp'][0], 
                    'O2': data_dict['O2_con_in'], 
                    'Oil': optimizer.compute_init_oil_sat(x, data_dict)}
            y_dict = kinetic_cell.get_rto_data(x, heating_data, IC) 
            ci_5 = np.percentile(O2_samples, 5, axis=0)
            ci_95 = np.percentile(O2_samples, 95, axis=0)

            ax.plot(data_dict['Time'], 100*data_dict['O2'], colors[i]+'-', linewidth=2, label='{} C/min'.format(hr))
            ax.plot(y_dict['Time'], 100*y_dict['O2'], colors[i]+'--')
            ax.fill_between(data_dict['Time'], ci_5, ci_95, color=colors[i], alpha=.1)


        ax.set_xlabel('Time')
        ax.set_ylabel(r'$O_2$ Consumption [% mol]')
        ax.legend()
        tikz_code = tikz.get_tikz_code()
        write_tex_file(tikz_code, os.path.join(figs_dir, opts.name, 'O2_confidence_interval.tex'))
        plt.savefig(os.path.join(figs_dir, opts.name, 'O2_confidence_interval.png'))

