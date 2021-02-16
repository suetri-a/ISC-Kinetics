import numpy as np
import os, pickle
from joblib import Parallel, delayed
import tikzplotlib as tikz
import matplotlib.pyplot as plt
from options.optimization_options import OptimizationOptions
from simulation import create_kinetic_cell
from data import create_data_cell
from optimization import create_optimizer

def find_optim_stage(line):
    '''
    Find which stage of optimization from input line
    '''
    line_split = line.split()

    if 'Warm' in line_split and '1)' in line_split:
        optim_stage = 'ws1'
    elif 'Warm' in line_split and '2)' in line_split:
        optim_stage = 'ws2'
    elif 'Warm' in line_split and '3)' in line_split:
        optim_stage = 'ws3'
    else:
        optim_stage = 'optim'

    return optim_stage


def get_model_losses(model_name):
    '''
    Parse optimization log file to produce list of parameters and loss
        values for each step in the optimizations

    '''

    optim_file_lines = open(os.path.join('results', model_name, 'optim_write_file.txt'), 'r').readlines()

    loss_dict = {'ws1': [],  # initialize dict to hold a list of losses for each stage
                'ws2': [],
                'ws3': [],
                'optim': []}
    param_dict = {'ws1': [],
                'ws2': [],
                'ws3': [],
                'optim': []}  # dict to hold list of params for each stage

    # Iterate over lines
    stage = 'ws1'
    
    for line in optim_file_lines:
        lines_split = line.split()
        
        # Assign to losses, parameters, or update the status
        if any([s in lines_split for s in ['Cost:', 'Loss']]): 
            loss_dict[stage].append(float(lines_split[-1]))

        elif 'Pre-exp' in lines_split or 'Activation' in lines_split:
            param_dict[stage].append(np.log(float(lines_split[2])))

        elif 'Coefficient' in lines_split:
            param_dict[stage].append(float(lines_split[1]))

        elif 'Status' in lines_split:
            stage = find_optim_stage(line)

    return loss_dict, param_dict



if __name__ == '__main__':
    
    # Load options
    opts = OptimizationOptions().parse()
    opts.load_from_saved = True

    # Create diretor for figure
    fig_dir = os.path.join('figures', 'convergence')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # Parse losses from log file
    if os.path.exists(os.path.join(fig_dir, 'loss_dict.pkl')):
        with open(os.path.join(fig_dir, 'loss_dict.pkl'), 'rb') as f:
            loss_dict = pickle.load(f)
    else:
        loss_dict, _ = get_model_losses(opts.name)
        with open(os.path.join(fig_dir, 'loss_dict.pkl'), 'wb') as f:
            pickle.dump(loss_dict, f) 
    

    # Warm start stage 1
    plt.figure()
    plt.semilogy(loss_dict['ws1'])
    plt.xlabel('Iterations')
    plt.ylabel(r'Peak-matching cost $C(E_a)$')
    # plt.ylim((1e-3, 1e0))
    tikz.save(os.path.join(fig_dir, 'stage1.tex'))
    plt.savefig(os.path.join(fig_dir, 'stage1.png'))

    # Warm start stage 2
    plt.figure()
    loss_dict['ws2'] = np.array(loss_dict['ws2'])
    plt.semilogy(np.arange(loss_dict['ws2'].shape[0])[loss_dict['ws2']<1e6], 
                loss_dict['ws2'][loss_dict['ws2']<1e6])
    plt.xlabel('Iterations')
    plt.ylabel(r'Start-peaks-end cost $C(\theta)$')
    tikz.save(os.path.join(fig_dir, 'stage2.tex'))
    plt.savefig(os.path.join(fig_dir, 'stage2.png'))


    # Warm start stage 3
    plt.figure()
    loss_dict['ws3'] = np.array(loss_dict['ws3'])
    plt.semilogy(np.arange(loss_dict['ws3'].shape[0])[loss_dict['ws3']<1e6], 
                loss_dict['ws3'][loss_dict['ws3']<1e6])
    plt.xlabel('Iterations')
    plt.ylabel(r'Loss value $C(\theta)$')
    plt.ylim((1e-8, 1e-4))
    tikz.save(os.path.join(fig_dir, 'stage3.tex'))
    plt.savefig(os.path.join(fig_dir, 'stage3.png'))


    # Optimization
    plt.figure()
    loss_dict['optim'] = np.array(loss_dict['optim'])
    plt.semilogy(np.arange(loss_dict['optim'].shape[0])[loss_dict['optim']<1e6], 
                loss_dict['optim'][loss_dict['optim']<1e6])
    plt.xlabel('Iterations')
    plt.ylabel(r'Loss value $\ell(\theta)$')
    plt.ylim((1e-3, 1e-1))
    tikz.save(os.path.join(fig_dir, 'optim.tex'))
    plt.savefig(os.path.join(fig_dir, 'optim.png'))
