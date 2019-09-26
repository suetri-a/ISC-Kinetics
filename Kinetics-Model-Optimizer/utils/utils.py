import numpy as np
import matplotlib.pyplot as plt
import os
import cvxpy as cvx

def make_overlay_plots(kinetic_cell, kc_ground_truth, title = None, save_mode = (False, 'plot.png')):
    '''
    Overlay O2 consumption for kinetic cells

    Inputs:
        kinetic_cell - optimized kinetic cell
        kc_ground_truth

    '''
    kinetic_cell.run_RTO_exps(verbose=False)

    for i, r in enumerate(kinetic_cell.rate_heat):
        plt.plot(kc_ground_truth.time_line/60, kc_ground_truth.consumption_O2[:,i], '--',linewidth=2)
        plt.scatter(kinetic_cell.time_line/60, kinetic_cell.consumption_O2[:,i], s = 0.1)
        
    if title is not None:
        plt.title(title)
    
    plt.xlabel('Time, min')
    plt.ylabel(r'$O_2$ Consumption')
    legend_list = ['{} C/min - Data'.format(np.around(r*60, decimals = 2)) for r in kinetic_cell.rate_heat]
    legend_list += ['{} C/min - Simulated'.format(np.around(r*60, decimals = 2)) for r in kinetic_cell.rate_heat]
    plt.legend(legend_list, loc='center left', bbox_to_anchor=(1, 0.5)) 
    
    if save_mode[0]:
        plt.savefig(save_mode[1])
    else:
        plt.show()
        

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)



#### OPTIONS FOR MAPPING IN SUBSTITUTION METHODS

def solve_const_lineq(A, x, b, constraints):
    '''
    Makes call to cvx to solve a constrained linear system of the form:
        min  ||Ax - b||_F^2
        s.t. f(x) = 0

    Inputs:
        A - matrix (cvx parameter)
        x - solution vector (cvx variable)
        b - vector (cvx parameter)
        constraints - list of constraints for the system
    '''

    obj = cvx.Minimize(cvx.norm(A*x - b)**2)
    prob = cvx.Problem(obj, constraints)
    prob.solve()

    return x.value, prob.value