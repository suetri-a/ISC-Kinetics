import numpy as np
import matplotlib.pyplot as plt
import os, warnings
import cvxpy as cvx
from scipy.integrate import cumtrapz
from scipy.optimize import approx_fprime
from sklearn.linear_model import LinearRegression

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


#### Isoconversional analysis functions

def isoconversional_analysis(heating_data, corrected=True):
    '''
    Calculate isoconversional analysis using Friedman method. 

    Inputs:
        heating_data - dictionary of dictionaries with keys [heating rate] and 
            values dictionaries with keys 'Time', 'O2', 'CO2', and 'Temp'
        corrected - option to discard data after HTO peak 
    
    '''
    R = 8.3145
    if corrected:
        maxes = []
        for hr in sorted(heating_data.keys()):
            O2_conversion = cumtrapz(heating_data[hr]['O2'], x=heating_data[hr]['Time'], initial=0.0)
            O2_conversion /= O2_conversion[-1]
            max_ind = np.argmax(heating_data[hr]['O2'])
            maxes.append(O2_conversion[max_ind])
        max_conv = np.mean(maxes)
    else:
        max_conv = 0.99
    
    conv_grid = np.linspace(0.01,max_conv,200)
    
    O2_eact, O2_rorder, O2_preexp = [], [], []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        O2_conv_dict = {}
        for hr in heating_data.keys():
            O2_conversion = cumtrapz(heating_data[hr]['O2'], x=heating_data[hr]['Time'], initial=0.0)
            O2_conv_dict[hr] = O2_conversion / O2_conversion[-1]
        
        for i in range(conv_grid.shape[0]):
            
            model = LinearRegression()
            conv = conv_grid[i]
            O2_temps = [np.interp(conv, O2_conv_dict[hr], -1/heating_data[hr]['Temp']/R) for hr in sorted(heating_data.keys())]
            dO2_convs = [np.interp(conv, O2_conv_dict[hr], np.log(heating_data[hr]['O2'])) for hr in sorted(heating_data.keys())]

            try:
                model.fit(np.column_stack((O2_temps, np.log(conv)*np.ones_like(O2_temps))), dO2_convs)
                O2_eact.append(model.coef_[0])
                O2_rorder.append(model.coef_[1])
                O2_preexp.append(model.intercept_)
            except:
                O2_eact.append(np.nan)
                O2_rorder.append(np.nan)
                O2_preexp.append(np.nan)

    
    return conv_grid, O2_eact, O2_rorder, O2_preexp


# Numerical functions
def numerical_hessian(x, func):
    '''
    Compute hessian of scalar-valued function func at point x

    Inputs:
        x - vector at which to evaluate numerical Hessian
        func - scalar valued function to compute the Hessian
    
    Returns:
        H - numerical Hessian of func

    '''

    N = x.shape[0]
    H = np.zeros((N,N))

    eps = 1e-8

    def grad_f(x):
        df = approx_fprime(x, func, eps)
        return df

    for i in range(N):
        def grad_f_i(x):
            df = grad_f(x)
            return df[i]
        H[i,:] = approx_fprime(x, grad_f_i, eps)

    return H
