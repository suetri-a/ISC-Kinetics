'''
Run kinetic cell optimization. 

'''
import time
import os
import argparse
import pickle
import numpy as np

from options.optimization_options import OptimizationOptions
from simulation import create_kinetic_cell
from data import create_data_cell
from optimization import create_optimizer

if __name__ == '__main__':
    
    # Set random seed
    np.random.seed(999)

    # Load optimization options
    opts = OptimizationOptions().parse()

    # Load data container
    data_cell = create_data_cell(opts)
    
    # Initialize kinetic cell to run simulations
    kinetic_cell = create_kinetic_cell(opts)

    # Create optimizer container
    optimizer = create_optimizer(kinetic_cell, data_cell, opts)

    # Optimize reaction
    optimizer.optimize_cell()

