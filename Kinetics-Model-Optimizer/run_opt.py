'''
Run kinetic cell optimization. 

Forms kinetic cell to hold data (either synthetic or experimental) and
    fits parameters in a target kinetic cell to the data. 

'''
import time
import os
import argparse
import pickle
import numpy as np

from options.optimization_options import OptimizationOptions
from simulation import create_kinetic_cell, create_data_cell
from optimization import create_optimizer

if __name__ == '__main__':
    opts = OptimizationOptions().parse()
    kinetic_cell = create_kinetic_cell(opts)
    data_cell = create_data_cell(opts)
    optimizer = create_optimizer(opts)
    optimizer.optimize_cell()

