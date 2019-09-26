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

from options.simulation_options import SimulationOptions
from simulation import create_kinetic_cell


if __name__ == '__main__':
    opts = SimulationOptions().parse()
    kinetic_cell = create_kinetic_cell(opts)
    kinetic_cell.run_RTO_experiments()
    kinetic_cell.save_plots()
