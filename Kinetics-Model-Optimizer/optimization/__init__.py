import argparse
import os
import importlib

from .base_optimizer import BaseOptimizer


def optimizer_from_name(optim_name):

    optimizer_filename = "optimization." + optim_name + "_optimizer"
    optimlib = importlib.import_module(optimizer_filename)
    
    optimizer = None
    target_optimlib = optim_name.replace('_', '') + 'optimizer'
    for name, cls in optimlib.__dict__.items():
        if name.lower() == target_optimlib.lower() and issubclass(cls, BaseOptimizer):
            optimizer = cls

    if optimizer is None:
        print("In %s.py, there should be a subclass of BaseOptimizer with class name that matches %s in lowercase." % (optimizer_filename, target_optimlib))
        exit(0)

    return optimizer


def create_optimizer(kinetic_cell, data_cell, opts):
    optimizer_type = optimizer_from_name(opts.optimizer_type)
    optim = optimizer_type(kinetic_cell, data_cell, opts)

    print("optimizer [%s] was created" % type(optim).__name__)
    
    return optim


def get_option_setter(optim_name):
    optimizer = optimizer_from_name(optim_name)
    return optimizer.modify_cmd_options