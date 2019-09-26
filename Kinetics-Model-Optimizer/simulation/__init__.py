import argparse
import os
import importlib

from .kinetic_cell_base import KineticCellBase

def cell_from_name(model_name):

    model_filename = "kinetics." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    
    kinetic_cell = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, KineticCellBase):
            kinetic_cell = cls

    if kinetic_cell is None:
        print("In %s.py, there should be a subclass of KineticCellBase with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return kinetic_cell


def create_kinetic_cell(opts):
    model = cell_from_name(opts.kinetics_model)
    kinetic_cell = model(opts)

    print("model [%s] was created" % type(kinetic_cell).__name__)
    
    return kinetic_cell


def create_data_cell(opts):
    model = cell_from_name('data')
    kinetic_cell = model(opts)

    print("model [%s] was created" % type(kinetic_cell).__name__)

    return kinetic_cell


def get_option_setter(cell_name):
    kinetic_cell = cell_from_name(cell_name)
    return kinetic_cell.modify_cmd_options

