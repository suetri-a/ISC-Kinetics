# Dataset container folder
import argparse
import os
import importlib

from .base_data import BaseData

def cell_from_name(experiment_type):
    
    model_filename = "data." + experiment_type + "_data"
    modellib = importlib.import_module(model_filename)
    
    dataset = None
    target_model_name = experiment_type.replace('_', '') + 'data'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseData):
            dataset = cls

    if dataset is None:
        print("In %s.py, there should be a subclass of KineticCellBase with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return dataset


def create_data_cell(opts):
    model = cell_from_name(opts.experiment_type)
    dataset = model(opts)

    print("dataset [%s] was created" % type(dataset).__name__)

    return dataset


def get_option_setter(cell_name):
    dataset = cell_from_name(cell_name)
    return dataset.modify_cmd_options

