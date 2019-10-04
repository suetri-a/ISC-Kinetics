# Utilities for interfacing with CMG-STARS
import numpy as np
import os
import importlib
from .rto_base import RtoBase


def get_vkc_model(model_name):

    model_filename = "rto." + model_name + "_vkc"
    modellib = importlib.import_module(model_filename)
    
    model = None
    target_modellib = model_name.replace('_', '') + 'vkc'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_modellib.lower() and issubclass(cls, RtoBase):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of RtoBase with class name that matches %s in lowercase." % (model_filename, target_modellib))
        # exit(0)

    return model