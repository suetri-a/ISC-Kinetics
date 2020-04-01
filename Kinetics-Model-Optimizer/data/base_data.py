import autograd.numpy as np 
import pandas as pd
import os
from abc import ABC, abstractmethod

class BaseData(ABC):

    @staticmethod
    def modify_cmd_options(parser):
        parser.add_argument('--dataset_dir', type=str, default='datasets/', help='location of experimental data')
        return parser


    def __init__(self, opts):
        # Load data
        self.dataset_dir = os.path.join(opts.data_dir, opts.dataset)
        self.data_load(opts)
        

    @abstractmethod
    def data_load(self, opts):
        '''
        Parse the input data file 

        '''

        pass