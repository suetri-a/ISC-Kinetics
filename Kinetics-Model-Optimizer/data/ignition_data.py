import autograd.numpy as np 
import pandas as pd
from .base_data import BaseData

class IgnitionData(BaseData):

    @staticmethod
    def modify_cmd_options(parser):
        
        parser.add_argument('--dataset_name', type=str, default='synthetic', help='name of dataset to load from \'datasets/Ignition\' directory')

        return parser


    def __init__(self, opts):
        super().__init__(opts)


    def get_O2_data(self):
        '''
        Return O2 consumption

        '''

        return None


    def get_T_data(self):

        return None


    def data_load(self, opts):
        '''
        Parse the input data file for experimental data kinetic cell models

        '''

        pass