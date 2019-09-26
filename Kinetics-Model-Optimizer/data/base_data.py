import autograd.numpy as np 
import pandas as pd
from abc import ABC, abstractmethod

class BaseData(ABC):

    @staticmethod
    def modify_cmd_options(parser):
        parser.add_argument('--data_dir', type=str, default='data/', help='location of experimental data')
        return parser


    def __init__(self, opts):
        self.data_load(opts)


    @abstractmethod
    def get_O2_data(self):
        '''
        Return O2 consumption data retrieved from the data file

        '''

        pass


    @abstractmethod
    def get_T_data(self):
        '''
        Return temperature data from 

        '''

        pass


    @abstractmethod
    def data_load(self, opts):
        '''
        Parse the input data file 

        '''

        pass