import numpy as np 
import os

class Component():
    '''
    Component for writing STARS runfile
    
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        

class Kinetics():
    '''
    Reaction definition for writing STARS runfile
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def print_rxn(self, filename):
        print('** {}'.format(self.NAME), file=filename)
        attrs_list = ['STOREAC', 'STOPROD', 'RORDER', 'FREQFAC', 'EACT', 'RENTH', 'O2PP']
        for attr in attrs_list:
            if hasattr(self, attr):
                print_ln = '*{}'.format(attr) + ' '
                if isinstance(getattr(self, attr), list):
                    print_ln += ' '.join(str(x) + ' ' for x in getattr(self, attr))
                else:
                    print_ln += str(getattr(self, attr)) + ' '
                print(print_ln, file=filename)
        print('\n',file=filename)