import numpy as np
import fitsio

class MEDSMaker(object):
    """
    Object to make MEDS files.
    """

    def __init__(self):
        raise NotImplementedError

    def add_gal(self):
        raise NotImplementedError
    
    def write(self,name):
        raise NotImplementedError

    
