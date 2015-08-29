#!/usr/bin/env python

class PSFMaker(object):
    def __init__(self):
        raise NotImplementedError
    
    def make_psf(self):
        raise NotImplementedError
        
    def make_atmos_psf(self):
        raise NotImplementedError

    def make_opt_psf(self):
        raise NotImplementedError
