#!/usr/bin/env python
import numpy as np
from .utils import get_maker
from . import utils

class SimpSimMaker(dict):
    def __init__(self,conf,global_seed,Nseeds=1,seed_index=0):
        self.update(conf)

        self.set_defaults()
        self.set_seeds()
        self.set_galaxy_psf_makers()    
        
    def set_defaults()
        self['global_seed'] = global_seed
        self['Nseeds'] = self.get('Nseeds',1)
        self['seed_index'] = self.get('seed_index',0)

        if 'sizes' not in self:
            self['sizes'] = utils.get_fft_sizes(min_size=self['min_size'],max_size=self['max_size'])        

        self['extra_data'] = self.get('extra_data',[])
        self['extra_percutout_data'] = self.get('extra_percutout_data',[])
        self['extra_percutout_data'].append([('psf_id','i8')])
            
    def set_galaxy_psf_makers(self):
        self.galaxy_maker = get_maker(self['galaxymaker']['type'])
        self.galaxy_maker = self.galaxy_maker(seed=self.galaxy_seeds[self['seed_index']],**self)
        self['extra_data'].extend(self.galaxy_maker.get_extra_data_dtype())
        self['extra_percutout_data'].extend(self.galaxy_maker.get_extra_percutout_data_dtype())
        
        self.psf_maker = get_maker(self['psfmaker']['type'])
        self.psf_maker = self.psf_maker(seed=self.psf_seeds[self['seed_index']],**self)
        
    def set_seeds(self)
        self.rng = np.random.RandomState(self['seed'])
        self.galaxy_seeds = self.rng.choice(10000000,size=self['Nseeds'],replace=False)
        self.psf_seeds = self.rng.choice(10000000,size=self['Nseeds'],replace=False)

    def get_object(self):
        pass
