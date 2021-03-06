#!/usr/bin/env python
import numpy as np
from .utils import get_maker
from . import utils
from . import medsmakers
import fitsio
import os,sys
import time
from .shearmakers import get_shear_maker

class SimpSimMaker(dict):
    def __init__(self,conf,num_seeds=1,seed_index=0):
        self.global_start_time = time.time()
        self.update(conf)

        self.set_defaults(num_seeds,seed_index)
        self.setup_shears()

        if not self['silent']:
            import pprint
            pprint.pprint(self)
            sys.stdout.flush()
        
        self.set_extra_and_percutout_data()
        self.set_seeds()
        self.set_galaxy_psf_makers()
        
    def set_defaults(self,num_seeds,seed_index):
        self['silent'] = self.get('silent',True)
        self['num_seeds'] = self.get('num_seeds',num_seeds)
        self['seed_index'] = self.get('seed_index',seed_index)
        self['seed_fmt'] = self.get('seed_fmt','%06d')
        self['Ngals'] = self.get('Ngals',1)
        
        if 'sizes' not in self:
            self['sizes'] = utils.get_fft_sizes(min_size=self['min_size'],max_size=self['max_size'])        

        self['output_base'] = self.get('output_base','')
        if len(self['output_base']) > 0 and self['output_base'][-1] != '_':
            self['output_base'] += '_'

    def setup_shears(self):
        if 'shear' not in self:
            print "no shear specified"
            shearpdf = None
        else:
            shearpdf = get_shear_maker(self['shear'])
            print "loaded shearpdf:",shearpdf
            
        self.shearpdf = shearpdf

    def set_extra_and_percutout_data(self):
        self.extra_data = self.get('extra_data',[])
        self.extra_percutout_data = self.get('extra_percutout_data',[])
        self.extra_percutout_data = [('psf_id','i8')]

    def set_seeds(self):
        self.rng = np.random.RandomState(self['global_seed'])
        self.galaxy_seeds = self.rng.choice(10000000,size=self['num_seeds'],replace=False)
        self.psf_seeds = self.rng.choice(10000000,size=self['num_seeds'],replace=False)
        
    def set_galaxy_psf_makers(self):
        if not self['silent']:
            print "getting galaxy maker..."
            sys.stdout.flush()

        self.galaxy_maker = get_maker(self['galaxymaker']['type'])        
        self.galaxy_maker = self.galaxy_maker(seed=self.galaxy_seeds[self['seed_index']],**self)
        self.extra_data.extend(self.galaxy_maker.get_extra_data_dtype())
        self.extra_percutout_data.extend(self.galaxy_maker.get_extra_percutout_data_dtype())

        if not self['silent']:
            print "getting psf maker..."
            sys.stdout.flush()
            
        self.psf_maker = get_maker(self['psfmaker']['type'])
        self.psf_maker = self.psf_maker(seed=self.psf_seeds[self['seed_index']],**self)
        
    def get_shear(self):
        """
        Get a shear dict.  If no shear was specified return None

        The shear dict contains a ngmix.Shape object in the 'shear' field,
        as well as further information such as 'shear_index' in the 'meta' field
        """
        if self.shearpdf is None:
            return {'shear':None,
                    'meta':{'shear_index':-1}}
        else:
            return self.shearpdf.sample()


    def get_object(self):
        # get PSF
        psf = self.psf_maker.get_psf()

        # get gal
        shdict = self.get_shear()
        gal = self.galaxy_maker.get_galaxy(psf=psf, shear=shdict['shear'])
        gal['extra_data']['shear_meta'] = shdict['meta']

        # recenter
        drow = gal['row'] - gal.image.shape[0]/2.0
        dcol = gal['col'] - gal.image.shape[1]/2.0
        psf = self.psf_maker.get_psf(psf=psf,shift=[dcol,drow]) # swap for input to dx,dy in galsim
        
        # set the psf
        gal.psf = psf
        
        return gal
    
    def make_meds(self):
        outputbase = self['output_base']

        psfs = []
        mm = medsmakers.MemoryMEDSMaker(extra_data=self.extra_data,extra_percutout_data=self.extra_percutout_data)

        if not self['silent']:
            print "making galaxies..."
            sys.stdout.flush()
            
        for i in xrange(self['Ngals']):
            if not self['silent']:
                print "gal:",i
                sys.stdout.flush()

            if i == 1:
                self.start_time = time.time()
                
            gal = self.get_object()
            seg = np.zeros_like(gal.image,dtype='i4')
            seg[:,:] = i+1
            row,col = gal['row'],gal['col']

            # append the psf
            psfs.append((i,gal.psf.image.copy()))
            psf_size = gal.psf.image.shape[0]

            # we want this copied to the meds object_data extension, so add it
            # directly to objinfo (gal['extra_data'] is also added later but is
            # not copied to the meds file)
            shear_index = gal['extra_data']['shear_meta']['shear_index']
            
            # put it into meds
            objinfo = dict(id=i,
                           number=i+1,
                           orig_row=np.array([-99,row]),
                           orig_col=[-99,col],
                           orig_start_row=[-99,0],
                           orig_start_col=[-99,0],
                           dudrow=[-99,gal['pixel_scale']],
                           dudcol=[-99,0.0],
                           dvdrow=[-99,0.0],
                           dvdcol=[-99,gal['pixel_scale']],
                           cutout_row=[-99,row],
                           cutout_col=[-99,col],
                           shear_index=shear_index)

            objinfo.update(gal['extra_data'])

            pdata = {}
            for nm,tp in self.extra_percutout_data:
                pdata[nm] = [-99.0]
            pdata['psf_id'].append(i)
            for nm in gal['extra_percutout_data'].keys():
                assert nm in pdata,"Percutout data %s not found!" % nm
                pdata[nm].extend(gal['extra_percutout_data'][nm])
            objinfo.update(pdata)

            mm.add_object(objinfo, \
                          [np.zeros_like(gal.image),gal.image], \
                          [np.zeros_like(gal.weight),gal.weight], \
                          [np.zeros_like(seg),seg])
            
            del gal
            del objinfo
            del seg
            del row
            del col
            del pdata
            del nm
            del tp

        if not self['silent']:
            print "writing files..."
            sys.stdout.flush()
            
        tail = '%s.fits' % self['seed_fmt']
        tail = tail % self['seed_index']
        mfname = outputbase+'meds'+tail
        mm.write(mfname)
        mm.fpack()
        os.remove(mfname)
        
        psfs = np.array(psfs,dtype=[('psf_id','i8'),('psf_im','f8',(psf_size,psf_size))])
        pfname = outputbase+'psf'+tail
        fitsio.write(pfname,psfs,clobber=True)

        self.end_time = time.time()
        if not self['silent']:
            tt = self.end_time-self.global_start_time
            print 'sim took %f seconds' % tt            
            tpg = (self.end_time-self.start_time)/(self['Ngals'] - 1.0)
            ohead = tt - tpg*self['Ngals']
            print 'init took %s seconds' % ohead
            print 'used %f seconds per galaxy' % tpg
            sys.stdout.flush()


