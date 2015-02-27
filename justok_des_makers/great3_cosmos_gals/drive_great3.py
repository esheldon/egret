#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import fitsio

import galsim
from noise import PlaceholderNoiseBuilder
#from psf import ConstPSFBuilder
from galaxies import COSMOSGalaxyBuilder
#import constants

"""
attempting to hack out just part of great3 needed for 
making HST galaxies
"""

real_gal = True
#obs_type = "ground"
#shear_type = "constant"
#multiepoch = False
gal_dir = "../../great3_data"
preload = False
#variable_psf = False
#epoch_index = 0
#field_index = 0
#subfield_index = 0
noise_mult = 1.0
optics_psf_fwhm = 0.1
atmos_psf_fwhm = 0.1 
tot_psf_fwhm = np.sqrt(optics_psf_fwhm**2 + atmos_psf_fwhm**2)
psf_sigma = tot_psf_fwhm/(2.0*np.sqrt(2.0*np.log(2.0)))
g1 = 0.0
g2 = 0.0

#hacking!
#constants.pixel_scale[obs_type][multiepoch] = 0.05

rng = galsim.UniformDeviate(123456)

cgb = COSMOSGalaxyBuilder(real_gal,gal_dir,preload)
nb = PlaceholderNoiseBuilder()

cgb_params = cgb.generateSubfieldParameters()
cpsfb_params = {}
cpsfb_params["atmos_psf_fwhm"] = tot_psf_fwhm

#make catalog
dlist = [('index','i8'),('x','i8'),('y','i8'),('n_epochs','i4')]
dlist.extend(cgb_params['schema'])
cat = np.zeros(2000,dtype=dlist)
cat['n_epochs'][:] = 1

#make gals
if True:
    import constants
    obs_type = "ground"
    multiepoch = False
    pixel_scale = constants.pixel_scale[obs_type][multiepoch]
    xsize = constants.xsize[obs_type][multiepoch]
    ysize = constants.ysize[obs_type][multiepoch]
    max_xsize = xsize + 2*(constants.centroid_shift_max + constants.epoch_shift_max)
    max_ysize = ysize + 2*(constants.centroid_shift_max + constants.epoch_shift_max)
else:
    pixel_scale = 0.01 #constants.pixel_scale[obs_type][multiepoch]
    #xsize = constants.xsize[obs_type][multiepoch]
    #ysize = constants.ysize[obs_type][multiepoch]
    max_xsize = 48 #xsize + 2*(constants.centroid_shift_max + constants.epoch_shift_max)
    max_ysize = 48 #ysize + 2*(constants.centroid_shift_max + constants.epoch_shift_max)
pixel = galsim.Pixel(pixel_scale)
params = galsim.GSParams(maximum_fft_size=10240)

psf = galsim.Gaussian(sigma=psf_sigma, flux=1.0)

d = fitsio.read(os.path.join(gal_dir,'real_galaxy_catalog_23.5.fits'),lower=True)

for i,record in enumerate(cat):
    print 'doing galaxy:',i
    nb_params = nb.generateEpochParameters(rng,record['n_epochs'],cpsfb_params["atmos_psf_fwhm"],noise_mult)
    cgb.generateCatalog(rng,[record],cgb_params,nb.typical_variance,noise_mult,seeing=cpsfb_params["atmos_psf_fwhm"])
    print '    record:',record
    
    galaxy = cgb.makeGalSimObject(record, max_xsize, max_ysize, pixel_scale, rng)
    print 'did gal'
    
    #galaxy.applyLensing(g1=g1, g2=g2, mu=1.0/np.abs(1.0 - g1*g1 - g2*g2))
    final = galsim.Convolve([psf, pixel, galaxy], gsparams=params)

    #originally had normalization = 'f' - I think newer versions of galsim do this by default
    galim = final.draw(scale=pixel_scale)

    """
    if hasattr(final,'noise'):
        current_var = final.noise.applyWhiteningTo(galim)
    else:
        current_var = 0.0
    
    nb_params = nb.generateEpochParameters(rng,record['n_epochs'],cpsfb_params["atmos_psf_fwhm"],noise_mult)
    nb.addNoise(rng,nb_params,galim,current_var)
    """

    plt.close('all')

    #get the cosmos image
    q, = np.where(d['ident'] == cat['cosmos_ident'][i])
    im = fitsio.read(os.path.join(gal_dir,d['gal_filename'][q][0]).strip(),ext=d['gal_hdu'][q][0])

    #make a side by side
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(im,cmap=plt.cm.Greys_r)
    axs[0].set_aspect('equal')
    axs[1].imshow(galim.array,cmap=plt.cm.Greys_r)
    axs[1].set_aspect('equal')
    plt.show()

    import ipdb
    ipdb.set_trace()
