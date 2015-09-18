#!/usr/bin/env python
import os
import sys
import numpy as np
import galsim
from .psfmaker import PSFMaker
from ..observation import Observation

class DESPSFMaker(PSFMaker):
    """
    make a DES-like PSF
    
    from esheldon's fork of the great3 code
    blessed by Mike and Aaron to roughly match DES
    
    M. R. Becker 2015
    E. Sheldon 2014
    """
    def __init__(self,seed=None,**kw):
        assert seed is not None,"Random seed must be given in DES psf maker!"

        self.rng = np.random.RandomState(seed)
        self.conf = {}
        self.conf.update(kw)
        
    def make_opt_psf(self,image_size=None,pixel_scale=None):
        """make DES optics PSF"""
        assert image_size is not None,"You must specify an image size!"
        assert pixel_scale is not None,"You must specify a pixel scale!"

        lam_over_diam = 0.036
        obscuration = 0.2
        rms_aberration = 0.26
        
        use_aber = ["defocus", "astig1", "astig2", "coma1", "coma2", "trefoil1", "trefoil2", "spher"]
        n_aber = len(use_aber)
        aber_weights = np.array((0.13, 0.13, 0.14, 0.06, 0.06, 0.05, 0.06, 0.03))
        
        aber_dict = {}
        tmp_vec = np.zeros(n_aber)
        for ind_ab in range(n_aber):
            tmp_vec[ind_ab] = rms_aberration * aber_weights[ind_ab] *\
                              self.rng.normal() / np.sqrt(np.sum(aber_weights**2))
            aber_dict[use_aber[ind_ab]] = tmp_vec[ind_ab]
        
        pad_factor = 1.5
        twoR = 2. * lam_over_diam / (
            0.005 * 0.5 * np.pi * np.pi * (1.-obscuration) )
        image_size_arcsec = image_size * pixel_scale
        if image_size_arcsec < twoR * pad_factor:
            pad_factor = image_size_arcsec / twoR
        
        psf = galsim.OpticalPSF(lam_over_diam,
                                obscuration=obscuration,
                                pad_factor=pad_factor*5.0,
                                suppress_warning=False,
                                max_size=image_size_arcsec,
                                **aber_dict)
        return psf
    
    def make_atmos_psf(self,atmos_psf_fwhm=None):
        """make DES atmos PSF"""
        assert atmos_psf_fwhm is not None,"You must specify atmos_psf_fwhm!"
        min_atmos_psf_e = np.sqrt(1.e-4)
        max_atmos_psf_e = np.sqrt(9.e-4)
        atmos_psf_e = self.rng.uniform()*(max_atmos_psf_e - min_atmos_psf_e) \
                      + min_atmos_psf_e
        atmos_psf_beta = self.rng.uniform()*180.0    
        atmos_psf = galsim.Gaussian(sigma=atmos_psf_fwhm)
        atmos_psf = atmos_psf.shear(e=atmos_psf_e,beta=atmos_psf_beta*galsim.degrees)
        return atmos_psf
    
    def make_psf(self,image_size=None,pixel_scale=None,atmos_psf_fwhm=None,full_output=False):
        """
        make DES PSF
        
        full_output: output all psf,opt_psf,atmos_psf
        """
        assert atmos_psf_fwhm is not None,"You must specify atmos_psf_fwhm!"
        assert image_size is not None,"You must specify an image size!"
        assert pixel_scale is not None,"You must specify a pixel scale!"
        opt_psf = self.make_opt_psf(image_size=image_size,pixel_scale=pixel_scale)
        atmos_psf = self.make_atmos_psf(atmos_psf_fwhm=atmos_psf_fwhm)
        psf = galsim.Convolve(atmos_psf,opt_psf)

        if full_output:
            return psf,opt_psf,atmos_psf
        else:
            return psf

    def get_psf(self,pixel_scale=None,seeing=None,**kwargs):
        """
        produce a DES-like PSF
        
        pixel_scale: pixel scale in arcsec
        seeing: atmos seeing in arcsec
        psfmaker: dict of PSF-specific options
        psf: previous PSF from this function (optional)
            can be used to produce the same PSF model, but rendered with a shift in the center
        shift: shift in arcsec of PSF model center, input as [col,row]
        
        """
        if pixel_scale is None:
            key = 'pixel_scale'
            assert key in self.conf,"You must specify '%s' for the PSF!" % key
            pixel_scale = self.conf['pixel_scale']
            
        if seeing is None:
            key = 'seeing'
            assert key in self.conf,"You must specify '%s' for the PSF!" % key
            seeing = self.conf.get(key)

        assert 'psfmaker' in self.conf,"You must specify '%s' for the PSF!" % 'psfmaker'
        assert 'size' in self.conf['psfmaker'],"You must specify '%s' for the PSF!" % 'size'
        psf_size = self.conf['psfmaker']['size']

        if 'psf' not in kwargs:
            psf = self.make_psf(image_size=psf_size, \
                                pixel_scale=pixel_scale, \
                                atmos_psf_fwhm=seeing, \
                                full_output=False)
        else:
            psf = kwargs['psf']['galsim_object']
        
        if 'shift' in kwargs:
            psf = psf.shift(kwargs['shift'][0],kwargs['shift'][1])
        
        psf_im = psf.drawImage(nx=psf_size,ny=psf_size,scale=pixel_scale)
        
        p = Observation()
        p.image = psf_im.array.copy()
        p['galsim_image'] = psf_im
        p['galsim_object'] = psf
        return p

def test():
    import matplotlib.pyplot as plt
    
    image_size=32
    pixel_scale=0.26
    atmos_psf_fwhm = 0.77
    sfunc=lambda x: np.log(np.abs(x))

    dpm = DESPSFMaker(seed=12345)
    psf,opt_psf,atmos_psf = dpm.make_psf(image_size=image_size,
                                         pixel_scale=pixel_scale,
                                         atmos_psf_fwhm=atmos_psf_fwhm,full_output=True)
    
        
    psfims = [sfunc(opt_psf.draw(scale=pixel_scale/10.0,nx=image_size*10.0,ny=image_size*10.0).array),
              sfunc(atmos_psf.draw(scale=pixel_scale/10.0,nx=image_size*10.0,ny=image_size*10.0).array),
              sfunc(psf.drawImage(scale=pixel_scale,nx=image_size,ny=image_size).array)]
    
    vmin = np.inf
    vmax = -np.inf
    for im in psfims:
        if np.min(im) < vmin:
            vmin = np.min(im)
        if np.max(im) > vmax:
            vmax = np.max(im)
    
    fig,axs = plt.subplots(1,3,figsize=(12.0,4.0))
    
    ax = axs[0]
    xyp = np.arange(image_size*10+1)*pixel_scale/10.0
    psfim = psfims[0]
    ax.imshow(psfim,extent=(0.0,image_size*pixel_scale,0.0,image_size*pixel_scale),vmin=vmin,vmax=vmax,aspect='equal')
    ax.set_title('optics PSF')
    ax.set_ylabel(r'$y\ [{\rm arcsec}]$')
    ax.set_xlabel(r'$x\ [{\rm arcsec}]$')
    
    ax = axs[1]
    xyp = np.arange(image_size*10+1)*pixel_scale/10.0
    psfim = psfims[1]
    ax.imshow(psfim,extent=(0.0,image_size*pixel_scale,0.0,image_size*pixel_scale),vmin=vmin,vmax=vmax,aspect='equal')
    ax.set_title('atmos PSF')
    ax.set_ylabel(r'$y\ [{\rm arcsec}]$')
    ax.set_xlabel(r'$x\ [{\rm arcsec}]$')
    
    ax = axs[2]
    xyp = np.arange(image_size+1)*pixel_scale
    psfim = psfims[2]
    ax.imshow(psfim,extent=(0.0,image_size*pixel_scale,0.0,image_size*pixel_scale),vmin=vmin,vmax=vmax,aspect='equal')
    ax.set_title('observed PSF w/ pixel window')
    ax.set_ylabel(r'$y\ [{\rm arcsec}]$')
    ax.set_xlabel(r'$x\ [{\rm arcsec}]$')
    
    #fig.tight_layout(rect=[0.0,0.03,1.0,0.97])
    
    plt.show()
