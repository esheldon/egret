import numpy as np
import galsim
import os

from .galaxymaker import GalaxyMaker
from .great3_cosmos_gals.galaxies import COSMOSGalaxyBuilder
from .great3_cosmos_gals.noise import PlaceholderNoiseBuilder

class COSMOSGalaxyMaker(GalaxyMaker):
    """
    Returns COSMOS galaxies as if viewed from the ground. 
    
    See the GREAT3 challenge docs for details. Code was pulled from the 
    GREAT3 simulations code base.
    
    You will need the COSMOS data, which can be downloaded from 
    
    http://great3.jb.man.ac.uk/leaderboard/data/public/COSMOS_23.5_training_sample.tar.gz
    http://great3.jb.man.ac.uk/leaderboard/data/public/great3_galaxy_selection_files.tar.gz
    
    Once the data is unpacked, place all files in a single directory and feed this path to 
    the code as `cosmos_dir` below.
    
    Examples:
    
    atmos_seeing = 0.55
    cosmos_dir = '/path/to/data'
    seed = 12345
    cgm = COSMOSGalaxyMaker(seed,cosmos_dir)
    
    # build a catalog
    cgm.build_catalog_for_seeing(seeing)
    
    # now draw from it
    for i in xrange(10):
        galaxy,galinfo = cgm.get_galaxy(seeing,n_epochs,max_size,pixel_scale)
    
    # you can also just draw galaxies at random, but then the catalog is rebuilt each time
    #  which is slow.
    galaxy,galinfo = cgm.get_galaxy(seeing,n_epochs,max_size,pixel_scale)
    
    # if you specify save_catalog=True, then you can skip the building step (the code
    #  will do it internally).
    galaxy,galinfo = cgm.get_galaxy(seeing,n_epochs,max_size,pixel_scale,save_catalog=True)    
    
    # the above gets you a galaxy with no PSF
    # one should then add in PSF and pixel effects and noise if wanted
    # try these methods
    
    # great3 clone
    cgm.apply_psf_and_noise_whiten_ala_great3(...)
    
    # like great3, but no extra noise added
    cgm.apply_psf_and_noise_whiten(...)
    
    """
    def __init__(self,seed,cosmos_data=None,real_galaxy=True,preload=False,**kw):
        if cosmos_data is None:
            cosmos_data = os.environ['GREAT3DATA']

        self.noise_mult = 1.0
        self.rng = galsim.UniformDeviate(seed)
        self.rng_np = np.random.RandomState(int(self.rng() * 1000000))
        self.cosmos_data = cosmos_data
        self.preload = preload
        self.cosmosgb = COSMOSGalaxyBuilder(real_galaxy,cosmos_data,preload=preload)
        self.real_galaxy = real_galaxy
        self.catalog_dtype = self.cosmosgb.generateSubfieldParameters()['schema']
        self.catalog_dtype.append(('n_epochs','i4'))
        self.catalogs = {}

    def get_galaxy_from_info(self,record_in,seeing,n_epochs,max_size,pixel_scale):
        """
        Get a COSMOS galaxy from a specific row in table.
        """
        record = record_in.copy()
        if record['n_epochs'] != n_epochs:
            rat = float(record['n_epochs']/n_epochs)
            record['n_epochs'] = n_epochs
            for tag in ["bulge_flux","disk_flux","flux_rescale"]:
                if tag in record.dtype.names:
                    record[tag] *= rat
        nb = PlaceholderNoiseBuilder()
        nb_params = nb.generateEpochParameters(self.rng,record['n_epochs'],seeing,self.noise_mult)
        galaxy = self.cosmosgb.makeGalSimObject(record, max_size, max_size, pixel_scale, self.rng)
        galinfo = {}
        galinfo['noise_builder'] = nb
        galinfo['noise_builder_params'] = nb_params
        galinfo['info'] = record
        galinfo['seeing'] = seeing
        galinfo['noise'] = np.sqrt(galinfo['noise_builder_params']['variance'])
        return galaxy,galinfo
    
    def build_catalog_for_seeing(self,seeing,verbose=False,randomly_rotate=True):
        """
        Build a galaxy catalog a specific seeing value.
        
        If you build a catalog and then get galaxies with the same seeing value,
        the code will skip subsequent building steps.
        """
        nb = PlaceholderNoiseBuilder()
        nb_params = nb.generateEpochParameters(self.rng,1,seeing,self.noise_mult)
        # NOTE
        # typical_variance is for SE by definition, so just make a typical gal for the seeing and one epoch
        # will handle increased variance for multiple epochs below
        # also will rescale flux comps below
        self.catalogs[seeing] = (self.cosmosgb.generateCatalog(self.rng,None,None,nb.typical_variance, \
                                                               self.noise_mult,seeing=seeing,verbose=verbose, \
                                                               randomly_rotate=randomly_rotate),
                                 nb.typical_variance)
        
    def get_catalog_for_seeing(self,seeing,verbose=False,randomly_rotate=True):
        """
        Get a catalog for a specific seeing value.
        """
        if seeing not in self.catalogs:
            self.build_catalog_for_seeing(seeing,verbose=verbose,randomly_rotate=randomly_rotate)
        return self.catalogs[seeing][0].copy()
        
    def get_galaxy(self,seeing,n_epochs,max_size,pixel_scale,verbose=False, \
                   randomly_rotate=True,save_catalog=False):
        """
        Get a galaxy from COSMOS postage stamp a la GREAT3.
        
        In GREAT3, seeing was set to atmospheric PSF FWHM.        
        """
        if save_catalog or seeing in self.catalogs:
            if seeing not in self.catalogs:
                self.build_catalog_for_seeing(seeing,verbose=verbose,randomly_rotate=randomly_rotate)
            
            #now get catalog
            catalog = self.catalogs[seeing][0]
            Ncosmos = len(catalog)
            
            #now draw at random with weights
            # seed numpy.random to get predictable behavior
            while True:                
                randind = self.rng_np.choice(Ncosmos,replace=True)
                if self.rng_np.uniform() < self.catalogs[seeing][0]['weight'][randind]:
                    break            
            record = catalog[randind].copy()
            record['n_epochs'] = n_epochs
            for tag in ["bulge_flux","disk_flux","flux_rescale"]:
                if tag in record.dtype.names:
                    record[tag] /= n_epochs
            nb = PlaceholderNoiseBuilder()
            nb_params = nb.generateEpochParameters(self.rng,record['n_epochs'],seeing,self.noise_mult)
            assert nb.typical_variance == self.catalogs[seeing][-1]
        else:
            record = np.zeros(1,dtype=self.catalog_dtype)[0]
            record['n_epochs'] = n_epochs
            nb = PlaceholderNoiseBuilder()
            nb_params = nb.generateEpochParameters(self.rng,record['n_epochs'],seeing,self.noise_mult)
            self.cosmosgb.generateCatalog(self.rng,[record],None,nb.typical_variance,self.noise_mult,seeing=seeing, \
                                          verbose=verbose,randomly_rotate=randomly_rotate)
        
        galaxy = self.cosmosgb.makeGalSimObject(record, max_size, max_size, pixel_scale, self.rng)
        galinfo = {}
        galinfo['noise_builder'] = nb
        galinfo['noise_builder_params'] = nb_params
        galinfo['info'] = record.copy()
        galinfo['seeing'] = seeing
        galinfo['noise'] = np.sqrt(galinfo['noise_builder_params']['variance'])
        
        return galaxy,galinfo

    def _get_sub_image(self,galim,max_size):
        curr_bounds = galim.getBounds()
        curr_xsize = curr_bounds.getXMax() - curr_bounds.getXMin()
        curr_ysize = curr_bounds.getYMax() - curr_bounds.getYMin()

        if curr_xsize > max_size or curr_ysize > max_size or curr_ysize != curr_xsize:
            sub_bounds = self._get_sub_bounds(curr_bounds,max_size)
            sub_galim = galim.subImage(sub_bounds)
            return sub_galim
        else:
            return galim
    
    def _get_sub_bounds(self,curr_bounds,max_size):
        xmin = curr_bounds.getXMin()
        xmax = curr_bounds.getXMax()
        curr_xsize = xmax - xmin + 1

        ymin = curr_bounds.getYMin()
        ymax = curr_bounds.getYMax()
        curr_ysize = ymax - ymin + 1
        
        final_size = np.min((curr_xsize,curr_ysize,max_size))
        
        offx = curr_xsize - final_size
        if offx > 0:
            offx = offx//2
            sub_xmin = xmin+offx
            sub_xmax = xmin+offx+final_size
        else:
            sub_xmin = xmin
            sub_xmax = xmax

        offy = curr_ysize - final_size
        if offy > 0:            
            offy = offy//2
            sub_ymin = ymin+offy
            sub_ymax = ymin+offy+final_size
        else:
            sub_ymin = ymin
            sub_ymax = ymax
            
        sub_bounds = galsim.BoundsI(sub_xmin,sub_xmax,sub_ymin,sub_ymax)
        return sub_bounds

    def finish_galaxy_image_ala_great3(self,galim,final_galaxy,galinfo,max_size):
        """
        This routine finishes the galaxies after they have been PSF convolved, etc.
        It adds the requested amount of noise in the galinfo dict to the image 
        taking into account the noise already in the HST image.
        
        You might want to do things like this first:
        
            galaxy.applyLensing(g1=g1, g2=g2, mu=mu)
            final = galsim.Convolve([psf, pixel, galaxy])        
            galim = final.draw(scale=pixel_scale)
        
        Doing the stuff above first matches how the GREAT3 sims were done.
        """
        if hasattr(final_galaxy,'noise'):
            current_var = final_galaxy.noise.applyWhiteningTo(galim)
        else:
            current_var = 0.0
        
        galinfo['noise_builder'].addNoise(self.rng,galinfo['noise_builder_params'],galim,current_var)

        final_galim = self._get_sub_image(galim,max_size)
        
        return final_galim, galinfo['noise_builder_params']['variance']

    def apply_psf_and_noise_whiten_ala_great3(self,galaxy,psf,pixel,galinfo,max_size):
        """
        Automates finishing of galaxies for a psf and pixel a la great3
        
        Shear should be applied already if wanted like this, for example,
        
            galaxy.applyLensing(g1=g1, g2=g2, mu=mu)
        """

        final_galaxy = galsim.Convolve([psf, pixel, galaxy])
        galim = final_galaxy.draw(scale=pixel.getScale())

        return self.finish_galaxy_image_ala_great3(galim,final_galaxy,galinfo,max_size)
    
    def apply_psf_and_noise_whiten(self,galaxy,psf,pixel,galinfo,max_size):
        """
        Automates finishing of galaxies for a psf and pixel, but adds no extra noise.
        
        Shear should be applied already if wanted.
        
        Noise can then be added to the image via (for example)
        
            noise_to_add = np.sqrt(total_noise**2 - current_var)
            noise = galsim.GaussianNoise(rng, sigma=noise_to_add)
            noise.applyTo(final_galim)        
        """

        final_galaxy = galsim.Convolve([psf, pixel, galaxy])
        galim = final_galaxy.draw(scale=pixel.getScale())

        if hasattr(final_galaxy,'noise'):
            current_var = final_galaxy.noise.applyWhiteningTo(galim)
        else:
            current_var = 0.0

        final_galim = self._get_sub_image(galim,max_size)

        return final_galim, current_var
    
        
                 

        
