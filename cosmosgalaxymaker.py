import numpy as np
import galsim

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
    
    #build a catalog
    cgm.build_catalog_for_seeing(seeing)
    
    # now draw from it
    for i in xrange(10):
        galaxy,galinfo = cgm.get_galaxy(seeing,n_epochs,max_xsize,max_ysize,pixel_scale)
    
    # you can also just draw galaxies at random, but then the catalog is rebuilt each time
    # which is slow.
    galaxy,galinfo = cgm.get_galaxy(seeing,n_epochs,max_xsize,max_ysize,pixel_scale)
    
    #if you specify save_catalog=True, then you can skip the building step (the code
    # will do it internally).
    galaxy,galinfo = cgm.get_galaxy(seeing,n_epochs,max_xsize,max_ysize,pixel_scale,save_catalog=True)    
    """
    def __init__(self,seed,cosmos_data,real_galaxy=True,preload=False,**kw):
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

    def get_galaxy_from_info(self,record_in,seeing,n_epochs,max_xsize,max_ysize,pixel_scale):
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
        galaxy = self.cosmosgb.makeGalSimObject(record, max_xsize, max_ysize, pixel_scale, self.rng)
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
        
    def get_galaxy(self,seeing,n_epochs,max_xsize,max_ysize,pixel_scale,verbose=False, \
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
        
        galaxy = self.cosmosgb.makeGalSimObject(record, max_xsize, max_ysize, pixel_scale, self.rng)
        galinfo = {}
        galinfo['noise_builder'] = nb
        galinfo['noise_builder_params'] = nb_params
        galinfo['info'] = record.copy()
        galinfo['seeing'] = seeing
        galinfo['noise'] = np.sqrt(galinfo['noise_builder_params']['variance'])
        
        return galaxy,galinfo

    def finish_galaxy_image(self,galim,final_galaxy,galinfo):
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

        return galim
