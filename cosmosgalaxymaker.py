import numpy as np
import galsim

from .galaxymaker import GalaxyMaker
from .great3_cosmos_gals.galaxies import COSMOSGalaxyBuilder
from .great3_cosmos_gals.noise import PlaceholderNoiseBuilder

class COSMOSGalaxyMaker(GalaxyMaker):
    def __init__(self,seed,cosmos_data,real_galaxy=True,preload=False,**kw):
        self.noise_mult = 1.0
        self.rng = galsim.UniformDeviate(seed)
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
        return galinfo
    
    def _build_catalog_per_seeing(self,seeing,verbose=False,randomly_rotate=True):
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
        nrm = np.sum(self.catalogs[seeing][0]['weight'])
        self.catalogs[seeing][0]['weight'] /= nrm
    
    def get_galaxy(self,seeing,n_epochs,max_xsize,max_ysize,pixel_scale,reuse_catalog=False,verbose=False,randomly_rotate=True):
        """
        Get a galaxy from COSMOS postage stamp a la GREAT3.
        
        In GREAT3, seeing was set to atmospheric PSF FWHM.        
        """
        if reuse_catalog:
            if seeing not in self.catalogs:
                self._build_catalog_per_seeing(seeing,verbose=verbose,randomly_rotate=randomly_rotate)
            
            #now get catalog
            catalog = self.catalogs[seeing][0]
            
            #now draw at random with weights
            # seed numpy.random to get predictable behavior
            np.random.seed(int(self.rng() * 1000000))
            randind = np.random.choice(len(catalog),replace=True,p=self.catalogs[seeing][0]['weight'])
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
            self.cosmosgb.generateCatalog(self.rng,[record],None,nb.typical_variance,self.noise_mult,seeing=seeing,verbose=verbose,randomly_rotate=randomly_rotate)
        
        galaxy = self.cosmosgb.makeGalSimObject(record, max_xsize, max_ysize, pixel_scale, self.rng)
        galinfo = {}
        galinfo['noise_builder'] = nb
        galinfo['noise_builder_params'] = nb_params
        galinfo['info'] = record.copy()
        galinfo['seeing'] = seeing
        
        return galaxy,galinfo

    def finish_galaxy_image(self,galim,final_galaxy,galinfo):
        """
        This routine finishes the galaxies after they have been PSF convolved, etc.
        
        Thus, you might want to do this stuff first...
        
            #galaxy.applyLensing(g1=g1, g2=g2, mu=1.0/np.abs(1.0 - g1*g1 - g2*g2))
            final = galsim.Convolve([psf, pixel, galaxy], gsparams=params)
        
            #originally had normalization = 'f' - I think newer versions of galsim do this by default
            galim = final.draw(scale=pixel_scale)
        """
        if hasattr(final_galaxy,'noise'):
            current_var = final_galaxy.noise.applyWhiteningTo(galim)
        else:
            current_var = 0.0
        
        galinfo['noise_builder'].addNoise(self.rng,galinfo['noise_builder_params'],galim,current_var)

        return galim
