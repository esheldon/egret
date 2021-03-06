import numpy as np
import galsim
import os

from .galaxymaker import GalaxyMaker
from .great3_cosmos_gals.galaxies import COSMOSGalaxyBuilder
from .great3_cosmos_gals.noise import PlaceholderNoiseBuilder
from .. import utils
from ..observation import Observation

class GREAT3COSMOSGalaxyMaker(GalaxyMaker):
    """
    Returns COSMOS galaxies as if viewed from the ground a la GREAT3
    
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
    def __init__(self,seed=None,**kw):
        assert seed is not None,"Random seed must be given in cosmos galaxy maker!"
        
        self.noise_mult = 1.0
        self.rng = galsim.UniformDeviate(seed)
        self.rng_np = np.random.RandomState(int(self.rng() * 1000000))

        self.cosmos_data = kw['galaxymaker'].get('cosmos_data',os.environ['GREAT3DATA'])
        self.real_galaxy = kw['galaxymaker'].get('real_galaxy',True)
        self.preload = kw['galaxymaker'].get('preload',False)
        
        self.cosmosgb = COSMOSGalaxyBuilder(self.real_galaxy,self.cosmos_data,preload=self.preload)
        self.catalog_dtype = self.cosmosgb.generateSubfieldParameters()['schema']
        self.catalog_dtype.append(('weight','f8'))
        self.catalog_dtype.append(('n_epochs','i4'))
        self.catalogs = {}

        self._setup(**kw)

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
        try:
            galinfo['orig_stamp_size_arcsec'] = galaxy.original.original_gal.image.array.shape[0]*0.03
        except:
            galinfo['orig_stamp_size_arcsec'] = galaxy.original.gal_image.array.shape[0]*0.03
        galinfo['max_size'] = max_size
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
        self.catalogs[seeing] = {'cat':self.cosmosgb.generateCatalog(self.rng,None,None,nb.typical_variance, \
                                                                     self.noise_mult,seeing=seeing,verbose=verbose, \
                                                                     randomly_rotate=randomly_rotate),
                                 'typical_var':nb.typical_variance}
        self.catalogs[seeing]['cat']['weight'] /= np.sum(self.catalogs[seeing]['cat']['weight'])
        
    def get_catalog_for_seeing(self,seeing,verbose=False,randomly_rotate=True):
        """
        Get a catalog for a specific seeing value.
        """
        if seeing not in self.catalogs:
            self.build_catalog_for_seeing(seeing,verbose=verbose,randomly_rotate=randomly_rotate)
        return self.catalogs[seeing]['cat'].copy()
        
    def get_prepsf_galaxy(self,seeing,n_epochs,max_size,pixel_scale,verbose=False, \
                          randomly_rotate=True,save_catalog=False):
        """
        Get a galaxy from COSMOS postage stamp a la GREAT3.
        
        In GREAT3, seeing was set to atmospheric PSF FWHM.        
        """
        if save_catalog or seeing in self.catalogs:
            if seeing not in self.catalogs:
                self.build_catalog_for_seeing(seeing,verbose=verbose,randomly_rotate=randomly_rotate)
            
            #now get catalog
            catalog = self.catalogs[seeing]['cat']
            Ncosmos = len(catalog)
            
            #now draw at random with weights
            # seed numpy.random to get predictable behavior
            randind = self.rng_np.choice(Ncosmos,replace=True,p=catalog['weight'])
            """
            while True:                
                randind = self.rng_np.choice(Ncosmos,replace=True)
                if self.rng_np.uniform() < catalog['weight'][randind]:
                    break            
            """
            record = catalog[randind].copy()
            record['n_epochs'] = n_epochs
            for tag in ["bulge_flux","disk_flux","flux_rescale"]:
                if tag in record.dtype.names:
                    record[tag] /= n_epochs
            nb = PlaceholderNoiseBuilder()
            nb_params = nb.generateEpochParameters(self.rng,record['n_epochs'],seeing,self.noise_mult)
            assert nb.typical_variance == self.catalogs[seeing]['typical_var']
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
        try:
            galinfo['orig_stamp_size_arcsec'] = galaxy.original.original_gal.image.array.shape[0]*0.03
        except:
            galinfo['orig_stamp_size_arcsec'] = galaxy.original.gal_image.array.shape[0]*0.03
        galinfo['max_size'] = max_size
        
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
            sub_xmax = xmin+offx+final_size - 1
        else:
            sub_xmin = xmin
            sub_xmax = xmax

        offy = curr_ysize - final_size
        if offy > 0:            
            offy = offy//2
            sub_ymin = ymin+offy
            sub_ymax = ymin+offy+final_size - 1
        else:
            sub_ymin = ymin
            sub_ymax = ymax
            
        sub_bounds = galsim.BoundsI(sub_xmin,sub_xmax,sub_ymin,sub_ymax)
        return sub_bounds

    def _get_final_size(self,orig_size,max_size,min_size,sizes):
        if min_size is None:
            min_size = 0
        if max_size is None:
            max_size = np.inf

        if orig_size > max_size:
            size = max_size            
        elif orig_size < min_size:
            size = min_size
        else:
            size = orig_size
        
        if sizes is not None and not ((min_size > np.max(sizes)) or (max_size < np.min(sizes))):
            sizes = np.array(sorted(sizes))
            q, = np.where((sizes >= size) & (sizes <= max_size) & (sizes >= min_size))
            assert len(q) > 0,"No possible stamp size given!"
            psizes = sizes[q]
            q = np.argmin(np.abs(orig_size-psizes))
            size = psizes[q]
            
        return size
    
    def apply_psf_and_noise_whiten(self,galaxy,galinfo,pixel,psf=None,max_size=None,min_size=None,sizes=None,use_great3_noise=False):
        """
        Automates finishing of galaxies for a psf and pixel. 
        
        Add the great3 noise level for ground observations w/ use_great3_noise = True.
        
        Shear should be applied already if wanted.
        
        Noise can then be added to the image via (for example)
        
            noise_to_add = np.sqrt(total_noise**2 - current_var)
            noise = galsim.GaussianNoise(rng, sigma=noise_to_add)
            noise.applyTo(final_galim)        
        """

        # cut to orig postage stamp in range
        if max_size is None:
            max_sz = galinfo['max_size']
        else:
            max_sz = np.min((max_size,galinfo['max_size']))
        orig_size = int(np.ceil(galinfo['orig_stamp_size_arcsec']/pixel.getScale()))
        size = self._get_final_size(orig_size,max_sz,min_size,sizes)
        
        # great3 did it like this
        # final_galaxy = galsim.Convolve([psf, pixel, galaxy])
        # galim = final_galaxy.draw(scale=pixel.getScale())
        # using newer galsim APIs
        if psf is not None:
            final_galaxy = galsim.Convolve([psf, pixel, galaxy])
        else:
            final_galaxy = galsim.Convolve([pixel, galaxy])
        galinfo['galsim_object'] = final_galaxy
        
        galim = final_galaxy.drawImage(scale=pixel.getScale(),method='no_pixel')
        
        if hasattr(final_galaxy,'noise'):
            #current_var = final_galaxy.noise.applyWhiteningTo(galim)
            current_var = final_galaxy.noise.whitenImage(galim)
        else:
            current_var = 0.0

        final_galim = self._get_sub_image(galim,size)

        im = final_galim.array.copy()
        wt = np.zeros_like(im)
        if current_var > 0:
            wt[:,:] = 1.0/current_var
        else:
            wt[:,:] = 1.0
        galinfo['center'] = utils.get_image_center(im,wt,rng=self.rng_np)
        
        if use_great3_noise:
            galinfo['noise_builder'].addNoise(self.rng,galinfo['noise_builder_params'],final_galim,current_var)
            return final_galim, galinfo['noise_builder_params']['variance']                
        else:
            return final_galim, current_var

    def get_extra_data_dtype(self):
        return [('cosmos_id','i8'),('g1_intrinsic','f8'),('g2_intrinsic','f8')]

    def get_extra_percutout_data_dtype(self):
        return [('variance','f8')]
        
    def get_galaxy(self,
                   psf=None,
                   shear=None,
                   n_epochs=None,
                   pixel_scale=None,
                   seeing=None,
                   **kwargs):

        if n_epochs is None:
            n_epochs = self.conf.get('n_epochs',1)

        if pixel_scale is None:
            key = 'pixel_scale'
            assert key in self.conf,"You must specify '%s' for stamps!" % key
            pixel_scale = self.conf['pixel_scale']

        if seeing is None:
            key = 'seeing'
            assert key in self.conf,"You must specify '%s' for stamps!" % key
            seeing = self.conf.get(key)
            
        for key in ['max_size','min_size','sizes']:
            assert key in self.conf,"You must specify '%s' for stamps!" % key

        psf['galsim_object'] = psf.get('galsim_object',None)
        
        galaxy,galinfo = self.get_prepsf_galaxy(seeing,n_epochs,self.conf['max_size'],pixel_scale, \
                                                verbose=self.conf['galaxymaker'].get('verbose',False), \
                                                randomly_rotate=self.conf['galaxymaker'].get('randomly_rotate',True), \
                                                save_catalog=self.conf['galaxymaker'].get('save_catalog',False))
        

        if shear is not None:
            galaxy = self._apply_shear(galaxy, shear)

        pixel = galsim.Pixel(scale=pixel_scale)
        
        final_gal_image,variance = self.apply_psf_and_noise_whiten(galaxy,galinfo,pixel,psf=psf['galsim_object'], \
                                                                   max_size=self.conf['max_size'], \
                                                                   min_size=self.conf['min_size'], \
                                                                   sizes=self.conf['sizes'], \
                                                                   use_great3_noise=self.conf['galaxymaker'].get('use_great3_noise',False))
        o = Observation()
        o.image = final_gal_image.array.copy()
        wt = np.zeros_like(o.image)
        wt[:,:] = 1.0/variance
        o.weight = wt
        row,col = galinfo['center']
        o.update(galinfo)
        o['row'] = row
        o['col'] = col
        o['variance'] = variance
        o['pixel_scale'] = pixel_scale
        o['n_epochs'] = n_epochs
        o.psf = psf
        o['galsim_image'] = final_gal_image
        o['prepsf_galsim_object'] = galaxy
        o['extra_percutout_data'] = {'variance':[variance]}
        gi = np.array([galinfo['info']],dtype=self.catalog_dtype)
        o['extra_data'] = dict(cosmos_id=gi['cosmos_ident'][0], \
                               g1_intrinsic=gi['g1_intrinsic'][0], \
                               g2_intrinsic=gi['g2_intrinsic'][0],
                               shear=shear)
        return o


    def _apply_shear(self, galaxy, shear):
        from ngmix import Shape
        if not isinstance(shear, Shape):
            # Use a Shape object, since it performs checks
            shear = Shape(shear[0], shear[1])

        sheared_galaxy = galaxy.shear(g1=shear.g1, g2=shear.g2)            

        return sheared_galaxy
