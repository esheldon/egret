import numpy as np
import galsim
import ngmix

from .galaxymaker import GalaxyMaker

class ConstShearPairedExpGalaxyMaker(GalaxyMaker):
    """
    Makes sersic index 1 galaxies with a constant shear and 
    paired for a ring test.    
    """
    def __init__(self,g1,g2,seed,**kw):
        #do config
        self.conf = {}
        self.conf.update(kw)
        def_conf = {'shape_noise':0.16, 'half_light_radius':2.0, 'flux':100.0, 'noise':1.0e-4}
        for key in def_conf:
            if key not in self.conf:
                self.conf[key] = def_conf[key]

        self.g1 = g1
        self.g2 = g2
        self.random = np.random.RandomState(seed)
        self.gaussian_noise = galsim.GaussianNoise(galsim.BaseDeviate(seed),sigma=self.conf['noise'])
        
    def _draw_gaussian_shape(self):
        ok = False
        while not ok:
            g1 = self.random.normal()*self.conf['shape_noise']
            g2 = self.random.normal()*self.conf['shape_noise']
            if np.abs(g1) < 1.0 and np.abs(g2) < 1.0 and g1*g1 + g2*g2 < 1.0:
                ok = True
        return g1,g2
    
    def _draw_galaxy(self,g1s,g2s,psf,pixel,nx=None,ny=None):
        pixel_scale = pixel.getScale()
        gal = galsim.Sersic(1.0,
                            half_light_radius=self.conf['half_light_radius'],
                            flux=self.conf['flux'])
        gal.applyShear(g1=g1s,g2=g2s)
        gal.applyShear(g1=self.g1,g2=self.g2)
        gal_final = galsim.Convolve([gal, psf, pixel])        
        if nx is not None and ny is not None:
            image_obj = gal_final.draw(scale=pixel_scale,nx=nx,ny=ny)
        else:
            image_obj = gal_final.draw(scale=pixel_scale)

        image_obj.addNoise(self.gaussian_noise)
        image = image_obj.array.astype('f8')
        wt = image*0.0 + (1.0/self.conf['noise']**2)

        return image, wt
                                                                                                        
    def get_galaxy_pair(self,psf,pixel,nx=None,ny=None,g1s=None,g2s=None):
        pixel_scale = pixel.getScale()

        if g1s is None and g2s is None:
            g1s,g2s = self._draw_gaussian_shape()
        else:
            assert g1s is not None and g2s is not None, "You must specify both shape noise parameters!"
        
        im1,wt1 = self._draw_galaxy(g1s,g2s,psf,pixel,nx=nx,ny=ny)
        if nx is None and ny is None:
            nx,ny = im1.shape
        im2,wt2 = self._draw_galaxy(-g1s,-g2s,psf,pixel,nx=ny,ny=ny)

        j = ngmix.Jacobian(im1.shape[0]/2.0,im1.shape[2]/2.0,pixel_scale,0.0,0.0,pixel_scale)
        if nx is not None and ny is not None:
            psf_im = psf.draw(scale=pixel_scale,nx=nx,ny=ny)
        else:
            psf_im = psf.draw(scale=pixel_scale)
        psf_obs = ngmix.Observation(image=psf_im,jacobian=j)
        
        meta = {}
        meta['g1s'] = g1s
        meta['g2s'] = g2s
        meta['g1'] = self.g1
        meta['g2'] = self.g2        
        obs1 = ngmix.Observation(image=im1,weight=wt1,jacobian=j,psf=psf_obs)
        obs1.update_meta_data(meta)
        obs1.update_meta_data(self.conf)

        meta = {}
        meta['g1s'] = -g1s
        meta['g2s'] = -g2s
        meta['g1'] = self.g1
        meta['g2'] = self.g2        
        obs1 = ngmix.Observation(image=im2,weight=wt2,jacobian=j,psf=psf_obs)
        obs1.update_meta_data(meta)
        obs1.update_meta_data(self.conf)
        
        return obs1,obs2

