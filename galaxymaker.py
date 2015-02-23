import numpy as np

class GalaxyMaker(object):
    def __init__(self):
        raise NotImplementedError

    def get_galaxy(self):
        raise NotImplementedError

    def get_galaxy_pair(self):
        raise NotImplementedError

class ConstShearExpGalaxyMaker(GalaxyMaker):
    import galsim
    
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
                
    def _draw_gaussian_shape(self):
        ok = False
        while not ok:
            g1 = np.random.normal()*self.conf['shape_noise']
            g2 = np.random.normal()*self.conf['shape_noise']
            if np.abs(g1) < 1.0 and np.abs(g2) < 1.0 and g1*g1 + g2*g2 < 1.0:
                ok = True

        return g1,g2
    
    def _draw_galaxy(self,g1s,g2s,psf,pixel,pixel_scale,nx=None,ny=None):
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

        image_obj.addNoise(galsim.GaussianNoise(sigma=self.conf['noise']))
        image1 = image_obj.array.astype('f8')
        wt = image*0.0 + (1.0/self.conf['noise']**2)

        return image, wt
                                                                                                        
    def _draw_galaxy_pair(self,g1s,g2s,psf,pixel,pixel_scale,nx=None,ny=None):
        im1,wt1 = self._draw_galaxy(g1s,g2s,psf,pixel,pixel_scale,nx=nx,ny=ny)
        if nx is None and ny is None:
            nx,ny = im1.shape
        im2,wt2 = self._draw_galaxy(-g1s,-g2s,psf,pixel,pixel_scale,nx=ny,ny=ny)

        return im1,wt1,im2,wt2
