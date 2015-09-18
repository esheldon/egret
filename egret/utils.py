#!/usr/bin/env python
import numpy as np
import ngmix

def get_maker(full_name):
    name = full_name.lower().replace('-','')
    if name == "great3cosmos":
        from .galaxymakers import GREAT3COSMOSGalaxyMaker
        return GREAT3COSMOSGalaxyMaker
    elif name == "despsf":
        from .psfmakers import DESPSFMaker
        return DESPSFMaker
    else:
        assert False,"Maker '%s' not found!" % full_name

def get_image_center(im,wt,rng=None,ntry=10,T0=10.0,**em_pars):
    """
    get the image center via fitting a single Gaussian with EM
    
    lifted from ngmix bootstrappers by E. Sheldon
    """

    if rng is None:
        rng = np.random.RandomState()
    em_pars['maxiter'] = em_pars.get('maxiter',4000)
    em_pars['tol'] = em_pars.get('tol',5e-6)
        
    srandu = lambda : rng.uniform(low=-1.0,high=1.0)
    
    row = im.shape[0]/2.0
    col = im.shape[1]/2.0
    jac = ngmix.Jacobian(0.0,0.0,1.0,0.0,0.0,1.0)
    obs = ngmix.Observation(im,weight=wt,jacobian=jac)
    
    for i in xrange(ntry):
        guess=[1.0*(1.0 + 0.01*srandu()),
               row*(1.0+0.5*srandu()),
               col*(1.0+0.5*srandu()),
               T0/2.0*(1.0 + 0.1*srandu()),
               0.1*ngmix.srandu(),
               T0/2.0*(1.0 + 0.1*srandu())]
        gm_guess = ngmix.GMix(pars=guess)
        fitter = ngmix.em.fit_em(obs,gm_guess,**em_pars)
        res = fitter.get_result()
        
        if res['flags']==0:
            break
        
    if res['flags']==0:
        gm = fitter.get_gmix()
        row,col = gm.get_cen()
    
    return row,col

def get_fft_sizes(min_size=32,max_size=256,facs=[2,3],minpows=[1,0]):
    """
    produce FFT friendly stamp sizes with only powers of facs
    """
    maxpows = [int(np.ceil(np.log(max_size)/np.log(fac))) for fac in facs]

    sizes = [1]
    for fac,minpow,maxpow in zip(facs,minpows,maxpows):
        vals = [fac**i for i in range(minpow,maxpow+1)]
        new_sizes = []
        for sz in sizes:
            for val in vals:
                new_sizes.append(val*sz)
        sizes = new_sizes

    sizes = [sz for sz in sizes if sz >= min_size and sz <= max_size]
    sizes = sorted(sizes)
    return sizes
        
