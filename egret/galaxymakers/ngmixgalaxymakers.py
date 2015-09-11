#!/usr/bin/env python
import os
import sys
import ngmix
import numpy as np
import copy
from ngmix import gmix
from .galaxymaker import GalaxyMaker
from ngmix.jacobian import UnitJacobian

__all__ = ['ExpNGMixGalaxyMaker']

class ExpNGMixGalaxyMaker(GalaxyMaker):
    def __init__(self,seed=None):
        self.seed = seed
        if self.seed is not None:
            np.random.seed(seed)
        self.random_state = np.random.get_state()

        self.set_params()
        self.make_priors()
        
    def set_params(self,T_obj=16.0,T_psf=4.0,
                   sigma_T_obj=0.1,sigma_T_psf=0.1,eps_psf=0.05,
                   cnts_obj=100.0,sigma_cnts_obj=0.1,sigma_cen=0.1,
                   noise_obj=1e-3,noise_psf=0.0):
        
        self.T_obj = T_obj
        self.sigma_T_obj = sigma_T_obj
        self.T_psf = T_psf
        self.sigma_T_psf = sigma_T_psf
        self.eps_psf = eps_psf
        self.cnts_obj = cnts_obj
        self.sigma_cnts_obj = sigma_cnts_obj
        self.sigma_cen = sigma_cen
        self.noise_obj = noise_obj
        self.noise_psf = noise_psf
        
    def make_priors(self):
        from ngmix.joint_prior import PriorSimpleSep
        from ngmix.priors import ZDisk2D

        g_prior = ngmix.priors.make_gprior_cosmos_sersic(type='erf')
        g_prior_flat = ZDisk2D(1.0)

        counts_prior = ngmix.priors.TwoSidedErf(-10.0, 1.0, 1.0e+09, 0.25e+08)
        T_prior = ngmix.priors.TwoSidedErf(-0.07, 0.03, 1.0e+06, 1.0e+05)
        cen_prior = ngmix.priors.CenPrior(0.0, 0.0, 0.25, 0.25)
        
        prior = PriorSimpleSep(cen_prior,
                               g_prior,
                               T_prior,
                               counts_prior)
        
        gflat_prior = PriorSimpleSep(cen_prior,
                                     g_prior_flat,
                                     T_prior,
                                     counts_prior)
        priors = {}
        priors['prior'] = prior
        priors['gflat_prior'] = gflat_prior
        priors['g_prior'] = g_prior
        self.priors = priors

    def draw_params(self):
        g1,g2 = self.priors['g_prior'].sample2d(1)
        T = np.exp(np.random.normal()*self.sigma_T_obj + np.log(self.T_obj))
        cnts = np.exp(np.random.normal()*self.sigma_cnts_obj + np.log(self.cnts_obj))
        x = np.random.normal()*self.sigma_cen
        y = np.random.normal()*self.sigma_cen
        return [x,y,g1,g2,T,cnts]

    def draw_psf_params(self):
        g1 = np.random.normal()*self.eps_psf
        g2 = np.random.normal()*self.eps_psf
        T = np.exp(np.random.normal()*self.sigma_T_psf + np.log(self.T_psf))
        return [0.0,0.0,g1,g2,T,100.0]
    
    def get_galaxy(self,pars_psf=None):
        """
        hacked out of ngmix
        """
        np.random.set_state(self.random_state)

        psf_model="gauss"
        if pars_psf is None:
            pars_psf = np.array(self.draw_psf_params())
        
        model = 'exp'
        pars_obj = np.array(self.draw_params())
        
        sigma = np.sqrt( (pars_obj[4] + pars_psf[4])/2. )
        dim = int(2.0*5.0*sigma)
        if dim%2 == 1: dim += 1
        dims = [dim]*2
        cen = [dims[0]/2., dims[1]/2.]
        j = UnitJacobian(cen[0],cen[1])
        
        dims_psf = [2.*5.*np.sqrt(self.T_psf*(1.0 + self.sigma_T_psf*10.0)/2.0)]*2
        cen_psf = [dims_psf[0]/2., dims_psf[1]/2.]
        j_psf = UnitJacobian(cen_psf[0],cen_psf[1])
        
        gm_psf = gmix.GMixModel(pars_psf, psf_model)
        gm_obj0 = gmix.GMixModel(pars_obj, model)
        gm = gm_obj0.convolve(gm_psf)
        
        im_psf = gm_psf.make_image(dims_psf, jacobian=j_psf, nsub=16)
        npsf = self.noise_psf*np.random.randn(im_psf.size).reshape(im_psf.shape)
        im_psf[:,:] += npsf
        if self.noise_psf > 0.0:
            wt_psf = np.zeros(im_psf.shape) + 1./self.noise_psf**2
        else:
            wt_psf = np.zeros(im_psf.shape) + 1.0

        im_obj = gm.make_image(dims, jacobian=j, nsub=16)
        n = self.noise_obj*np.random.randn(im_obj.size).reshape(im_obj.shape)
        im_obj[:,:] += n
        if self.noise_obj > 0.0:
            wt_obj = np.zeros(im_obj.shape) + 1./self.noise_obj**2
        else:
            wt_obj = np.zeros(im_obj.shape) + 1.0

        meta = {'pars_psf':pars_psf,
                'im_psf':im_psf,
                'wt_psf':wt_psf,
                'noise_psf':copy.copy(self.noise_psf),
                'pars_obj':pars_obj,
                'noise_obj':copy.copy(self.noise_obj)}

        self.random_state = np.random.get_state()

        psf_obs = ngmix.Observation(image=im_psf,weight=wt_psf,jacobian=j_psf)
        psf_obs.update_meta_data(meta)
        obs = ngmix.Observation(image=im_obj,weight=wt_obj,psf=psf_obs,jacobian=j)
        obs.update_meta_data(meta)
        return obs
