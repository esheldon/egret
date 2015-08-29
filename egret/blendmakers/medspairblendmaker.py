#!/usr/bin/env python
import os
import sys
import numpy as np
from ..galaxymakers import ExpNGMixGalaxyMaker
from ..medsmakers import MemoryMEDSMaker
import fitsio
import matplotlib.pyplot as plt
import copy

class BlendedPairMEDSMaker(object):
    def __init__(self,Np,Npsf=25,**kwargs):
        if 'seed' in kwargs:
            self.rs = np.random.RandomState(seed=kwargs['seed'])
        else:
            self.rs = np.random.RandomState()

        if 'noise_obj' not in kwargs:
            kwargs['noise_obj'] = 1e-4
        
        gm_pars = {}
        gm_pars.update(kwargs)
        if 'noise_obj' in gm_pars:
            del gm_pars['noise_obj']
        gm_pars['noise_obj'] = 0.0        
        if 'noise_psf' in gm_pars:
            del gm_pars['noise_psf']
        gm_pars['noise_psf'] = 0.0
        self.gm = ExpNGMixGalaxyMaker()        
        self.gm.set_params(**gm_pars)        
        
        self.Npsf = Npsf
        self.Np = Np
        self.Nobj = Np*2
        self.noise_obj = kwargs['noise_obj']

        if 'minoff' not in kwargs:
            kwargs['minoff'] = 4
        if 'maxoff' not in kwargs:
            kwargs['maxoff'] = 4            
        self.minoff = kwargs['minoff']
        self.maxoff = kwargs['maxoff']

        self.tail = 'data'
        
    def make_data(self):
        self.make_psfs()
        self.make_gals()
        self.make_nbrs_fofs()
        self.do_blends()

    def write_data(self):
        fitsio.write('psf_%s.fits'%self.tail,self.psf_data,clobber=True)
        fitsio.write('obj_%s.fits'%self.tail,self.obj_data,clobber=True)
        fitsio.write('nbrs_%s.fits'%self.tail,self.nbrs,clobber=True)
        fitsio.write('fof_%s.fits'%self.tail,self.fofs,clobber=True)

        self.mm.write('meds_%s.fits' % self.tail)
        self.mm.fpack()
        os.remove('meds_%s.fits' % self.tail)

        self.bmm.write('blended_meds_%s.fits' % self.tail)
        self.bmm.fpack()
        os.remove('blended_meds_%s.fits' % self.tail)
        
    def do_blends(self):
        # do blends

        noise_obj = self.noise_obj
        
        self.mm = MemoryMEDSMaker(extra_percutout_data=[('ind_psf','i8')])
        self.bmm = MemoryMEDSMaker(extra_percutout_data=[('ind_psf','i8')])
        for i in xrange(self.Np):
            g1 = self.gals[i*2]
            g2 = self.gals[i*2+1]

            maxoff = self.maxoff
            minoff = self.minoff
            off_x = self.rs.choice(maxoff,size=1,replace=True)[0]
            off_y = self.rs.choice(maxoff,size=1,replace=True)[0]
            if off_x < minoff:
                off_x = minoff
            if off_y < minoff:
                off_y = minoff

            if self.rs.uniform() > 0.5:
                off_x *= -1
            if self.rs.uniform() > 0.5:
                off_y *= -1

            minx = min([0,off_x])
            maxx = max([g1.image.shape[0],g2.image.shape[0]+off_x])
            
            miny = min([0,off_y])
            maxy = max([g1.image.shape[1],g2.image.shape[1]+off_y])
            
            sze = max([maxx-minx,maxy-miny])
            imtot = np.zeros((sze,sze))
            
            if off_x < 0:
                xl1 = abs(off_x)
            else:
                xl1 = 0

            if off_y < 0:
                yl1 = abs(off_y)
            else:
                yl1 = 0
            imtot[xl1:xl1+g1.image.shape[0],yl1:yl1+g1.image.shape[1]] = g1.image

            if off_x < 0:
                xl2 = 0
            else:
                xl2 = off_x

            if off_y < 0:
                yl2 = 0
            else:
                yl2 = off_y
            imtot[xl2:xl2+g2.image.shape[0],yl2:yl2+g2.image.shape[1]] += g2.image

            nse = self.rs.normal(size=imtot.shape)*self.noise_obj
            imtot += nse

            cens = [[xl1+g1.image.shape[0]/2.0,yl1+g1.image.shape[1]/2.0],
                    [xl2+g2.image.shape[0]/2.0,yl2+g2.image.shape[1]/2.0]]
            nums = [i*2+1,i*2+2]
            seg = get_seg(imtot,self.noise_obj,10.0,cens,nums)

            fitsio.write('images%d_%s.fits' % (i,self.tail),imtot,clobber=True)
            fitsio.write('images%d_%s.fits' % (i,self.tail),seg)
            
            bobjinfo1 = dict(id=i*2,number=i*2+1,
                             orig_row=xl1+g1.image.shape[0]/2.0,
                             orig_col=yl1+g1.image.shape[1]/2.0,
                             orig_start_row=xl1,
                             orig_start_col=yl1,
                             dudrow=1.0,
                             dudcol=0.0,
                             dvdrow=0.0,
                             dvdcol=1.0,
                             cutout_row=g1.image.shape[0]/2.0,
                             cutout_col=g1.image.shape[1]/2.0,
                             ind_psf=g1.meta['ind_psf'])
            self.bmm.add_object(bobjinfo1,[imtot[xl1:xl1+g1.image.shape[0],yl1:yl1+g1.image.shape[1]]],
                                [g1.image*0.0 + 1.0/noise_obj/noise_obj],
                                [seg[xl1:xl1+g1.image.shape[0],yl1:yl1+g1.image.shape[1]]])

            objinfo1 = dict(id=i*2,number=i*2+1,
                            orig_row=g1.image.shape[0]/2.0,
                            orig_col=g1.image.shape[1]/2.0,
                            orig_start_row=0,
                            orig_start_col=0,
                            dudrow=1.0,
                            dudcol=0.0,
                            dvdrow=0.0,
                            dvdcol=1.0,
                            cutout_row=g1.image.shape[0]/2.0,
                            cutout_col=g1.image.shape[1]/2.0,
                            ind_psf=g1.meta['ind_psf'])
            self.mm.add_object(objinfo1,[g1.image+nse[xl1:xl1+g1.image.shape[0],yl1:yl1+g1.image.shape[1]]],
                          [g1.image*0.0 + 1.0/noise_obj/noise_obj],
                          [np.zeros(g1.image.shape,dtype='i4')+i*2+1])

            bobjinfo2 = dict(id=i*2+1,number=i*2+1+1,
                             orig_row=xl2+g2.image.shape[0]/2.0,
                             orig_col=yl2+g2.image.shape[1]/2.0,
                             orig_start_row=xl2,
                             orig_start_col=yl2,
                             dudrow=1.0,
                             dudcol=0.0,
                             dvdrow=0.0,
                             dvdcol=1.0,
                             cutout_row=g2.image.shape[0]/2.0,
                             cutout_col=g2.image.shape[1]/2.0,
                             ind_psf=g2.meta['ind_psf'])
            self.bmm.add_object(bobjinfo2,[imtot[xl2:xl2+g2.image.shape[0],yl2:yl2+g2.image.shape[1]]],
                                [g2.image*0.0 + 1.0/noise_obj/noise_obj],
                                [seg[xl2:xl2+g2.image.shape[0],yl2:yl2+g2.image.shape[1]]])
            
            objinfo2 = dict(id=i*2+1,number=i*2+1+1,
                            orig_row=g2.image.shape[0]/2.0,
                            orig_col=g2.image.shape[1]/2.0,
                            orig_start_row=0,
                            orig_start_col=0,
                            dudrow=1.0,
                            dudcol=0.0,
                            dvdrow=0.0,
                            dvdcol=1.0,
                            cutout_row=g2.image.shape[0]/2.0,
                            cutout_col=g2.image.shape[1]/2.0,
                            ind_psf=g2.meta['ind_psf'])
            self.mm.add_object(objinfo2,[g2.image+nse[xl2:xl2+g2.image.shape[0],yl2:yl2+g2.image.shape[1]]],
                          [g2.image*0.0 + 1.0/noise_obj/noise_obj],
                          [np.zeros(g2.image.shape,dtype='i4')+i*2+1+1])

    def make_gals(self,verbose=False):
        gals = []
        d =[]
        if verbose:
            import progressbar
            bar = progressbar.ProgressBar(maxval=self.Nobj,widgets=[progressbar.Bar(marker='|',left='doing work: |',right=''),' ',progressbar.Percentage(),' ',progressbar.AdaptiveETA()])
            bar.start()
        for i in xrange(self.Nobj):
            if verbose:
                bar.update(i+1)
            psf_ind = self.rs.choice(self.Npsf,size=1,replace=True)
            obs = self.gm.get_galaxy(pars_psf=self.psf_data['pars'][psf_ind[0]])
            meta = dict(ind_psf=psf_ind)
            obs.update_meta_data(meta)
            gals.append(obs)
            d.append((i,i+1,obs.meta['pars_obj']))
        if verbose:
            bar.finish()
        d = np.array(d,dtype=[('id','i8'),('number','i4'),('pars','f8',len(obs.meta['pars_obj']))])
        self.obj_data = d
        self.gals = gals
        
    def make_nbrs_fofs(self):
        fofs = []
        nbrs = []
        for i in xrange(self.Np):
            fofs.append((i,i*2+1))
            fofs.append((i,i*2+2))
            nbrs.append((i*2+1,i*2+2))
            nbrs.append((i*2+2,i*2+1))
        self.fofs = np.array(fofs,dtype=[('fofid','i8'),('number','i4')])
        self.nbrs = np.array(nbrs,dtype=[('number','i4'),('nbr_number','i4')])

    def make_psfs(self):
        ms = -np.inf
        psf_data = []
        for i in xrange(self.Npsf):
            obs = self.gm.get_galaxy()
            im_psf = obs.get_psf().image
            psf_data.append((im_psf,obs.meta['pars_psf']))
            if im_psf.shape[0] > ms:
                ms = im_psf.shape[0]
                assert im_psf.shape[0] == ms
        psf_data = np.array(psf_data,dtype=[('im','f8',(ms,ms)),('pars','f8',len(obs.meta['pars_psf']))])
        self.psf_data = psf_data


def get_seg(im,sigma,nsigma,cens,nums):
    """
    stupid simple code to make a pseudo-seg map
    """

    xc = []
    yc = []
    for ceni in cens:
        xc.append(ceni[0])
        yc.append(ceni[1])
    xc = np.array(xc)
    yc = np.array(yc)

    seg = np.zeros_like(im,dtype='i4')
    
    qx,qy = np.where(im > sigma*nsigma)    
    for i,j in zip(qx,qy):
        d2 = (i*1.0-xc)**2 + (j*1.0-yc)**2.0
        q = np.argmin(d2)
        seg[i,j] = nums[q]

    return seg
