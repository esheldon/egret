import os
import numpy as np
import fitsio

__all__ = ['MEDSMaker']

class MEDSMaker(object):
    """
    Object to make MEDS files.
    
    Example:
    
        extra_data = [('cosmos_id','i8')]
        extra_percutout_data = [('id_psf','i4')]
        mm = MEDSMaker(extra_data=extra_data,extra_percutout_data=extra_percutout_data)
        objinfo = dict(id=id,number=number,file_id=file_id,orig_row=orig_row,orig_col=orig_col,
                       orig_start_row=orig_start_row,orig_start_col=orig_start_col,
                       dudrow=dudrow,dudcol=dudcol,dvdrow=dvdrow,dvdcol=dvdcol,
                       cutout_row=cutout_row,cutout_col=cutout_col,
                       cosmos_id=cosmos_id,id_psf=id_psf)    
        mm.add_object(objinfo,[im1,im2,im3],[wgt1,wgt2,wgt3],[seg1,seg2,seg3])
        mm.write('medstest.fit')    
    """
    
    def __init__(self,extra_data=None,extra_percutout_data=None):
        self.imgpixels = []
        self.wgtpixels = []
        self.segpixels = []
        self.objinfo = []
        self.start_rows = []
        self.ncutouts = []
        self.box_sizes = []
        self.num_objs = 0
        self.max_cutouts = -1
        self.npix_tot = 0
        self.extra_data = extra_data
        self.extra_percutout_data = extra_percutout_data

        if extra_percutout_data is not None:
            self.extra_vector_names = []
            for nm,dt in extra_percutout_data:
                self.extra_vector_names.append(nm)
        else:
            self.extra_vector_names = []
        
        #some defaults
        self.NEG_INT_NONEVAL = -9999
        self.POS_INT_NONEVAL = 9999
        self.NEG_FLOAT_NONEVAL = -9999.0
        self.POS_FLOAT_NONEVAL = 9999.0
        
    def add_object(self,objinfo,imgs,wgts,segs):
        """
        Adds an object to the file.
        
        imgs, wgts, segs: lists for each cutout        
        objinfo: dict with all of the information needed for the 
            MEDS object_data extension, except the box_size, # of cutouts and 
            location in the MEDS image extensions. (These last things get 
            computed by the MEDSMaker.)
        """
        assert len(imgs) == len(wgts)
        assert len(imgs) == len(segs)
                
        self.objinfo.append(objinfo)
        start_rows = []
        for img in imgs:
            start_rows.append(self.npix_tot)
            self.npix_tot += img.shape[0]*img.shape[1]
        self.start_rows.append(start_rows)
        self.num_objs += 1
        self.ncutouts.append(len(imgs))
        if len(imgs) > self.max_cutouts:
            self.max_cutouts = len(imgs)

        if len(imgs) > 0:
            self.box_sizes.append(imgs[0].shape[0])
        else:
            self.box_sizes.append(0)
            
        for cuts,pixels in zip([imgs,wgts,segs],[self.imgpixels,self.wgtpixels,self.segpixels]):
            for cut in cuts:
                pixels.extend(list(cut.reshape(-1)))
            
    def _get_object_data_dtype(self,nmax):
        if nmax <= 1:
            nmax = 2
        dlist = [('number', '>i4'),
                 ('ncutout', '>i4'),
                 ('id', '>i4'),
                 ('box_size', '>i4'),
                 ('file_id', '>i4', (nmax,)),
                 ('start_row', '>i4', (nmax,)),
                 ('orig_row', '>f8', (nmax,)),
                 ('orig_col', '>f8', (nmax,)),
                 ('orig_start_row', '>i4', (nmax,)),
                 ('orig_start_col', '>i4', (nmax,)),
                 ('dudrow', '>f8', (nmax,)),
                 ('dudcol', '>f8', (nmax,)),
                 ('dvdrow', '>f8', (nmax,)),
                 ('dvdcol', '>f8', (nmax,)),
                 ('cutout_row', '>f8', (nmax,)),
                 ('cutout_col', '>f8', (nmax,))]

        self.vector_names = ['file_id','orig_row','orig_col','orig_start_row','orig_start_col',
                             'dudrow','dudcol','dvdrow','dvdcol','cutout_row','cutout_col']

        #set user data
        if self.extra_data is not None:
            dlist.extend(self.extra_data)

        if self.extra_percutout_data is not None:
            for nm,dt in self.extra_percutout_data:
                dlist.append((nm,dt,(nmax,)))
                self.vector_names.append(nm)
        
        return np.dtype(dlist)
    
    def _write_object_data(self,fname):
        dtype = self._get_object_data_dtype(self.max_cutouts)
        odata = np.zeros(self.num_objs,dtype=dtype)
        names = odata.dtype.names
        
        #set info
        for i,objinfo in enumerate(self.objinfo):
            for name in names:
                if name in objinfo:
                    if name in self.vector_names:
                        if self.ncutouts[i] > 0:
                            odata[name][i,0:self.ncutouts[i]] = objinfo[name]
                    else:
                        odata[name][i] = objinfo[name]

            #set tags that need special attention
            odata['ncutout'][i] = self.ncutouts[i]
            odata['box_size'][i] = self.box_sizes[i]
            odata['start_row'][i,0:odata['ncutout'][i]] = np.array(self.start_rows[i][:])
            
        fitsio.write(fname,odata,extname='object_data')
        self.odata = odata

    def write(self,name,image_info=None,metadata=None,clobber=True,compress=None):
        """
        Writes data to a MEDS file <name>.
        
        The image_info table and metadata can be supplied if needed.
        The file will be clobbered by default.        
        """
        #clobber the file if needed
        if clobber and os.path.exists(name):
            os.remove(name)

        #write info tables
        self._write_object_data(name)

        blank_data = np.zeros(1,dtype=[('blank_table','i8')])        
        if image_info is not None:
            fitsio.write(name,image_info,extname='image_info')
        else:
            fitsio.write(name,blank_data,extname='image_info')

        if metadata is not None:
            fitsio.write(name,metadata,extname='metadata')
        else:
            fitsio.write(name,blank_data,extname='metadata')

        #write pixel info
        self.imgpixels = np.array(self.imgpixels,dtype='f4')
        fitsio.write(name,self.imgpixels,extname='image_cutouts',compress=compress)
        self.imgpixels = list(self.imgpixels)
        
        self.wgtpixels = np.array(self.wgtpixels,dtype='f4')
        fitsio.write(name,self.wgtpixels,extname='weight_cutouts',compress=compress)
        self.wgtpixels = list(self.wgtpixels)
        
        self.segpixels = np.array(self.segpixels,dtype='i2')
        fitsio.write(name,self.segpixels,extname='seg_cutouts',compress=compress)
        self.segpixels = list(self.segpixels)

        
def test():
    """
    Test suite for MEDSMaker
    """

    #function for test obs
    def make_test_obs(rng,ncutout,shape,blank=False):
        oi = {}
        for tag in ['id','number','cosmos_id']:
            oi[tag] = int(rng.uniform(low=0,high=1e6))

        odata = {}
        odata['oi'] = oi
        odata['imgs'] = []
        odata['wgts'] = []
        odata['segs'] = []

        for tag in ['id_psf','file_id','orig_start_row','orig_start_col']:
            if not blank:
                oi[tag] = rng.uniform(low=0,high=1e6,size=ncutout).astype('i4')
            else:
                oi[tag] = np.zeros(ncutout,dtype='i4')

        for tag in ['orig_row','orig_col',
                    'dudrow','dudcol','dvdrow','dvdcol',
                    'cutout_row','cutout_col']:
            if not blank:
                oi[tag] = rng.uniform(low=0,high=1e6,size=ncutout)
            else:
                oi[tag] = np.zeros(ncutout,dtype='f8')

        if not blank:
            for i in xrange(ncutout):
                odata['imgs'].append(rng.uniform(low=0,high=1e6,size=shape).astype('f4'))
                odata['wgts'].append(rng.uniform(low=0,high=1e6,size=shape).astype('f4'))
                odata['segs'].append(rng.uniform(low=0,high=1e6,size=shape).astype('i2'))
        
        return odata

    #comp test obs to meds ind
    def comp_odata_medsind(odata,ncutout,shape,ind,m):
        assert m['ncutout'][ind] == ncutout,"Error computing 'ncutout' in MEDSMaker!"
        assert m['box_size'][ind] == shape[0],"Error computing 'box_size' in MEDSMaker!"
        
        for tag in ['id','number','cosmos_id']:
            assert odata['oi'][tag] == m[tag][ind],"Error writing tag '%s' in MEDSMaker!" % tag

        for tag in ['orig_row','orig_col','orig_start_row','orig_start_col',
                    'dudrow','dudcol','dvdrow','dvdcol','file_id',
                    'cutout_row','cutout_col']:
            assert np.array_equal(odata['oi'][tag],m[tag][ind,0:ncutout]),"Error writing tag '%s' in MEDSMaker!" % tag
            
        for i in xrange(ncutout):
            for tag,tpe in zip(['imgs','wgts','segs'],['image','weight','seg']):
                assert np.array_equal(odata[tag][i],m.get_cutout(ind,i,type=tpe)),"Error in writing cutout %d for type '%s' in MEDSMaker!" % (i,tpe)

        return True

    #do tests
    from .oorandom import OORandom
    rng = OORandom(12345)
    
    obslist = []    
    obslist.append((make_test_obs(rng,0,(0,0),blank=True),0,(0,0)))
    #obslist.append((make_test_obs(rng,1,(64,64)),1,(64,64)))
    obslist.append((make_test_obs(rng,1,(64,64)),1,(64,64)))
    obslist.append((make_test_obs(rng,11,(13,13)),11,(13,13)))

    extra_data = [('cosmos_id','i8')]
    extra_percutout_data = [('id_psf','i4')]

    import meds
    
    for perm in [[0,1],[1,0],[0,2],[2,0],[1,2],[2,1],[0,1,2],[2,0,1],[1,2,0],[0,2,1],[1,0,2],[2,1,0]]:
        mm = MEDSMaker(extra_data=extra_data,extra_percutout_data=extra_percutout_data)

        for ind in perm:
            mm.add_object(obslist[ind][0]['oi'],
                          obslist[ind][0]['imgs'],
                          obslist[ind][0]['wgts'],
                          obslist[ind][0]['segs'])
        mm.write('medstest.fit')    
        del mm

        with meds.MEDS('medstest.fit') as m: 
            for mind,ind in enumerate(perm):
                comp_odata_medsind(obslist[ind][0],obslist[ind][1],obslist[ind][2],mind,m)

    os.remove('medstest.fit')
                
    print "MEDSMaker passed all tests!"



del os
del np
del fitsio
