import numpy as np
import fitsio

class MEDSMaker(object):
    """
    Object to make MEDS files.
    """

    def __init__(self,extra_object_data_tags=None,extra_percutout_names=None):
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
        self.extra_tags = extra_object_data_tags
        self.extra_vector_names = extra_percutout_names
        
        #some defaults
        self.NEG_INT_NONEVAL = -9999
        self.POS_INT_NONEVAL = 9999
        self.NEG_FLOAT_NONEVAL = -9999.0
        self.POS_FLOAT_NONEVAL = 9999.0
        
    def add_object(self,objinfo,imgs,wgts,segs):
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

        if self.extra_tags is not None:
            dlist.extend(self.extra_tags)

        self.vector_names = ['file_id','orig_row','orig_col','orig_start_row','orig_start_col',
                             'dudrow','dudcol','dvdrow','dvdcol','cutout_row','cutout_col']
        if self.extra_vector_names is not None:
            self.vector_names.extend(self.extra_vector_names)
            
        return np.dtype(dlist)
    
    def _write_object_data(self,name):
        dtype = self._get_object_data_dtype(self.max_cutouts)
        odata = np.zeros(self.num_objs,dtype=dtype)
        names = odata.dtype.names
        
        #set info
        for i,objinfo in enumerate(self.objinfo):
            for name in names:
                if name in objinfo:
                    if name in self.vector_names:
                        odata[name][i,0:self.ncutouts[i]] = objinfo[name][:]
                    else:
                        odata[name][i] = objinfo[name]

                #set tags that need special attention
                odata['ncutout'][i] = self.ncutouts[i]
                odata['box_size'][i] = self.box_sizes[i]
                for icut in xrange(odata['ncutout'][i]):
                    odata['start_row'][i,icut] = self.start_rows[i][icut]

        fitsio.write(name,odata,extname='object_data')
        self.odata = odata

    def write(self,name,image_info,metadata,clobber=True):
        #clobber the file if needed
        if clobber and os.path.exists(name):
            os.remove(name)

        #write info tables
        self._write_object_data(name)
        fitsio.write(name,image_info,extname='image_info')
        fitsio.write(name,metadata,extname='metadata')

        #write pixel info
        self.imgpixels = np.array(self.imgpixels,dtype='f4')
        fitsio.write(name,self.imgpixels,extname='image_cutouts',compress='RICE')
        self.imgpixels = list(self.imgpixels)
        
        self.wgtpixels = np.array(self.wgtpixels,dtype='f4')
        fitsio.write(name,self.wgtpixels,extname='weight_cutouts',compress='RICE')
        self.wgtpixels = list(self.wgtpixels)
        
        self.segpixels = np.array(self.segpixels,dtype='i2')
        fitsio.write(name,self.segpixels,extname='seg_cutouts',compress='RICE')
        self.segpixels = list(self.segpixels)

        
        
        
        
