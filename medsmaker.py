import numpy as np
import fitsio

class MEDSMaker(object):
    """
    Object to make MEDS files.
    """

    def __init__(self,extra_object_data_tags=None):
        self.imglist = []
        self.wgtlist = []
        self.seglist = []
        self.objinfo = []
        self.num_objs = 0
        self.max_cutouts = -1
        self.npix_tot = 0
        self.extra_tags = extra_object_data_tags
        
        #some defaults
        self.NEG_INT_NONEVAL = -9999
        self.POS_INT_NONEVAL = 9999
        self.NEG_FLOAT_NONEVAL = -9999.0
        self.POS_FLOAT_NONEVAL = 9999.0
        
    def add_objects(self,objinfo,imgs,wgts,segs=None):
        self.objinfo.append(objinfo)
        self.imglist.append(imgs)
        self.wgtlist.append(wgts)
        self.seglist.append(segs)
        self.num_objs += 1
        if len(imgs) > self.max_cutouts:
            self.max_cutouts = len(imgs)

        for img in imgs:
            self.npix_tot += img.shape[0]*img.shape[1]
        
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

        return np.dtype(dlist)
    
    def _write_object_data(self,name):
        dtype = self._get_object_data_dtype(self.max_cutouts)
        odata = np.zeros(self.num_objs,dtype=dtype)
        names = odata.dtype.names
        odata['box_size'][:] = self.NEG_INT_NONEVAL
        odata['start_row'][:,:] = self.NEG_INT_NONEVAL

        #set info
        npix = 0        
        for i,objinfo in enumerate(self.objinfo):
            for name in names:
                if name in objinfo:
                    odata[name][i] = objinfo[name]

                #set tags that need special attention
                odata['ncutout'][i] = len(self.imglist[i])
                if odata['ncutout'][i] > 0:
                    odata['box_size'][i] = self.imglist[i][0].shape[0]
                    for icut in xrange(odata['ncutout'][i]):
                        odata['start_row'][i,icut] = npix
                        imshape = self.imglist[i][icut].shape
                        npix += imshape[0]*imshape[1]                        

        assert npix == self.npix_tot
        fitsio.write(name,odata,extname='object_data')
        self.odata = odata

    def _write_pixel_sequence(self,name,pixlist,extname,dtype,max_pixels_in_mem=25000000):
        pixels = []
        start = True
        with fitsio.FITS(name,'rw') as fits:
            for imgs in pixlist:
                #get pixels for image
                subpixels = []
                for img in imgs:
                    subpixels.extend(list(img.reshape(-1)))

                #write if needed
                if len(subpixels) + len(pixels) > max_pixels_in_mem:
                    pixels = np.array(pixels,dtype=dtype)
                    if start:
                        start = False
                        fits.write(pixels,extname=extname)
                    else:
                        fits[extname].append(pixels)
                    pixels = []
                    
                #add to list
                pixels.extend(subpixels)

            #do last set if needed
            if len(pixels) > 0:
                pixels = np.array(pixels,dtype=dtype)
                if start:
                    start = False
                    fits.write(pixels,extname=extname)
                else:
                    fits[extname].append(pixels)
                
    def _write_img_data(self,name):
        self._write_pixel_sequence(self,name,self.imglist,'image_cutouts','f4',max_pixels_in_mem=25000000)

    def _write_wgt_data(self,name):
        self._write_pixel_sequence(self,name,self.wgtlist,'weight_cutouts','f4',max_pixels_in_mem=25000000)

    def _write_seg_data(self,name):
        self._write_pixel_sequence(self,name,self.seglist,'seg_cutouts','i2',max_pixels_in_mem=25000000)
        
    def write(self,name,image_info,metadata,clobber=True):
        #clobber the file if needed
        if clobber and os.path.exists(name):
            os.remove(name)

        #write info tables
        self._write_object_data(name)
        fitsio.write(name,image_info,extname='image_info')
        fitsio.write(name,metadata,extname='metadata')

        self._write_img_data(name)
        self._write_wgt_data(name)
        self._write_seg_data(name)

        
        
