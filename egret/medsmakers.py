import os
import numpy as np
import fitsio
import time

__all__ = ['MemoryMEDSMaker','DiskMEDSMaker']

class MemoryMEDSMaker(object):
    """
    Make MEDS files, buffering all data in memory.
    
    Example:
    
        extra_data = [('cosmos_id','i8')]
        extra_percutout_data = [('id_psf','i8')]
        mm = MemoryMEDSMaker(extra_data=extra_data,extra_percutout_data=extra_percutout_data)
        objinfo = dict(id=id,number=number,file_id=file_id,orig_row=orig_row,orig_col=orig_col,
                       orig_start_row=orig_start_row,orig_start_col=orig_start_col,
                       dudrow=dudrow,dudcol=dudcol,dvdrow=dvdrow,dvdcol=dvdcol,
                       cutout_row=cutout_row,cutout_col=cutout_col,
                       cosmos_id=cosmos_id,id_psf=id_psf)    
        mm.add_object(objinfo,[im1,im2,im3],[wgt1,wgt2,wgt3],[seg1,seg2,seg3])
        mm.write('medstest.fit')    
        mm.fpack(options=options) #pack the file with fpack

    The columns are (from the meds docs - https://github.com/esheldon/meds) 
    
    id                 i8       id from coadd catalog
    ncutout            i8       number of cutouts for this object
    box_size           i8       box size for each cutout
    file_id            i8[NMAX] zero-offset id into the file names in the 
    second extension
    start_row          i8[NMAX] zero-offset, points to start of each cutout.
    orig_row           f8[NMAX] zero-offset position in original image
    orig_col           f8[NMAX] zero-offset position in original image
    orig_start_row     i8[NMAX] zero-offset start corner in original image
    orig_start_col     i8[NMAX] zero-offset start corner in original image
    cutout_row         f8[NMAX] zero-offset position in cutout imag
    cutout_col         f8[NMAX] zero-offset position in cutout image
    dudrow             f8[NMAX] jacobian of transformation 
    row,col->ra,dec tangent plane (u,v)
    dudcol             f8[NMAX]
    dvdrow             f8[NMAX]
    dvdcol             f8[NMAX]

    """
    
    def __init__(self,extra_data=None,extra_percutout_data=None):
        self.fname = None
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
        dlist = [('number', 'i8'),
                 ('ncutout', 'i8'),
                 ('id', 'i8'),
                 ('box_size', 'i8'),
                 ('file_id', 'i8', (nmax,)),
                 ('start_row', 'i8', (nmax,)),
                 ('orig_row', 'f8', (nmax,)),
                 ('orig_col', 'f8', (nmax,)),
                 ('orig_start_row', 'i8', (nmax,)),
                 ('orig_start_col', 'i8', (nmax,)),
                 ('dudrow', 'f8', (nmax,)),
                 ('dudcol', 'f8', (nmax,)),
                 ('dvdrow', 'f8', (nmax,)),
                 ('dvdcol', 'f8', (nmax,)),
                 ('cutout_row', 'f8', (nmax,)),
                 ('cutout_col', 'f8', (nmax,)),
                 ('shear_index','i8')]

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

    def fpack(self,options='-t 10240,1',fname=None):
        """
        fpack the file.
        
        The default options were suggested by E. Sheldon to be 
        good for 1d images.
        """
        if fname is None:
            if self.fname is not None:
                fname = self.fname

        assert fname is not None, \
            "The file name was not given and the internal file name is not defined for fpack!"

        if os.path.exists(fname+'.fz'):
            os.remove(fname+'.fz')
            
        os.system('fpack %s %s' % (options,fname))        
        
    def write(self,name,image_info=None,metadata=None,clobber=True,compress=None):
        """
        Writes data to a MEDS file <name>.
        
        The image_info table and metadata can be supplied if needed.
        The file will be clobbered by default.        
        """
        self.fname = name
        
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

def _mem_test():
    """
    Test suite for MemoryMEDSMaker
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
                oi[tag] = rng.uniform(low=0,high=1e6,size=ncutout).astype('i8')
            else:
                oi[tag] = np.zeros(ncutout,dtype='i8')

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
            assert np.array_equal(odata['oi'][tag],m[tag][ind,0:ncutout]), \
                "Error writing tag '%s' in MEDSMaker!" % tag
            
        for i in xrange(ncutout):
            for tag,tpe in zip(['imgs','wgts','segs'],['image','weight','seg']):
                assert np.array_equal(odata[tag][i],m.get_cutout(ind,i,type=tpe)), \
                    "Error in writing cutout %d for type '%s' in MEDSMaker!" % (i,tpe)

        return True

    #do tests
    rng = np.random.RandomState(12345)
    
    obslist = []    
    obslist.append((make_test_obs(rng,0,(0,0),blank=True),0,(0,0)))
    #obslist.append((make_test_obs(rng,1,(64,64)),1,(64,64)))
    obslist.append((make_test_obs(rng,1,(64,64)),1,(64,64)))
    obslist.append((make_test_obs(rng,11,(13,13)),11,(13,13)))

    extra_data = [('cosmos_id','i8')]
    extra_percutout_data = [('id_psf','i8')]

    import meds
    
    for perm in [[0,1],[1,0],[0,2],[2,0],[1,2],[2,1],[0,1,2],[2,0,1],[1,2,0],[0,2,1],[1,0,2],[2,1,0]]:
        mm = MemoryMEDSMaker(extra_data=extra_data,extra_percutout_data=extra_percutout_data)

        for ind in perm:
            mm.add_object(obslist[ind][0]['oi'],
                          obslist[ind][0]['imgs'],
                          obslist[ind][0]['wgts'],
                          obslist[ind][0]['segs'])
        mm.write('medstest.fit')    
        del mm

        with meds.MEDS('medstest.fit') as m: 
            for mind,ind in enumerate(perm):
                assert comp_odata_medsind(obslist[ind][0],obslist[ind][1],obslist[ind][2],mind,m), \
                    "Error in writing object %d to MEDS!" % ind

    os.remove('medstest.fit')
                
    print "MemoryMEDSMaker passed all tests!"

        
class DiskMEDSMaker(object):
    """
    Make MEDS files given a defined set of objects and # of pixels
    
    Example:
    
    # make object data
    # must have at least box_size and ncutout fields, but
    # you probably want to set an ID
    object_data = ...
    
    The columns are (from the meds docs - https://github.com/esheldon/meds) 
    
    id                 i8       id from coadd catalog
    ncutout            i8       number of cutouts for this object
    box_size           i8       box size for each cutout
    file_id            i8[NMAX] zero-offset id into the file names in the 
    second extension
    start_row          i8[NMAX] zero-offset, points to start of each cutout.
    orig_row           f8[NMAX] zero-offset position in original image
    orig_col           f8[NMAX] zero-offset position in original image
    orig_start_row     i8[NMAX] zero-offset start corner in original image
    orig_start_col     i8[NMAX] zero-offset start corner in original image
    cutout_row         f8[NMAX] zero-offset position in cutout imag
    cutout_col         f8[NMAX] zero-offset position in cutout image
    dudrow             f8[NMAX] jacobian of transformation row,col->ra,dec tangent plane (u,v)
    dudcol             f8[NMAX]
    dvdrow             f8[NMAX]
    dvdcol             f8[NMAX]    
    
    # specify file name and optionally image_info and metadata tables
    fname = 'blah'    
    ii = ...
    mm = ...
    
    # then make the DiskMEDSMaker object
    # this will write the full uncompressed MEDS file to disk
    mm = DiskMEDSMaker(fname,object_data,image_info=ii,metadata=md)
    
    # to add images, you just do
    imgs = [im1,...]
    wgts = [wgt1,...]
    segs = [seg1,...]
    mm.add_image_data(imgs,wgts,segs,...)
    
    # in the ... above one must specify exactly one way to 
    # add an object.
    # the ways are order of preference/precedence:
    #  0) specify start_row as keyword
    #  1) specify id as keyword
    #  2) specify number as keyword
    # 
    # In the case of ids or numbers, the internal object_data table 
    # is searched with np.where and the location specified is used.
    # This is probably slow, so passing start_row is better if you can 
    # do it.
    
    # one can buffer the pixels in memory to try and do more efficient I/O
    # if you do this, you have to clear the buffer at the end with
    mm.flush_buffer()
    
    # finally, one usually wants to fpack these files
    # assuming fpack is on your machine, there is a method that will 
    # fpack the file. It is possible to write to compressed files, but 
    # this is slow, so it is best to just wait until the end.
    options = list of options for fpack
    mm.fpack(options=options)


    """
    
    def __init__(self,fname,object_data,image_info=None,metadata=None,nmax=None,verbose=False):
        self.fname = fname
        self.verbose = verbose
        if nmax is None:
            self.nmax = -1
        else:
            self.nmax = nmax
        self.object_data = self._init_object_data(object_data)
        self.image_info = image_info
        self.metadata = metadata
        self._init_file()

        # the buffers are dicts keyed on the starting
        # location of the pixels in the dict.
        # when adding to the buffer, one can check if the pixels
        # you have are at the end of some other part of it
        # by looking at the keys you have and the lengths of the pixels
        # in the buffer.
        # we use python lists for speed
        self.img_buffer = {}
        self.wgt_buffer = {}
        self.seg_buffer = {}
        self.num_in_buffer = 0
        self.add_time = 0.0
        self.write_time = 0.0
        self.buff_write_time = 0.0
        self.toarray_time = 0.0
        self.num_writes = 0
        
    def fpack(self,options='-t 10240,1'):
        """
        fpack the MEDS file
        
        Note that the filename is not updated, so if 
        more data is added, then the fpacked file will not 
        be the same.
        
        The default options were suggested by E. Sheldon to be 
        good for 1d images.
        """
        os.system('fpack %s %s' % (options,self.fname))        
    
    def get_object_data(self):
        """
        get a copy of the object_data
        """
        return self.object_data.copy()    
        
    def add_image_data(self,imgs,wgts,segs,id=None,number=None, \
                 start_row=None,buffer_size=0):
        """
        Adds an object's images to the file.
        
        imgs, wgts, segs: lists with each cutout        
        id: object id to find location
        number: object number to find location
        start_row: location in MEDS file
        buffer_size: set to a nonzero value to buffer buffer_size pixels in memory
            Some attempt is made to concatenate I/O calls, but all of this is done 
            in python and so may not be very efficient.
            Note that this is pixels per imgs, wgts and segs separately, so total
            amount of pixels buffered is 3x larger.

        Objects get added by searching for location in MEDS file from input 
        data in the following order of priority:
        
          0) specify start_row as keyword
          1) specify id as keyword
          2) specify number as keyword
        
        In the case of ids or numbers, the internal object_data table 
        is searched with numpy.where and the location specified is used.
        This is probably slow, so passing start_row is better if you can 
        do it.        
        """

        # find location in file
        start_row_use = None
        if start_row is not None:
            start_row_use = start_row
        elif id is not None:
            q, = np.where(self.object_data['id'] == id)
            assert len(q) == 1, \
                "Could not find object by id or multiple objects with same id! id = %d, # found = %d" \
                % (id,len(q))
            start_row_use = self.object_data['start_row'][q[0],0]
        elif number is not None:
            q, = np.where(self.object_data['number'] == number)
            assert len(q) == 1, \
                "Could not find object by number or multiple objects with same number! number = %d, # found = %d" \
                % (number,len(q))
            start_row_use = self.object_data['start_row'][q[0],0]
        else:
            assert False,"No valid way to locate object in file given!"

        # write data if present
        if len(imgs) == len(wgts) and len(wgts) == len(segs) and len(imgs) > 0:
            # now write the data
            imgpix = self._concat_pixels(imgs)
            wgtpix = self._concat_pixels(wgts)
            segpix = self._concat_pixels(segs)
            
            if buffer_size > 0:
                # if buffering, always check that don't
                # have too much
                if self.num_in_buffer > buffer_size:
                    self._write_buffer()                

                # now add pixels
                self._add_to_buffer(imgpix,wgtpix,segpix,start_row_use,buffer_size)
            else:
                # just write to disk right away
                cnt = len(imgpix)
                self.toarray_time -= time.time()
                pixlist = [np.fromiter(imgpix,dtype='f4',count=cnt), \
                           np.fromiter(wgtpix,dtype='f4',count=cnt), \
                           np.fromiter(segpix,dtype='i2',count=cnt)]
                self.toarray_time += time.time()
                extlist =  ['image_cutouts', \
                            'weight_cutouts', \
                            'seg_cutouts']                
                self._write_pixels(start_row_use,pixlist,extlist)

    def _add_to_buffer(self,imgpix,wgtpix,segpix,start_row,buffer_size):
        """
        adds pixels to the buffer
        """

        if self.num_in_buffer + len(imgpix) > buffer_size:
            self._write_buffer()

        self.add_time -= time.time()
            
        # search for pixels to add onto
        loc = None
        for tloc in self.img_buffer:
            if tloc + len(self.img_buffer[tloc]) == start_row:
                loc = tloc
                break

        if loc is not None:
            # if we found soemthing, then add on
            self.img_buffer[loc].extend(imgpix)
            self.wgt_buffer[loc].extend(wgtpix)
            self.seg_buffer[loc].extend(segpix)
        else:
            # just put in buffer
            self.img_buffer[start_row] = imgpix
            self.wgt_buffer[start_row] = wgtpix
            self.seg_buffer[start_row] = segpix
        self.num_in_buffer += len(imgpix)

        # this is all well and good, but there might
        # a key which could go at the end of our latest
        # edition

        # first get where data was put
        if loc is None:
            loc = start_row

        # now look at all dicts and see if they could go at end of loc
        oloc = None
        for tloc in self.img_buffer:
            if loc + len(self.img_buffer[loc]) == tloc:
                oloc = tloc
                break

        if oloc is not None:
            # extend
            self.img_buffer[loc].extend(self.img_buffer[oloc])
            self.wgt_buffer[loc].extend(self.wgt_buffer[oloc])
            self.seg_buffer[loc].extend(self.seg_buffer[oloc])

            # remove keys
            del self.img_buffer[oloc]
            del self.wgt_buffer[oloc]
            del self.seg_buffer[oloc]

        self.add_time += time.time()
            
    def _write_buffer(self):
        """
        write the buffers to disk
        """

        if self.num_in_buffer > 0:
            extlist = ['image_cutouts','weight_cutouts','seg_cutouts']
            types = ['f4','f4','i2']
            start_rows = sorted(self.img_buffer.keys())
            with fitsio.FITS(self.fname,'rw') as f:
                for extname,buff,tp in zip(extlist, \
                                           [self.img_buffer, \
                                            self.wgt_buffer, \
                                            self.seg_buffer],types):
                    for start_row in start_rows:
                        cnt = len(buff[start_row])                    
                        if self.num_in_buffer > 1e6 and self.verbose:
                            if tp == 'f4':
                                print "    buffer size: %0.2lf MB" % (cnt*4.0/1024.0**2.0)
                            else:
                                print "    buffer size: %0.2lf MB" % (cnt*2.0/1024.0**2.0)
                        self.toarray_time -= time.time()
                        pixels = np.fromiter(buff[start_row],dtype=tp,count=cnt)
                        self.toarray_time += time.time()
                        self.buff_write_time -= time.time()
                        self.num_writes += 1
                        f[extname].write(pixels, start=[start_row])
                        self.buff_write_time += time.time()

            # reset
            self.img_buffer = {}
            self.wgt_buffer = {}
            self.seg_buffer = {}
            self.num_in_buffer = 0
        
    def flush_buffer(self):
        """
        Clear the internal pixel buffer.
        """
        self._write_buffer()
            
    def _concat_pixels(self,imlist):
        pixels = []
        for im in imlist:
            pixels.extend(list(im.reshape(-1)))
        return pixels
        
    def _write_pixels(self,start_row,pixlist,extlist):
        """
        subroutine to write pixels to extension
        """
        self.write_time -= time.time()
        
        with fitsio.FITS(self.fname,'rw') as f:
            for pixels,extname in zip(pixlist,extlist):
                self.num_writes += 1
                f[extname].write(pixels, start=[start_row])

        self.write_time += time.time()
                
    def _init_file(self):
        """
        create file on disk and fill with table data
        """

        if os.path.exists(self.fname):
            os.remove(self.fname)
        
        with fitsio.FITS(self.fname,'rw') as f:
            # first do tables
            f.write(self.object_data,extname='object_data')
            
            blank_data = np.zeros(1,dtype=[('blank_table','i8')])
            if self.image_info is not None:
                f.write(self.image_info,extname='image_info')
            else:
                f.write(blank_data,extname='image_info')

            if self.metadata is not None:
                f.write(self.metadata,extname='metadata')
            else:
                f.write(blank_data,extname='metadata')

            # now do images
            f.create_image_hdu(dims=[self.npix],dtype='f4',extname='image_cutouts')
            f.create_image_hdu(dims=[self.npix],dtype='f4',extname='weight_cutouts')
            f.create_image_hdu(dims=[self.npix],dtype='i2',extname='seg_cutouts')
        
    def _init_object_data(self,object_data):
        """
        make and verify object_data
        """
        # get basic props of input object_data
        din = object_data.dtype.descr
        namesin = []
        for tup in din:
            namesin.append(tup[0].lower())

        # get nmax and verify vector fields
        vector_names = ['file_id','orig_row','orig_col','orig_start_row','orig_start_col',
                        'dudrow','dudcol','dvdrow','dvdcol','cutout_row','cutout_col']        
        for nm in vector_names:
            if nm in namesin:
                loc = namesin.index(nm)
                assert len(din[loc]) == 3, \
                    "data type for '%s' is not correct!" % nm
                if self.nmax < 0:
                    self.nmax = len(object_data[nm][0,:])
                else:
                    assert self.nmax == len(object_data[nm][0,:]), \
                        "field '%s' in object_data is not the right shape!" % nm

        assert self.nmax != -1, \
            "One must pass enough information to determine nmax!"
        
        assert self.nmax >= 2, \
            "One must have a maximum of at least 2 cutouts to avoid numpy tossing dimensions!"
        
        # now can make dtype
        nmax = self.nmax
        dlist = [('number', 'i8'),
                 ('ncutout', 'i8'),
                 ('id', 'i8'),
                 ('box_size', 'i8'),
                 ('file_id', 'i8', (nmax,)),
                 ('start_row', 'i8', (nmax,)),
                 ('orig_row', 'f8', (nmax,)),
                 ('orig_col', 'f8', (nmax,)),
                 ('orig_start_row', 'i8', (nmax,)),
                 ('orig_start_col', 'i8', (nmax,)),
                 ('dudrow', 'f8', (nmax,)),
                 ('dudcol', 'f8', (nmax,)),
                 ('dvdrow', 'f8', (nmax,)),
                 ('dvdcol', 'f8', (nmax,)),
                 ('cutout_row', 'f8', (nmax,)),
                 ('cutout_col', 'f8', (nmax,))]

        # verify required entries present
        for tup in dlist:
            # we must have a box size and # of cutouts
            # we can make anything else
            if tup[0] in ['box_size','ncutout']:
                assert tup[0] in namesin,"field '%s' must be present in object_data!" % tup[0]

            #make sure entry is OK
            if tup[0] in namesin:
                loc = namesin.index(tup[0])
                if tup[0] in vector_names:
                    assert din[loc][2] == tup[2] and din[loc][1][-2:] == tup[1], \
                        "data type for '%s' is not correct!" % tup[0]
                else:
                    assert din[loc][1][-2:] == tup[1], \
                        "data type for '%s' is not correct!" % tup[0]

        # add other entries if needed
        remake_data = False
        for tup in dlist:
            if tup[0] not in namesin:
                din.append(tup)
                remake_data = True
                
        # now remake data if needed
        if remake_data:
            Nobjs = len(object_data)            
            new_data = np.zeros(Nobjs,dtype=din)
            for nm in namesin:
                new_data[nm] = object_data[nm]
        else:
            new_data = object_data
        Nobjs = len(new_data)
        
        # now fill out start_row
        loc = 0
        for i in xrange(Nobjs):
            for j in xrange(new_data['ncutout'][i]):
                new_data['start_row'][i,j] = loc
                loc += new_data['box_size'][i]*new_data['box_size'][i]
        self.npix = loc
        
        return new_data

    
def _disk_test(buff=25000000):
    """
    Test suite for DiskMEDSMaker
    """
    
    #function for test obs
    def make_test_obs(rng,ncutout,shape,nmax,blank=False,mindata=False):
        dlist = [('box_size','i8'),('ncutout','i8')]
        if not mindata:
            dlist.extend([('cosmos_id','i8'),
                          ('id_psf','i8',(nmax,)),
                          ('number', 'i8'),
                          ('id', 'i8'),
                          ('file_id', 'i8', (nmax,)),
                          ('start_row', 'i8', (nmax,)),
                          ('orig_row', 'f8', (nmax,)),
                          ('orig_col', 'f8', (nmax,)),
                          ('orig_start_row', 'i8', (nmax,)),
                          ('orig_start_col', 'i8', (nmax,)),
                          ('dudrow', 'f8', (nmax,)),
                          ('dudcol', 'f8', (nmax,)),
                          ('dvdrow', 'f8', (nmax,)),
                          ('dvdcol', 'f8', (nmax,)),
                          ('cutout_row', 'f8', (nmax,)),
                          ('cutout_col', 'f8', (nmax,))])
        
        oi = np.zeros(1,dtype=dlist)
        oi['box_size'][0] = shape[0]
        if not blank:
            oi['ncutout'][0] = ncutout
        
        odata = {}
        odata['oi'] = oi
        odata['imgs'] = []
        odata['wgts'] = []
        odata['segs'] = []
        
        if not mindata:
            for tag in ['id','number','cosmos_id']:
                oi[tag] = rng.randint(low=0,high=1e6)

            if ncutout > 0:
                for tag in ['id_psf','file_id','orig_start_row','orig_start_col']:
                    if not blank:
                        oi[tag][0,0:ncutout] = rng.randint(low=0,high=1e6,size=ncutout)
                    else:
                        oi[tag][0,0:ncutout] = np.zeros(ncutout,dtype='i8')

                for tag in ['orig_row','orig_col',
                            'dudrow','dudcol','dvdrow','dvdcol',
                        'cutout_row','cutout_col']:
                    if not blank:
                        oi[tag][0,0:ncutout] = rng.uniform(low=0,high=1e6,size=ncutout)
                    else:
                        oi[tag][0,0:ncutout] = np.zeros(ncutout,dtype='f8')

        if not blank:
            for i in xrange(ncutout):
                odata['imgs'].append(rng.uniform(low=0,high=1e6,size=shape).astype('f4'))
                odata['wgts'].append(rng.uniform(low=0,high=1e6,size=shape).astype('f4'))
                odata['segs'].append(rng.uniform(low=0,high=1e6,size=shape).astype('i2'))
        
        return odata

    #comp test obs to meds ind
    def comp_odata_medsind(odata,ncutout,shape,ind,m,mindata=False):
        assert m['ncutout'][ind] == ncutout,"Error computing 'ncutout' in MEDSMaker!"
        assert m['box_size'][ind] == shape[0],"Error computing 'box_size' in MEDSMaker!"

        if not mindata:
            for tag in ['id','number','cosmos_id']:
                assert odata['oi'][tag] == m[tag][ind],"Error writing tag '%s' in MEDSMaker!" % tag

            if ncutout > 0:
                for tag in ['orig_row','orig_col','orig_start_row','orig_start_col',
                            'dudrow','dudcol','dvdrow','dvdcol','file_id',
                            'cutout_row','cutout_col']:
                    assert np.array_equal(odata['oi'][tag][0,0:ncutout],m[tag][ind,0:ncutout]), \
                        "Error writing tag '%s' in MEDSMaker!" % tag
            
        for i in xrange(ncutout):
            for tag,tpe in zip(['imgs','wgts','segs'],['image','weight','seg']):
                assert np.array_equal(odata[tag][i],m.get_cutout(ind,i,type=tpe)), \
                    "Error in writing cutout %d for type '%s' in MEDSMaker!" % (i,tpe)
        return True

    # stuff for tests
    fname = 'medstest.fit'
    rng = np.random.RandomState(12345)
    import meds
    
    # make a list of entries
    # randomly permute them and use all different ways of writing them
    # then read back and check that you get back the right thing
    odict = {}
    object_data = []
    Nobj = 1000
    nmax = 13
    ids = rng.choice(1000000,size=Nobj,replace=False)
    nums = rng.choice(1000000,size=Nobj,replace=False)
    for i in xrange(Nobj):
        ncutout = int(rng.uniform(low=0.5,high=nmax+0.5))
        sh = int(rng.uniform(low=32+0.5,high=256+0.5))
        if rng.uniform() < 0.1:
            odata = make_test_obs(rng,ncutout,(sh,sh),nmax,blank=True)
        else:
            odata = make_test_obs(rng,ncutout,(sh,sh),nmax,blank=False)
        odata['oi']['id'][0] = ids[i]
        odata['oi']['number'][0] = nums[i]
        odict[ids[i]] = odata
        object_data.append(tuple(odata['oi'][0]))
    object_data = np.array(object_data,dtype=odata['oi'].dtype)
    object_data['id'] = ids
    object_data['number'] = nums

    import time
    for bf in [buff,0]:
        print 'testing buffering:',bf
        for i in [0,2]:
            t0 = time.time()
            mm = DiskMEDSMaker(fname,object_data,verbose=False)
            ood = mm.get_object_data()
            if i == 0:
                # write in order to test buffering
                pinds = np.arange(Nobj)
            elif i == 1:
                # write blocks backwards to test buffering
                pinds = np.arange(Nobj)
                pinds[0] = 3
                pinds[1] = 4
                pinds[2] = 5
                pinds[3] = 0
                pinds[4] = 1
                pinds[5] = 2
            else:
                # just do it randomly
                pinds = rng.permutation(Nobj)

            for ind in pinds:
                assert ood['id'][ind] == object_data['id'][ind], \
                    "output object_data does not match input!"

                # randomly turn on and off buffering
                if i == 4 and rng.uniform() < 0.5:
                    bf_fac = 0
                else:
                    bf_fac = 1
                
                chc = int(rng.uniform(low=0,high=3.0))
                if chc == 0:
                    mm.add_image_data(odict[object_data['id'][ind]]['imgs'], \
                                      odict[object_data['id'][ind]]['wgts'], \
                                      odict[object_data['id'][ind]]['segs'], \
                                      id=object_data['id'][ind],buffer_size=bf*bf_fac)
                elif chc == 1:
                    mm.add_image_data(odict[object_data['id'][ind]]['imgs'], \
                                      odict[object_data['id'][ind]]['wgts'], \
                                      odict[object_data['id'][ind]]['segs'], \
                                      number=object_data['number'][ind],buffer_size=bf*bf_fac)
                else:
                    mm.add_image_data(odict[object_data['id'][ind]]['imgs'], \
                                      odict[object_data['id'][ind]]['wgts'], \
                                      odict[object_data['id'][ind]]['segs'], \
                                      start_row=ood['start_row'][ind,0],buffer_size=bf*bf_fac)
            mm.flush_buffer()
            t0 = time.time() - t0
            print 'make time: %lf seconds' % t0
            print "    add time:",mm.add_time
            print "    buff write time:",mm.buff_write_time
            print "    write time:",mm.write_time
            print "    to array time:",mm.toarray_time
            print "    # of writes:",mm.num_writes
            
            t0 = time.time()
            with meds.MEDS(fname) as m:
                for ind in xrange(Nobj):
                    assert comp_odata_medsind(odict[object_data['id'][ind]], \
                                              object_data['ncutout'][ind], \
                                              (object_data['box_size'][ind],object_data['box_size'][ind]), \
                                              ind,m), \
                        "Error in writing index %d to MEDS!" % ind
            t0 = time.time() - t0
            print 'check time: %lf seconds' % t0

    #check duplicate indexes
    object_data['id'] = ids
    object_data['number'] = nums
    object_data['id'][100] = object_data['id'][0]
    ok = False
    try:
        ind = 100
        mm = DiskMEDSMaker(fname,object_data)
        mm.add_image_data(odict[object_data['id'][ind]]['imgs'], \
                          odict[object_data['id'][ind]]['wgts'], \
                          odict[object_data['id'][ind]]['segs'], \
                          id=object_data['id'][ind])
    except AssertionError as e:
        ok = True
        print "caught error on purpose:",e
        pass
    assert ok,"DiskMEDSMaker did not catch duplicate id!"

    object_data['id'] = ids
    object_data['number'] = nums
    object_data['number'][100] = object_data['number'][0]
    ok = False
    try:
        ind = 100
        mm = DiskMEDSMaker(fname,object_data)
        mm.add_image_data(odict[object_data['id'][ind]]['imgs'], \
                          odict[object_data['id'][ind]]['wgts'], \
                          odict[object_data['id'][ind]]['segs'], \
                          number=object_data['number'][ind])
    except AssertionError as e:
        ok = True
        print "caught error on purpose:",e
        pass
    assert ok,"DiskMEDSMaker did not catch duplicate number!"
                    
    os.remove(fname)
            
    ########################################
    # now check that error checking works
    ########################################
    
    # nmax <= 1
    for mindata in [False]:
        for blank in [True,False]:
            for ncutout in [0,1]:
                odata = make_test_obs(rng,ncutout,(16,16),1,blank=blank,mindata=mindata)
                ok = False
                try:
                    mm = DiskMEDSMaker(fname,odata['oi'])
                except AssertionError as e:
                    ok = True
                    print "caught error on purpose:",e
                    pass
                assert ok,"DiskMEDSMaker did not catch too small nmax!"

    # nmax not given
    for mindata in [True]:
        for blank in [True,False]:
            for ncutout in [0,1]:
                odata = make_test_obs(rng,ncutout,(16,16),5,blank=blank,mindata=mindata)
                ok = False
                try:
                    mm = DiskMEDSMaker(fname,odata['oi'])
                except AssertionError as e:
                    ok = True
                    print "caught error on purpose:",e
                    pass
                assert ok,"DiskMEDSMaker did not catch nmax not given!"

    # give wrong nmax
    odata = make_test_obs(rng,ncutout,(16,16),5)
    ok = False
    try:
        mm = DiskMEDSMaker(fname,odata['oi'],nmax=10)
    except AssertionError as e:
        ok = True
        print "caught error on purpose:",e
        pass
    assert ok,"DiskMEDSMaker did not catch wrong nmax!"

    # make odata without proper fields
    for dhave in [('box_size','i8'),('ncutout','i8')]:
        oi = np.zeros(1,dtype=[dhave])
        ok = False
        try:
            mm = DiskMEDSMaker(fname,oi,nmax=10)
        except AssertionError as e:
            ok = True
            print "caught error on purpose:",e
            pass
        assert ok,"DiskMEDSMaker did not catch missing required field %s!" % dhave[0]

    # make fields with wrong dtype
    for dhave in [('orig_row','i8'),('number','f8'),('dudrow','f8',(3,))]:
        oi = np.zeros(1,dtype=[dhave,('file_id','i8',(5,)),('box_size','i8'),('ncutout','i8'),('dudcol','f8',(5,))])
        ok = False
        try:
            mm = DiskMEDSMaker(fname,oi)
        except AssertionError as e:
            print "caught error on purpose:",e
            ok = True            
            pass
        assert ok,"DiskMEDSMaker did not catch improperly formatted field %s!" % dhave[0]

    print 'DiskMEDSMaker passed all tests!'

def test(buff=25000000):
    """
    test suite for MEDS makers
    
    buff: size of buffer to test for DiskMEDSMaker
    """
    _disk_test(buff=buff)
    _mem_test()
