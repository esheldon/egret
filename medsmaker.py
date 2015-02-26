import os
import numpy as np
import fitsio

__all__ = ['MemoryMEDSMaker','DiskMEDSMaker']

class MemoryMEDSMaker(object):
    """
    Make MEDS files, buffering all data in memory.
    
    Example:
    
        extra_data = [('cosmos_id','i8')]
        extra_percutout_data = [('id_psf','i4')]
        mm = MemoryMEDSMaker(extra_data=extra_data,extra_percutout_data=extra_percutout_data)
        objinfo = dict(id=id,number=number,file_id=file_id,orig_row=orig_row,orig_col=orig_col,
                       orig_start_row=orig_start_row,orig_start_col=orig_start_col,
                       dudrow=dudrow,dudcol=dudcol,dvdrow=dvdrow,dvdcol=dvdcol,
                       cutout_row=cutout_row,cutout_col=cutout_col,
                       cosmos_id=cosmos_id,id_psf=id_psf)    
        mm.add_object(objinfo,[im1,im2,im3],[wgt1,wgt2,wgt3],[seg1,seg2,seg3])
        mm.write('medstest.fit')    
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
        dlist = [('number', 'i4'),
                 ('ncutout', 'i4'),
                 ('id', 'i4'),
                 ('box_size', 'i4'),
                 ('file_id', 'i4', (nmax,)),
                 ('start_row', 'i4', (nmax,)),
                 ('orig_row', 'f8', (nmax,)),
                 ('orig_col', 'f8', (nmax,)),
                 ('orig_start_row', 'i4', (nmax,)),
                 ('orig_start_col', 'i4', (nmax,)),
                 ('dudrow', 'f8', (nmax,)),
                 ('dudcol', 'f8', (nmax,)),
                 ('dvdrow', 'f8', (nmax,)),
                 ('dvdcol', 'f8', (nmax,)),
                 ('cutout_row', 'f8', (nmax,)),
                 ('cutout_col', 'f8', (nmax,))]

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
            "The filen ame was not given and the internal file name is not defined for fpack!"
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
            assert np.array_equal(odata['oi'][tag],m[tag][ind,0:ncutout]), \
                "Error writing tag '%s' in MEDSMaker!" % tag
            
        for i in xrange(ncutout):
            for tag,tpe in zip(['imgs','wgts','segs'],['image','weight','seg']):
                assert np.array_equal(odata[tag][i],m.get_cutout(ind,i,type=tpe)), \
                    "Error in writing cutout %d for type '%s' in MEDSMaker!" % (i,tpe)

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
                comp_odata_medsind(obslist[ind][0],obslist[ind][1],obslist[ind][2],mind,m)

    os.remove('medstest.fit')
                
    print "MemoryMEDSMaker passed all tests!"

        
class DiskMEDSMaker(object):
    """
    Make MEDS files given a defined set of objects and # of pixels
    
    Exmaple:
    
    # make object data
    # must have at least box_size and ncutout fields, but
    # you probably want to set an ID
    object_data = ...
    
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
    mm.add_object(objinfo,imgs,wgts,segs,...)
    
    # in the ... above one must specify exactly one way to 
    # add an object.
    # the ways are order of preference/precedence:
    #  0) specify start_row as keyword
    #  1) specify id as keyword
    #  2) specify number as keyword
    #  3) use one of 0) 1) or 2) (in that order) but pulling data 
    #     from objinfo
    # 
    # In the case of ids or numbers, the internal object_data table 
    # is searched with np.where and the location specified is used.
    # This is probably slow, so passing start_row is better if you can 
    # do it.
    
    # one can buffer the pixels in memory to try and do more efficient I/O
    # if you do this, you have to clear the buffer at the end with
    mm.clear_buffer()
    
    # finally, one usually wants to fpack these files
    # assuming fpack is on your machine, there is a method that will 
    # fpack the file. It is possible to write to compressed files, but 
    # this is slow, so it is best to just wait until the end.
    options = list of options for fpack
    mm.fpack(options=options)
    """

    def __init__(self,fname,object_data,image_info=None,metadata=None):
        self.fname = fname
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
        
    def add_object(self,objinfo,imgs,wgts,segs,id=None,number=None, \
                   start_row=None,buffer_size=0):
        """
        Adds an object's images to the file.
        
        imgs, wgts, segs: lists with each cutout        
        objinfo: dict with all of the information needed for the 
            MEDS object_data extension
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
          3) use one of 0) 1) or 2) (in that order) but pulling data 
             from objinfo
         
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
            start_row_use = self.object_data['start_row'][q[0]]
        elif number is not None:
            q, = np.where(self.object_data['number'] == number)
            assert len(q) == 1, \
                "Could not find object by number or multiple objects with same number! number = %d, # found = %d" \
                % (number,len(q))
            start_row_use = self.object_data['start_row'][q[0]]
        elif objinfo is not None and 'start_row' in objinfo:
            start_row_use = objinfo['start_row']
        elif objinfo is not None and 'id' in objinfo:
            q, = np.where(self.object_data['id'] == objinfo['id'])
            assert len(q) == 1, \
                "Could not find object by id or multiple objects with same id! id = %d, # found = %d" \
                % (objinfo['id'],len(q))
            start_row_use = self.object_data['start_row'][q[0]]
        elif objinfo is not None and 'number' in objinfo:
            q, = np.where(self.object_data['number'] == objinfo['number'])
            assert len(q) == 1, \
                "Could not find object by number or multiple objects with same number! number = %d, # found = %d" \
                % (objinfo['number'],len(q))
            start_row_use = self.object_data['start_row'][q[0]]
        else:
            assert False,"No valid way to locate object in file given!"

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
            self._add_to_buffer(imgpix,wgtpix,segpix,start_row_use)
        else:
            # just write to disk right away
            self._write_pixels(start_row_use,[imgpix,wgtpix,segpix],['image_cutouts','weight_cutouts','seg_cutouts'])

    def _add_to_buffer(self,imgpix,wgtpix,segpix,start_row):
        """
        adds pixels to the buffer
        """
        loc = None

        # search for pixels to add onto
        for loc in self.img_buffer:
            if loc + len(self.img_buffer[loc]) == start_row:
                break

        if loc is not None:
            # if we found soemthing, then add on
            self.img_buffer[loc].extend(list(impix))
            self.wgt_buffer[loc].extend(list(wgtpix))
            self.seg_buffer[loc].extend(list(segpix))            
        else:
            # just put in buffer
            self.img_buffer[start_row] = list(impix)
            self.wgt_buffer[start_row] = list(wgtpix)
            self.seg_buffer[start_row] = list(segpix)
            self.num_in_buffer += len(impix)

    def _write_buffer(self):
        for start_row in self.img_buffer:
            pixlist = [self.img_buffer[start_row], \
                       self.wgt_buffer[start_row], \
                       self.seg_buffer[start_row]]
            extlist = ['image_cutouts','weight_cutouts','seg_cutouts']
            self._write_pixels(start_row,pixlist,extlist)
            
    def clear_buffer(self):
        """
        Clear the internal pixel buffer.
        """
        self._write_buffer()
            
    def _concat_pixels(self,imlist):
        pixels = []
        for im in imlist:
            pixels.extend(list(im.reshape(-1)))

        if len(pixels) > 0:
            dtype = imlist[0].dtype
            pixels = np.array(pixels,dtype=dtype)
        else:
            pixels = np.array([])
        return pixels
        
    def _write_pixels(self,start_row,pixlist,extlist):
        """
        subroutine to write pixels to extension
        """
        with fitsio.FITS(self.fname,'rw') as f:
            for pixels,extname in zip(pixlist,extlist):
                f[extname].write(pixels, start=[start_row])
        
    def _init_file(self):
        """
        create file on disk and fill with table data
        """
        
        with fitsio.FITS(self.fname,'rw',clobber=True) as f:
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
        nmax = -1
        for nm in vector_names:
            if nm in namesin:
                if nmax < 0:
                    nmax = len(object_data[nm][0,:])
                else:
                    assert nmax == len(object_data[nm][0,:]), \
                        "field '%s' in object_data is not the right shape!" % nm
        self.nmax = nmax
        assert nmax >= 2, \
            "One must have a maximum of at least 2 cutouts to avoid numpy tossing dimensions!"

        # now can make dtype
        dlist = [('number', 'i4'),
                 ('ncutout', 'i4'),
                 ('id', 'i4'),
                 ('box_size', 'i4'),
                 ('file_id', 'i4', (nmax,)),
                 ('start_row', 'i4', (nmax,)),
                 ('orig_row', 'f8', (nmax,)),
                 ('orig_col', 'f8', (nmax,)),
                 ('orig_start_row', 'i4', (nmax,)),
                 ('orig_start_col', 'i4', (nmax,)),
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
                assert din[loc][2] == tup[2] and din[loc][1][-2:] == tup[1], \
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

        # now fill out start_row
        loc = 0
        for i in xrange(Nobjs):
            if new_data['ncutout'] > 0:
                new_data['start_row'][i] = loc
                loc += new_data['box_size'][i]*new_data['box_size'][i]*new_data['ncutout'][i]
        self.npix = loc
        
        return new_data
            
        
        
                
        


    

def _disk_test():
    assert False,"DiskMEDSMaker is not define!"
    
        
def test():
    _mem_test()
    _disk_test()
