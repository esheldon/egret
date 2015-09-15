
class Observation(dict):
    def __init__(self,image=None,weight=None,seg=None,psf=None,**kwargs):
        self.update(kwargs)
        self.image = image
        self.weight = weight
        self.seg = seg
        self.psf = psf

    def __repr__(self):
        ostr = ""
        if self.image is not None:
            ostr += 'image:           [%d,%d] %s\n' % (self.image.shape[0],self.image.shape[1],self.image.dtype.descr[0][1])
        else:
            ostr += 'image:           None\n'

        if self.weight is not None:
            ostr += 'weight:          [%d,%d] %s\n' % (self.weight.shape[0],self.weight.shape[1],self.weight.dtype.descr[0][1])
        else:
            ostr += 'weight:          None\n'
            
        if self.seg is not None:
            ostr += 'seg:             [%d,%d] %s\n' % (self.seg.shape[0],self.seg.shape[1],self.seg.dtype.descr[0][1])
        else:
            ostr += 'seg:             None\n'

        ostr += 'meta data:       '
        ostr += super(Observation,self).__repr__()
        ostr += '\n'

        if self.psf is not None:
            ostr += 'psf:             <%s.%s object at %s>' % (
                self.psf.__class__.__module__,
                self.psf.__class__.__name__,
                hex(id(self.psf)))
        else:
            ostr += 'psf:             None\n'

        if ostr[-1] == '\n':
            ostr = ostr[:-1]
            
        return ostr

    def __str__(self):
        return self.__repr__()
