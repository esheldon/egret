import numpy
import random
from .shearbase import ShearGeneratorBase

class ConstShearSelector(ShearGeneratorBase):
    """
    Select from a set of user-input shears, e.g. a list
    of lists

        shears=[ [ 0.01, 0.00],
                 [-0.15, 0.12],
                 [ 0.30, 0.10] ]

    or a list of ngmix.Shape

        shears = [ s1, s2, s3, .... ]

    """
    def __init__(self, shears):
        from ngmix import Shape

        if not isinstance(shears, list):
            raise ValueError("shears must be a list")

        if not isinstance(shears[0], (list,Shape)):
            shears = [shears]

        shlist=[]
        for s in shears:
            if isinstance(s, Shape):
                sh = s
            else:
                if len(s) != 2:
                    raise ValueError("shears must be a ngmix.Shape "
                                     "or a pair, e.g. [s1,s2], got %s" % s)
                
                sh=Shape(s[0],s[1])

            shlist.append( sh )

        self.nshear=len(shlist)
        self.shears=shlist

    def sample(self):
        """
        get a dict with a shear instance

        returns
        -------
        sdict: dict
            keys in the dict are 

            shear: an ngmix.Shape instance
            shear_index: integer into the original input shear list
        """
        ri = random.randint(0, self.nshear-1)
        sh = self.shears[ri]

        meta = {'shear_index':ri}

        out = {'shear':sh, 'meta': meta}
        return out

def generate_great3_style_shears(minshear, maxshear, nshear):
    """
    In great3, shears were generated uniformly in |shear| from
    some min value to some max value.
    """
    g = numpy.random.uniform(low=minshear, high=maxshear, size=nshear)
    theta = numpy.random.uniform(low=0.0, high=numpy.pi*2, size=nshear)

    g1=g*numpy.cos(2.0*theta)
    g2=g*numpy.sin(2.0*theta)

    return g1, g2

def print_great3_style_shears(minshear, maxshear, nshear):
    """
    print the shears for copying into the yaml file
    """
    g1,g2=generate_great3_style_shears(minshear, maxshear, nshear)

    print '  shears: ['
    for i in xrange(g1.size):
        print '      [%g, %g],' % (g1[i],g2[i])
    print '    ]'
