import random
from .shearbase import ShearGeneratorBase

class ConstShearSelector(ShearGeneratorBase):
    """
    Select from a set of user-input shears, e.g.

    shears=[ [ 0.01, 0.00],
             [-0.15, 0.12],
             [ 0.30, 0.10] ]
    """
    def __init__(self, shears):
        from ngmix import Shape

        if not isinstance(shears, list):
            raise ValueError("shears must be a list")

        if not isinstance(shears[0], list):
            shears = [shears]

        shlist=[]
        for s in shears:
            if len(s) != 2:
                raise ValueError("shears must be a pair, e.g. "
                                 "[s1,s2], got %s" % s)
            
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

        out = {'shear':sh,
               'shear_index':ri}
        return out
