import random

class ShearGeneratorBase(object):
    def sample(self):
        """
        get a sample shear

        Should be a dict {'shear':(ngmix.Shape), 'meta':(dict)}}
        """
        raise NotImplementedError("implement sample()")
