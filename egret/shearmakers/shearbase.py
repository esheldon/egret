import random

class ShearGeneratorBase(object):
    def sample(self):
        """
        get a sample shear
        """
        raise NotImplementedError("implement sample()")
