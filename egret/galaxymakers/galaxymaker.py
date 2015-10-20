from .. import shearmakers

class GalaxyMaker(object):
    def __init__(self):
        raise NotImplementedError

    def get_galaxy(self):
        raise NotImplementedError

    def get_shear(self):
        return self.shearpdf.sample()

    def get_galaxy_pair(self):
        raise NotImplementedError

    def _setup(self, **kw):
        self.conf = {}
        self.conf.update(kw)

        self._setup_shears()

    def _setup_shears(self):
        shconf = self.conf['shear']

        if shconf['type'] == 'const-select':
            shearpdf = shearmakers.ConstShearSelector(shconf['shears'])
        else:
            raise NotImplementedError("shear type '%s' not "
                                      "implemented" % shconf['type'])
        
        self.shearpdf = shearpdf
