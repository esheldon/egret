import os
import glob
from distutils.core import setup

scripts=glob.glob('./bin/*')
scripts = [os.path.basename(f) for f in scripts if f[-1] != '~']
scripts=[os.path.join('bin',s) for s in scripts]

setup(name="egret",
      version="0.1",
      description="code to make lensing sims",
      license = "3-clause BSD",
      author="Matthew R. Becker w/ code from GREAT3 team",
      author_email="becker.mr@gmail.com",
      scripts=scripts,
      packages=['egret',
                'egret.galaxymakers',
                'egret.galaxymakers.great3_cosmos_gals',
                'egret.blendmakers',
                'egret.psfmakers'])
                
