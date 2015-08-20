import os
import glob
from distutils.core import setup

setup(name="egret",
            version="0.1",
            description="code to make lensing sims",
            license = "3-clause BSD",
            author="Matthew R. Becker w/ code from GREAT3 team",
            author_email="becker.mr@gmail.com",
            packages=['egret','egret.makers',
                      'egret.blendmakers',
                      'egret.makers.great3_cosmos_gals'])
