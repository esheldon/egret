import os
import glob
from distutils.core import setup

setup(name="justok_des",
            version="0.1",
            description="code to make DES-like GREAT3-like sims",
            license = "3-clause BSD",
            author="Matthew R. Becker w/ code from GREAT3 team",
            author_email="becker.mr@gmail.com",
            packages=['justok_des','justok_des.makers','justok_des.makers.great3_cosmos_gals'])
