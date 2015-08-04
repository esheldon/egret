import os
import glob
from distutils.core import setup

setup(name="lensing_sims_tools",
            version="0.1",
            description="code to make lensing sims",
            license = "3-clause BSD",
            author="Matthew R. Becker w/ code from GREAT3 team",
            author_email="becker.mr@gmail.com",
            packages=['lensing_sims_tools','lensing_sims_tools.makers','lensing_sims_tools.blendmakers','lensing_sims_tools.makers.great3_cosmos_gals'])
