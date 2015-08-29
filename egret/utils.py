#!/usr/bin/env python
import numpy as np

def get_fft_sizes(min_size=32,max_size=256,facs=[2,3],minpows=[1,0]):
    """
    produce FFT friendly stamp sizes with only powers of facs
    """
    maxpows = [int(np.ceil(np.log(max_size)/np.log(fac))) for fac in facs]

    sizes = [1]
    for fac,minpow,maxpow in zip(facs,minpows,maxpows):
        vals = [fac**i for i in range(minpow,maxpow+1)]
        new_sizes = []
        for sz in sizes:
            for val in vals:
                new_sizes.append(val*sz)
        sizes = new_sizes

    sizes = [sz for sz in sizes if sz >= min_size and sz <= max_size]
    sizes = sorted(sizes)
    return sizes
        
