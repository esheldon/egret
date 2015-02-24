import numpy as np

class OORandom(object):
    """
    Object Oriented Interface to numpy's random number generator
    
    Allows for the user to get multiple repeatable sequences from numpy.random    
    """
    def __init__(self,seed,state=None):
        if state is None:
            np.random.seed(seed)
            self.state = np.random.get_state()
        else:
            self.state = state

    #basic pattern is to
    # 1) set state
    # 2) draw randoms
    # 3) get state

    #WARNING
    # The order in which things are drawn matters.
    #END OF WARNING
    
    def uniform(self,low=0.0,high=1.0,size=None):
        np.random.set_state(self.state)
        rands = np.random.uniform(low=low,high=high,size=size)
        self.state = np.random.get_state()
        return rands

    def normal(self,loc=0.0,scale=1.0,size=None):
        np.random.set_state(self.state)
        rands = np.random.normal(loc=loc,scale=scale,size=size)
        self.state = np.random.get_state()
        return rands

    def standard_t(self,df,size=None):
        np.random.set_state(self.state)
        rands = np.random.student_t(df,size=size)
        self.state = np.random.get_state()
        return rands

    #add your favorite from numpy.random here
    
def test():
    # number of rngs to test and # of calls 
    seeds = [1,2,3]
    nseeds = len(seeds)
    ncalls = 10000

    #get results for each generator
    base_res = {}
    for seed in seeds:
        rng = OORandom(seed)
        base_res[seed] = []
        for i in xrange(ncalls):
            base_res[seed].append(rng.uniform())


    #now call randomly 
    test_res = {}
    rngs = {}
    for seed in seeds:
        rngs[seed] = OORandom(seed)
        test_res[seed] = []
    for i in xrange(ncalls):
        for seed in np.random.permutation(seeds):
            test_res[seed].append(rngs[seed].uniform())

    #compare
    for seed in seeds:
        assert set(base_res[seed]) == set(test_res[seed]),"RNG w/ seed %d failed!" % seed

    print "OORandom passed all tests!"
            

    
