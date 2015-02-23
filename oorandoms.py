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

    def normal(loc=0.0,scale=1.0,size=None):
        np.random.set_state(self.state)
        rands = np.random.normal(loc=loc,scale=scale,size=size)
        self.state = np.random.get_state()
        return rands

    def standard_t(df,size=None):
        np.random.set_state(self.state)
        rands = np.random.student_t(df,size=size)
        self.state = np.random.get_state()
        return rands

    #add your favorite from numpy.random here
    
