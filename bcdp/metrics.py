import abc
from .registry import register

class Metric:
    """Calculates and plots metrics for an ensemble of datasets.
    
    All metrics have two operations:
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, func, core='xarray'):
        """Metric constructor.
        
        Parameters
        ----------
        func : Callable
            Functiond defining metric to be applied to each ensemble member.
            Signature should be func(array_like, *args, **kwargs)
        binary : {True, False}
            If True, assume func is a binary metric, with the 2nd argument
            being the reference dataset. In this case, func's signature should
            be func(target, reference, *args, **kwargs).
        """
        self.func = func
        self.binary = binary
    
    def apply(self, ens, **kwargs):
        """Applies metric calculation to each ensemble member.
        
        Parameters
        ----------
        ens : bcdp.Ensemble
            Ensemble of datasets.
        kwargs :
        """
        pass
    
    @abc.abstractmethod
    def plot(self, ens):
        pass
        
    def __call__(self, ens):
        self.calculate(ens)
        self.plot(ens)
        
    
    
