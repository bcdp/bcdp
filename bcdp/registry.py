import collections
import warnings

class Registry(collections.MutableMapping):
    """Global registry containing classes and functions.

    This mechanism allows for custom functions and usecases to become directly
    integrated into bcdp
    """
    def __init__(self, parent=None, **kwargs):
        self.store = dict()
        self.update(**kwargs)  # use the free update to set keys
        self.parent = parent

    def __getitem__(self, key):
        keys = key.split('.')
        reg = self
        for k in keys[:-1]:
            reg = reg.store[self.__keytransform__(k)]
        return reg.store[self.__keytransform__(keys[-1])]

    def __setitem__(self, key, value):
        keys = key.split('.')
        reg = self
        for k in keys[:-1]:
            if k not in reg.keys():
                reg.store[k] = Registry(parent=reg)
            reg = reg[k]
        key = keys[-1]
        if key in reg.store:
            warnings.warn('Warning: Resetting registry key {} with value {}'
                          .format(key, value))
        reg.store[key] = value

    def __delitem__(self, key):
        raise ValueError('Cannot remove keys')

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return repr(self.store)

    def __keytransform__(self, key):
        return key


def register(key):
    """Register callable object to global registry.

    This is primarily used to wrap classes and functions into the bcdp pipeline.
    It is also the primary means for which to customize bcdp for your own
    usecases when overriding core functionality is required.

    Parameters
    ----------
    key : str
        Key for obj in registry. Append periods ('.') to navigate the registry
        tree. Example: 'data_source.rcmed'

    Returns
    -------
    dec : function
        Generic decorator which returns the wrapped class or function.
    """
    def dec(obj):
        registry[key] = obj
        return obj
    return dec


registry = Registry()
