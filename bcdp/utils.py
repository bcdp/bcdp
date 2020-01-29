import functools
import types
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from .constants import ATTRS


def infer_freq(da):
    """Infer temporal resolution of a dataset.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to process.

    Returns
    -------
    str
        Inferred temporal resolution.
    """
    # If the data is uniformally spaced in time (eg, hourly, daily), the
    # temporal resolution is instantly inferred by pandas / xarray.
    #if encoding in ['noleap', 'all_leap', '365_day', '366_day', '360_day']:
    #    pass
    idx = da.indexes['time']
    freq = pd.infer_freq(idx)

    if not freq:
        # Because input might be seasonalized, it may no longer be uniform
        # in space. In that case, we will assume the data follows a pattern
        # and extrapolate from the first three time steps. We should try to
        # use a better approach for this in the future.
        idx = idx.to_series().apply(lambda dt: dt.replace(day=1, hour=0, minute=0))
        freq = pd.infer_freq(idx[:3])
    if freq:
        return freq
    else:
        raise ValueError('Could not infer frequency.')


def decode_month_units(ds):
    """Decode datasets with month time units.

    This is a workaround to xarray (and netCDF4 python) not supporting monthly
    time units. This is because months are not uniform, so this should only be
    used with monthly data, and the time values must be integer.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray Dataset object with time in units of month.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with decoded times.
    """
    # Find time variable
    for dim, coord in ds.coords.items():
        if ((hasattr(coord, 'axis') and coord.axis.upper() == 'T')
            or dim in ATTRS['T']['labels']):
            break
    else:
        return ds

    # Input data must be monthly and units must be integers
    units = coord.units
    if (not units.startswith('months')
        or not (coord.values == coord.values.astype(int)).all()):
        raise ValueError('Dataset time units cannot be inferred.')

    # Assume 1 month = 30 days
    coord.attrs['calendar'] = '360_day'
    coord.attrs['units'] = units.replace('months', 'days')
    coord.values *= 30
    ds = xr.decode_cf(ds)
    return ds


def grid_from_data(da):
    """Given a dataset, extract only the grid information.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to extract grid information from. Must have "ocw" conventions,
        ie 2D lats and lons variables.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing grid infromation.
    """
    ds = da.coords.to_dataset()
    return ds


def grid_from_res(res, domain):
    """Given lat/lon resolution, generate a grid array.

    Parameters
    ----------
    res : tuple
        Grid spacing (dlon, dlat) in degrees.
    domain : bcdp.Bounds
        Grid domain.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing grid infromation.
    """
    x = np.arange(*domain.lon_bnds, res[0])
    y = np.arange(*domain.lat_bnds, res[1])
    dims = ('y', 'x')
    lon, lat = np.meshgrid(x, y)
    ds = xr.Dataset(data_vars={'lon': (dims, lon), 'lat': (dims, lat)},
                    coords={'x': x, 'y': y})
    return ds


def build_series(datasets):
    """Build a pandas Series of xarray.DataArray objects.
    
    Used because the default constructor is very slow for building series
    of xarray objects.
    
    Parameters
    ----------
    datasets : list or dict
        Data structure consisting of DataArrays. For multiple variables,
        a dictionary keyed by variable name can be used.
        
    Returns
    -------
    pandas.Series
        New Series object containing data arrays.
    """
    dummies = [None for d in datasets]
    names = [d.name for d in datasets]
    s = pd.Series(data=dummies, index=names)
    for i, name in enumerate(names):
        s.at[name] = datasets[i]
    return s


def subset_1d(da, dim, domain):
    """Subsets data along a single dimension.
    
    Parameters
    ----------
    da : xarray.DataArray
        Data to subset.
    dim : str
        Name of dimension to subset along.
    domain : bcdp.Domain
        1D Domain object with .min and .max accessors.
        
    Returns
    -------
    xarray.DataArray
        Subsetted data.
    """
    selection = (da[dim] >= domain.min) & (da[dim] <= domain.max)
    return da.isel(**{dim: selection})


def get_dropped_varnames(f, varnames=None, group=None):
    """Determines list of variables that should not be loaded from the file.

    Parameters
    ----------
    f : str
        Data filename
    varnames : list of str, optional
        Names of variables to keep.
    group : str, optional
        Group name to load from netcdf file.

    Returns
    -------
    dropped_vars : list of str
        Names of variables to drop.
    """
    if not varnames:
        return []
    with Dataset(f) as nc:
        if group:
            nc = nc[group]
        vars = set(nc.variables.keys())
    dropped_vars = list(vars.symmetric_difference(set(varnames)))
    return dropped_vars


def inherit_docs(cls):
    """Forces a subclass to inherit method docstrings.

    Code provided by Raymond Hettinger.
    xref: https://stackoverflow.com/questions/8100166

    Parameters
    ----------
    cls : Class
        Any class definition

    Returns
    -------
    cls : Class
        Input class with docstrings inherited
    """
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls
