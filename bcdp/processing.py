import numpy as np
import pandas as pd
import xarray as xr
from . import utils
from .bounds import PolygonBounds

def subset(da, domain):
    """Subset dataset to specified domain.

    Parameters
    ----------
    da : xarray.DataArray
        Dataset
    domain : bcdp.Bounds
        Boundary to subset on.

    Returns
    -------
    xarray.DataArray
        Subsetted dataset.
    """
    # Use simpler 1D subsettting if bounds is just a bounding box.
    domain = self.overlap if domain is None else domain
    if isinstance(domain, PolygonBounds):
        pts = np.c_[da.lon.values.ravel(), da.lat.values.ravel()]
        mask = domain.contains(pts).reshape(da.lon.shape)
        md = xr.DataArray(mask, dims=('y', 'x'))
        da = da.where(md).isel(x=md.any('y'), y=md.any('x'))
    else:
        da = (da.pipe(utils.subset_1d, 'y', domain.lat_bnds)
                .pipe(utils.subset_1d, 'x', domain.lon_bnds))

    if domain.time_bnds._bnds:
        da = utils.subset_1d(da, 'time', domain.time_bnds)
    return da


def normalize_times(da, assume_gregorian=False):
    """Normalize times in dataset.
    If frequency is monthly, set day of month to 1. If daily, set hour to 0Z.

    Parameters
    ----------
    da : xarray.DataArray
        Dataset
    assume_gregorian : bool, optional
        If True, express datetimes on nonstandard calendars to gregorian.
        
    Returns
    -------
    xarray.DataArray
        Normalized dataset.
    """
    da = da.copy()
    idx = da.indexes['time']
    times = idx.to_series()
    if assume_gregorian and not isinstance(idx, pd.DatetimeIndex):
        times = times.apply(lambda d: pd.Timestamp(str(d)))
        da = da.assign_coords(time=times)
    freq = utils.infer_freq(da)
    if freq == 'M' or freq == 'MS':
        da['time'] = times.apply(lambda d: d.replace(day=1))
    elif freq == 'D':
        da['time'] = times.apply(lambda d: d.replace(hour=0))
    return da


def resample(da, freq):
    """Resample datasets to a standard frequency.

    Parameters
    ----------
    da : xarray.DataArray
        Dataset
    freq : str, optional
        Pandas frequency string.

    Returns
    -------
    xarray.DataArray
        Subsetted dataset.
    """
    if freq != utils.infer_freq(da):
        attrs = da.attrs
        da = da.resample(time=freq).mean('time')
        da.attrs.update(attrs)
    return da


def select_season(da, season=None):
    """Subset dataset to only selected season.

    Parameters
    ----------
    da : xarray.DataArray
        Dataset
    season : str or tuple, optional
        Season. Can be a string (eg 'NDJFM'), or a tuple
        (start_month, end_month)

    Returns
    -------
    xarray.DataArray
        Seasonalized dataset.
    """
    if season:
        freq = utils.infer_freq(da)
        ys, ye = da.time.dt.year.values.min(), da.time.dt.year.values.max()
        if isinstance(season, str):
            ms, me = utils.season_to_range(season)
        else:
            # Subset data to include only selected season
            ms, me = season
        mask1 = da.time.dt.month >= ms
        mask2 = da.time.dt.month <= me
        if ms > me:
            cond1 = mask1|mask2
        else:
            cond1 = mask1&mask2

        # Additionally remove years which do not contain all months
        # in the season
        locut = f'{ys}-{ms}'
        hicut = f'{ye}-{me}'
        str_times = da.time.astype(str)
        cond2 = (str_times >= locut) & (str_times <= hicut)
        da = da.isel(time=cond1&cond2)
    return da
