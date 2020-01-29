import functools
import numpy as np
import pandas as pd
import xarray as xr
from collections import OrderedDict
from . import utils, processing
from .registry import registry
from .bounds import Bounds


class Ensemble(object):
    """An "ensemble" of datasets.
    An ensemble, loosely speaking, is a collection of datasets, which may or
    may not be homogeneous. Here, we define a "homogeneous" ensemble as one
    which has datasets with the the same spatial and temporal
    domains and resolutions such that direct intercomparisons are possible.
    At minmimum, this requires subsetting and regridding to a standard grid.

    An optional parameter, adapter, allows for additional
    pre-processing of the loaded DataArray objects. The default adapter,
    an instance of bcdp.Adapter, simply changes the coordinate
    dimension names and attributes to standard names (ie lat, lon, and time).
    For more complex post-processing use-cases, we recommend subclassing Adapter
    and adding it to the registry.
    """
    def __init__(self, data, adapter=None, dims=None):
        """Ensemble constructor.

        Parameters
        ----------
        data : list or pandas.Series
            Data structure consisting of DataArrays denoting each input dataset.
        adapter : str, optional
            Name of adapter class to use, which must be a valid key in the
            adapters registry.
        dims : dict, optional
            Mapping of dimension labels (x, y, z and/or t) to their respective
            names in the file(s). This should rarely ever be needed to be set,
            since these can be easily inferred most of the time.
        """
        self.assign(data, inplace=True)

        # Conform datasets to ocw standards via selected adapter
        if adapter:
            adapter_cls = registry['adapter'][adapter]
            adapt = adapter_cls()
            self.apply(adapt, inplace=True, **dims)

    def __getitem__(self, name):
        return self.data[name]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def concat(*objs):
        """Combine Ensemble objects into one.

        *objs : list of Ensemble
            Objects to combine.

        Returns
        -------
        bcdp.Ensemble
            Combined object.
        """
        return Ensemble(pd.concat([obj.data for obj in objs]))

    def assign(self, data, inplace=False):
        """Construct a new Ensemble or modify this instance in place.

        Parameters
        ----------
        data : list, pandas.Series, or xarray.Dataset
            New data.
        inplace : bool, optional
            If True, modify data in place.
        """
        if not inplace:
            return Ensemble(data)
        if isinstance(data, xr.Dataset):
            names = data.name.values
            data = [data.isel(name) for name in self._names]
            for d, name in zip(data, names):
                d.name = name
            self.data = utils.build_series(data)
            self._names = names
        elif isinstance(data, pd.Series):
            self._names = [dataset.name for dataset in data.values]
            self.data = data
        else:
            self.data = utils.build_series(data)
            self._names = list(self.data.index)
        return self

    def apply(self, func, *args, **kwargs):
        """Apply a function on each dataset.

        Parameters
        ----------
        func : callable
            Function in which the first argument and return value
            is an xarray.DataArray.
        *args : list
            Positional arguments to pass to func.
        **kwargs : dict
            Keyword arguments to pass to func. Set inplace=True to modify this
            instance in place rather than create a new one.

        Returns
        -------
        bcdp.Ensemble
            Modified Ensemble.
        """
        inplace = kwargs.pop('inplace', False)
        data = self.data.apply(func, args=args, **kwargs)
        names = data.index
        for da, name in zip(data, names):
            da.name = name
        return self.assign(data, inplace=inplace)

    def add_mean(self, name, label='ENS'):
        """Calculate simple arithmetic mean of ensemble.

        Appends an extra xarray.DataArray to a named bundle with the ensemble mean.
        Means are calculated along the names dimension, so the name must
        represent a bundle of datasets with identical dimensions.

        Parameters
        ----------
        name : str
            Name of dataset bundle.
        label : str, optional
            Name to use for labeling the ensemble mean xarray.DataArray.

        Returns
        -------
        bcdp.Ensemble
            Modified Ensemble.
        """
        bundle = self[name].copy()
        if label in bundle.names.values:
            raise ValueError('Ensemble mean already calculated!')
        mean = bundle.mean('names').expand_dims('names')
        mean['names'] = [label]
        new = xr.concat([bundle, mean], dim='names')
        data = self.data.copy()
        data[name] = new
        return self.assign(data)

    def subset(self, domain=None):
        """Subset dataset to specified domain.

        Parameters
        ----------
        domain : bcdp.Bounds, optional
            Boundary to subset on. Default: Subset to common overlap of all
            datasets.

        Returns
        -------
        bcdp.Ensemble
            Subsetted dataset.
        """
        # Use simpler 1D subsettting if bounds is just a bounding box.
        domain = self.overlap if domain is None else domain
        return self.apply(processing.subset, domain)

    def normalize_times(self, assume_gregorian=False):
        """Normalize times in dataset.
        If frequency is monthly, set day of month to 1. If daily, set hour to 0Z.

        Parameters
        ----------
        assume_gregorian : bool, optional
            If True, express datetimes on nonstandard calendars to gregorian.

        Returns
        -------
        bcdp.Ensemble
            Normalized dataset.
        """
        return self.apply(processing.normalize_times,
                          assume_gregorian=assume_gregorian)

    def resample(self, freq=None, how='mean', reference=None):
        """Resample datasets to a standard frequency.

        Parameters
        ----------
        freq : str, optional
            Pandas frequency string.

        Returns
        -------
        bcdp.Ensemble
            Subsetted dataset.
        """
        if freq is None:
            if reference:
                freq = utils.infer_freq(self[reference])
            else:
                freq = utils.infer_freq(self.first)
        return self.apply(processing.resample, freq)

    def select_season(self, season=None):
        """Subset dataset to only selected season.

        Parameters
        ----------
        season : str or tuple, optional
            Season. Can be 'DJF', 'MAM', 'JJA', 'SON', or a tuple
            (start_month, end_month)

        Returns
        -------
        bcdp.Ensemble
            Seasonalized dataset.
        """
        return self.apply(processing.select_season, season=season)

    def regrid(self, output_grid=None, backend=None,
               method=None, clean=True, inplace=False, **kwargs):
        """Regrid all datasets to a single output grid.

        All regridding operations can be broken down into two steps: (1) setup,
        where regridding parameters are calculated based on the dimensions of
        the input and output grids (ie, weights), and (2) formally regridding
        each dataset. To save time, all datasets with the same grid dimensions
        are grouped together, since this means the more expensive setup step
        is performed just once per group.

        Parameters
        ----------
        output_grid : str or xarray.Dataset, optional
            xarray Dataset object containing grid dimensions. Must include
            lat and lon variables, which can be either 1D (for cartesian grids)
            or 2D (for cartesian or curvilinear grids). If a str is provided
            instead, use the grid from the dataset labeled by this value.
            If no name is given, then the reference grid is derived from the
            first ordinal dataset.
        backend : str, optional
            Name of the regridding backend to use. The following are included
            with bcdp: 'scipy', 'pyresample' and 'esmf'. Additional backends can be
            supported by registering a subclass of bcdp.regridder.Regridder.
        method : str, optional
            Regridding method to use (eg, bilinear or nearest neighbor).
            Supported values for this string depend on the backend, for quick
            reference (default listed first):
            - scipy: 'linear', 'nearest'
            - pyresample: 'nearest'
            - esmf: 'bilinear', 'conservative', 'patch', 'nearest_s2d', 'nearest_d2s',
        clean : bool, optional
            Remove any extra files generated by the regridder (Default: True).
        **kwargs : dict, optional
            Additional parameters to pass based on choice of backend and method.

        Returns
        -------
        bcdp.Ensemble
            Modified Ensemble.
        """
        # Regrid on reference by default
        if output_grid is None:
            output_grid = utils.grid_from_data(self.first)

        # Can also regrid to labeled dataset grid.
        elif isinstance(output_grid, str):
            output_grid = utils.grid_from_data(self[output_grid])

        # Group together all datasets by common grid dimensions. This way
        # the most expensive aspect of the regridding calculation is done just
        # the minimal number of times.
        keys = self.data.apply(lambda da: da.shape[1:])
        groups = self.data.groupby(keys)

        # Select regridding backend / library to use
        backend = backend if backend else 'scipy'
        regridder_cls = registry['regridder'][backend]

        # Applies regridding to each group
        def regrid_each(group):
            input_grid = utils.grid_from_data(group.values[0])
            regridder = regridder_cls(input_grid, output_grid, method, **kwargs)
            out = group.apply(regridder)
            if clean:
                regridder.clean()
            return out

        data = groups.apply(regrid_each)
        return self.assign(data, inplace=inplace)

    def homogenize(self, assume_gregorian=False, freq=None, domain=None, season=None,
                   output_grid=None, backend=None, method=None, clean=True,
                   **kwargs):
        """Homogenize all datasets in this ensemble.

        This operation essentially does all the following in one go:
        - Datetime Normalization
        - Spatial Subset
        - Temporal (Seasonal) Subset
        - Resampling
        - Regridding

        Parameters
        ----------
        assume_gregorian : bool, optional
            If True, express datetimes on nonstandard calendars to gregorian.
        freq : str, optional
            Pandas frequency string.
        domain : bcdp.Bounds, optional
            Boundary to subset on.
        season : str or tuple, optional
            Season. Can be 'DJF', 'MAM', 'JJA', 'SON', or a tuple
            (start_month, end_month)
        output_grid : xarray.Dataset, optional
            xarray Dataset object containing grid dimensions. Must include
            lat and lon variables, which can be either 1D (for cartesian grids)
            or 2D (for cartesian or curvilinear grids). By default this
            is inferred from the reference dataset.
        backend : str, optional
            Name of the regridding backend to use. The following are included
            with bcdp: 'scipy', 'pyresample' and 'esmf'. Additional backends can be
            supported by registering a subclass of bcdp.regridder.Regridder.
        method : str, optional
            Regridding method to use (eg, bilinear or nearest neighbor).
            Supported values for this string depend on the backend, for quick
            reference (default listed first):
            - scipy: 'linear', 'nearest'
            - pyresample: 'nearest'
            - esmf: 'bilinear', 'conservative', 'patch', 'nearest_s2d', 'nearest_d2s',
        clean : bool, optional
            Remove any extra files generated by the regridder (Default: True).
        **kwargs : dict, optional
            Additional parameters to pass based on choice of backend and method.

        Returns
        -------
        bcdp.Ensemble
            Modified Ensemble.
        """
        ens = (self.normalize_times(assume_gregorian=assume_gregorian)
                   .subset(domain=domain)
                   .select_season(season=season)
                   .resample(freq=freq)
                   .regrid(output_grid=output_grid, backend=backend, method=method,
                           clean=clean, **kwargs))
        return ens

    def bundle(self, name=None, **names):
        """Bundle (combine) multiple dataset objects inside the ensemble into one.

        Parameters
        ----------
        name : str, optional
            If provided, bundle all datasets into this ensemble into
        names : dict, optional
            Mapping of new name to list of old names in this ensemble.

        Returns
        -------
        bcdp.Ensemble
            Modified Ensemble.
        """
        if name:
            names = {name: self.names}
        datasets = []
        excluded_labels = []
        for name, labels in names.items():
            data = self[labels].values
            combined = (xr.concat(data, dim='names', coords='minimal', compat='override')
                          .assign_coords(names=labels))
            combined.name = name
            datasets.append(combined)
            excluded_labels.extend(labels)
        unbundled_names = set(self.names).difference(excluded_labels)
        if unbundled_names:
            datasets.append(self[unbundled_names])
        return Ensemble(datasets)

    def unbundle(self, *names):
        """Unbundle (combine) multiple dataset objects inside the ensemble into one.

        Parameters
        ----------
        *names : str, optional
            Dataset names to unbundle.

        Returns
        -------
        bcdp.Ensemble
            Modified Ensemble.
        """
        datasets = []
        for name in names:
            data = self[name]
            for label in list(data.names.values):
                d = data.sel(names=label).drop('names')
                d.name = str(label)
                datasets.append(d)
        unbundled_names = set(self.names).difference(names)
        if unbundled_names:
            datasets.append(self[unbundled_names])
        return Ensemble(datasets)

    def apply_metric(self, metric, *labels, along_labels=False, args=(), **kwargs):
        """Apply metric to datasets contained in Ensemble.

        A metric is a function whose signature is of the form:
        def <metric>(data, *other_data, *args, **kwargs):
            ...

        Where *other_data are positional arguments that are also datasets
        contained in the ensemble. These datasets must be labeled. Example:
        def bias(data, reference):
            return data - reference
        ens = ens.apply_metric(bias, 'reference')

        Parameters
        ----------
        metric : callable
            Metric function to apply, whose first positional argument (and all
            positoinal arguments thereafter denoted by *labels) is an xarray
            dataarray.
        *labels : list of str
            Labels used to pass additional datasets as positional arguments to
            the metric function. For example, for binary metrics such as a bias
            which is calculated relative to another dataset called 'reference',
            the function called would be bias(data, ens['reference']).
        along_labels : bool, optional
            If True, metric is applied along the labels. For example, this would
            allow a bias metric to be applied to the reference dataset. As
            evaluation metrics with multiple dataset arguments generally aren't
            meaningful in these situations, it is set to False to default, which
            results in all labeled datasets being unmodified.
        args : tuple, optional
            Additional positional arguments to pass to the metric function that
            are not datasets labeled by this instance.
        **kwargs : dict, optional
            Additonal keyword arguments to pass to the metric function.

        Returns
        -------
        bcdp.Ensemble
            Ensemble with metric applied.
        """
        def wrapped_metric(data, *args, **kwargs):
            if not along_labels and data.name in labels:
                return data
            return metric(data, *args, **kwargs)
        return self.apply(wrapped_metric, *args, **kwargs)

    def apply_on_labels(self, func, *labels, args=None, **kwargs):
        """Apply function to specific labeled datasets contained in Ensemble.

        Parameters
        ----------
        func : callable
            Function to apply, whose first positional argument (and all
            positoinal arguments thereafter denoted by *labels) is an xarray
            dataarray.
        *labels : list of str
            Labels specifying which datasets in the ensemble will be modified
            by the applied function.
        args : tuple, optional
            Additional positional arguments to pass to the metric function that
            are not datasets labeled by this instance.
        **kwargs : dict, optional
            Additonal keyword arguments to pass to the metric function.

        Returns
        -------
        bcdp.Ensemble
            Ensemble with metric applied.
        """
        args = () if not args else args
        def wrapped_func(data, *args, **kwargs):
            if data.name not in labels:
                return data
            return func(data, *args, **kwargs)
        return self.apply(wrapped_func, *args, **kwargs)

    def plot(self, label_func=None, func=None, label_args=None,
             args=None, label_kwargs=None, **kwargs):
        """Apply plotting functions to datasets contained in Ensemble.

        Parameters
        ----------
        func : callable
            Plotting function to apply to each dataset in the ensemble.
        *labels : list of str
            Labels used to pass additional datasets as positional arguments to
            the metric function. For example, for binary metrics such as a bias
            which is calculated relative to another dataset called 'reference',
            the function called would be bias(data, ens['reference']).
        along_labels : bool, optional
            If True, metric is applied along the labels. For example, this would
            allow a bias metric to be applied to the reference dataset. As
            evaluation metrics with multiple dataset arguments generally aren't
            meaningful in these situations, it is set to False to default, which
            results in all labeled datasets being unmodified.
        args : tuple, optional
            Additional positional arguments to pass to the metric function that
            are not datasets labeled by this instance.
        **kwargs : dict, optional
            Additonal keyword arguments to pass to the metric function.

        Returns
        -------
        bcdp.Ensemble
            Ensemble with metric applied.
        """
        pass

    @property
    def homogeneous(self):
        """True if all datasets have equal dimensions (space and time)."""
        result = True
        for da in self.data.values[1:]:
            if da.shape != self.first.shape:
                return False
            all_lats = (da.x.values == self.first.x.values).all()
            all_lons = (da.y.values == self.first.y.values).all()
            all_times = (da.time.values == self.first.time.values).all()
            result &= all_lats & all_lons & all_times
            if not result:
                return False
        return True

    @property
    def names(self):
        return self._names

    @property
    def first(self):
        return self[0]

    @property
    def overlap(self):
        """Spatial and temporal overlap of all the datasets"""
        ref = self.first
        time_min, time_max = ref.time.values.min(), ref.time.values.max()
        lon_min, lon_max = ref.lon.values.min(), ref.lon.values.max()
        lat_min, lat_max = ref.lat.values.min(), ref.lat.values.max()
        if not isinstance(self.data, xr.Dataset):
            for da in self.data.values:
                time_min, time_max = (max(da.time.values.min(), time_min),
                                      min(da.time.values.max(), time_max))
                lon_min, lon_max = (max(da.lon.values.min(), lon_min),
                                    min(da.lon.values.max(), lon_max))
                lat_min, lat_max = (max(da.lat.values.min(), lat_min),
                                    min(da.lat.values.max(), lat_max))
        return Bounds(lon_bnds=(lon_min, lon_max), lat_bnds=(lat_min, lat_max),
                      time_bnds=(time_min, time_max))
