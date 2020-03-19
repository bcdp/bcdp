import abc
import os
import inspect
import requests
import pandas as pd
import xarray as xr
from .adapters import Adapter
from .bounds import Bounds
from .constants import (RCMED_QUERY_URL, ESGF_NODES,
                        DEFAULT_INTAKE_CAT, DEFAULT_INTAKE_ESM_CAT)
from .ensemble import Ensemble
from .registry import registry, register
from .utils import inherit_docs, decode_month_units, get_dropped_varnames


class DataSource(object):
    """Loads datasets into ocw workspace.

    A data source describes where the input data comes from (eg, from your
    local filesystem or a remote server) and how to obtain it (via load()).
    All loaded datasets are represented in OCW as instances of
    xarray.DataArray.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, adapter=None):
        """Data source constructor.

        A data source describes where the input data comes from (eg, from your
        local filesystem or a remote server) and how to obtain it (via load()).
        All loaded datasets are represented in OCW as instances of
        xarray.DataArray. An optional parameter, adapter, allows for additional
        post-processing of the loaded DataArray object. The default adapter,
        an instance of bcdp.adapters.Adapter, simply changes the coordinate
        dimension names and attributes to comply with CF-format. For more
        complex post-processing use-cases, we recommend subclassing Adapter and
        passing an instance of such to this constructor.

        Parameters
        ----------
        adapter : bcdp.Adapter
            Default adapter to use for post-processing.
        """
        self.adapter = adapter if adapter else 'basic'
        self._cache = {}

    def __call__(self, *args, **kwargs):
        """Main interface for loading datasets.

        This combines the action of loading the dataset(s) from source into
        an `xarray.DataArray` object (via load()) and then building the
        evaluation Ensemble.

        Parameters
        ----------
        adapter : str, optional
            Name of adapter class to use, which must be a valid key in the
            adapters registry.
        dims : dict, optional
            Mapping of dimension labels (x, y, z and/or t) to their respective
            names in the file(s). This should rarely ever be needed to be set,
            since these can be easily inferred most of the time.
        labels : dict, optional
            Mapping of labels
        **kwargs : dict
            Keyword arguments to pass to load().

        Returns
        -------
        bcdp.Ensemble
            List of datasets to be evaluated.
        """
        dims = kwargs.pop('dims', {})
        labels = kwargs.pop('labels', {})
        datasets = self.load(*args, **kwargs)
        return Ensemble(datasets, adapter=self.adapter, dims=dims, **labels)

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        """Loads datasets from given parameters."""
        pass
        
    def _prep_datasets(self, variable, dset_dict):
        datasets = []
        for name, ds in dset_dict.items():
            variables = dict(ds.data_vars)
            if len(variables) == 1:
                # If no project or variable name information, infer it
                # from file_metadata. This will only work if the file has
                # one non-coordinate variable.
                variable = list(variables.keys())[0]
            elif not variable:
                raise ValueError('Variable name must be specified for files'
                                 ' with more than one non-coord variable.')

            # Check if variable name is defined in metadata.
            da = ds[variable].squeeze(drop=True)
            da.attrs['variable_name'] = da.name
            da.name = name
            datasets.append(da)
        return datasets


@register('source.local')
class LocalFileSource(DataSource):
    """Local Filesystem data source"""
    def load(self, paths='./*.nc', variable=None, names=None, convert_times=True,
             dims=None, project=None, load_all=False, **kwargs):
        """Loads datasets from given parameters.

        Parameters
        ----------
        paths : str or list of str, optional
            Regex or (list of) full path(s) to the netcdf file(s) to open.
        variable : str, optional
            Variable Name. If input files have only one non-coordinate variable,
            that variable's name is used by default.
        names : list of str, optional
            List of dataset names. By default these are inferred directly from
            the input `paths` attribute.
        convert_times : bool, optional
            If True (default), files are assumed to be split by time and values
            are automatically converted to pandas.Timestamp objects.
            Does nothing if `project` is not set.
        project : str, optional
            A project name that encapsulates all of the datasets to be loaded.
            This must be a valid key in bcdp.extractors.metadata_extractors,
            which includes `CMIP5`, `CORDEX`, and `obs4MIPS`. When set, this
            replaces the default behavior for defining the variable and dataset
            names. For this reason, this parameter should only be set if you are
            sure that all of the input filenames correctly conform to the
            conventions required by the given project.
        load_all : bool, optional
            If True, datasets spanned by multiple files are loaded into
            memory rather than lazily (Default False). Ignored if chunks argument
            is passed to open_mfdataset.
        **kwargs
            Keyword Arguments to `xarray.open_(mf)dataset()`

        Returns
        -------
        datasets : list
            xarray DataArray objects.
        """
        if not isinstance(paths, list):
            paths = [paths]

        # Determine dataset names
        if not names:
            if project:
                # Use project specific conventions
                extractor_cls = registry['metadata'][project]
                extractor = extractor_cls(*paths)
                names = sorted(extractor.names)
                paths = sorted(extractor.files)
            else:
                # Infer dataset names from filenames
                pre = os.path.commonprefix(paths)
                post = os.path.commonprefix([p[::-1] for p in paths])[::-1]
                names = [p.replace(pre, '').replace(post, '') for p in paths]
                root_dir = os.path.dirname(pre) + os.path.sep
                paths = [root_dir + '*{}*'.format(name) for name in names]

        # Generic dataset loader
        def open_dataset(path, **kwargs):
            dropped = get_dropped_varnames(variable) if variable else None
            if os.path.isfile(path):
                    ds = self._cache.get(
                        path,
                        xr.open_dataset(path, drop_variables=dropped, **kwargs)
                    )
                        
            else:
                chunks = kwargs.pop('chunks', None)
                ds = self._cache.get(
                    path,
                    xr.open_mfdataset(path, drop_variables=dropped, **kwargs)
                )
                ds = ds.chunk(chunks)
                if load_all and chunks is None:
                    ds = ds.load()
            self._cache[path] = ds
            return ds

        # Open each dataset
        dset_dict = {}
        for name, path in zip(names, paths):
            try:
                ds = open_dataset(path, **kwargs)
            except ValueError:
                # Custom datetime decoding required for monthly time units
                kwargs.update(decode_times=False)
                ds = decode_month_units(open_dataset(path, **kwargs))
            if project:
                # Get variable name from filename if project given.
                meta = extractor.query(filename=path)[0]
                if not variable:
                    variable = meta['variable']
                concat_dim = kwargs.pop('concat_dim', 'time')
                if concat_dim not in ds.coords:
                    dim_vals = meta[concat_dim]
                    if convert_times:
                        dim_vals = [pd.Timestamp(t) for t in dim_vals]
                    ds = ds.assign_coords({concat_dim: dim_vals})
            dset_dict[name] = ds
        return self._prep_datasets(variable, dset_dict)


@register('source.intake')
class IntakeSource(DataSource):
    """"Load remote data via the intake library."""
    def load(self, variable=None, names=None, depth=5,
             catfile=DEFAULT_INTAKE_CAT, auth_token=None):
        """Loads datasets from given parameters.

        Parameters
        ----------
        variable : str, optional
            Variable Name. If input files have only one non-coordinate variable,
            that variable's name is used by default.
        names : list of str, optional
            List of dataset names.
        depth : int, optional
            Depth of catalog search (default: 5)
        catfile : str, optional
            Path to catalogue metadata file, can be a remote URL. The pangeo
            Intake master catalogue is used by default.
        auth_token : str, optional
            Path to credentials key file to use for accessing cloud storage
            buckets.

        Returns
        -------
        datasets : list
            xarray DataArray objects.
        """
        import intake
        if auth_token:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auth_token
        cat = intake.Catalog(catfile)
        meta = cat.walk(depth=depth)
        sel = [name for name, ent in meta.items() if ent.container == 'xarray']
        names = sel if not names else names
        entries = [cat[name] for name in sel]
        shortnames = [name.split('.')[-1] for name in sel]
        dset_dict = {name: ent.to_dask() for name, ent in zip(shortnames, entries)
                     if name in names}
        return self._prep_datasets(variable, dset_dict)


@register('source.intake-esm')
class IntakeESMSource(DataSource):
    """"Load remote data via the intake-esm library."""
    def load(self, query, catfile=DEFAULT_INTAKE_ESM_CAT, **kwargs):
        """Loads datasets from given parameters.
        
        Parameters
        ----------
        query: dict
            Key, value pairs used to search the catalogue.
            Depth of catalog search (default: 5)
        catfile : str, optional
            Path to catalogue metadata file, can be a remote URL. The pangeo
            intake-esm CMIP6 catalogue is used by default.
        **kwargs : dict, optional
            Keyword Arguments for `intake_esm.core.esm_datastore.to_dataset_dict()`

        Returns
        -------
        datasets : list
            xarray DataArray objects.
        """
        import intake
        col = intake.open_esm_datastore(catfile)
        cat = col.search(**query)
        dset_dict = cat.to_dataset_dict(**kwargs)
        variable = kwargs.get('variable_id')
        return self._prep_datasets(variable, dset_dict)
        

@register('source.rcmed')
class RCMEDSource(DataSource):
    """JPL Regional Climate Model Evaluation Database (RCMED) Data Source"""
    def load(self, dataset_id, parameter_id, domain, chunks=None):
        """Loads datasets from given parameters.

        Parameters
        ----------
        dataset_id : int
            Dataset ID in RCMED.
        variable_id : int
            Variable (Parameter) ID in RCMED.
        domain : bcdp.bounds.Bounds
            Bounds defining the spatial and temporal domain to search
            (and subset) the requested data.
        chunks : dict, optional
            If set, load data into a dask array with given chunk sizes.
        Returns
        -------
        list of xarray.DataArray
            xarray DataArray object with loaded data.
        """
        # Read RCMED metadata table from RCMES website
        metadata = self.get_metadata()
        idx = metadata.parameter_id == parameter_id
        if not idx.any():
            raise ValueError(f'Invalid Parameter ID: {parameter_id}. '
                              'Should be one of: {metadata.parameter_id}')

        # Format remaining parameters for request query
        params = dict(datasetId=dataset_id, parameterId=variable_id,
                      **domain.to_dict())

        # Make request to RCMED server
        r = requests.get(RCMED_QUERY_URL, params=params)
        r.raise_for_status()
        result = r.text.split('\r\n')
        meta = result[:12]
        data = result[12:-1]


        # Create DataArray object from output
        variable_name = meta[2].split('Parameter:')[1].split('(')[0].strip()
        name =  metadata.database[idx].values[0]
        df = pd.DataFrame(data=data)
        df = df[0].str.split(',', expand=True)
        numeric_dims = ['lat', 'lon', 'lev', variable_name]
        df.columns = ['lat', 'lon', 'lev', 'time', variable_name]
        df[numeric_dims] = df[numeric_dims].astype(float)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index(['time', 'lev', 'lat', 'lon'])
        da = df[variable_name].to_xarray().squeeze()
        if chunks:
            da = da.chunk(chunks)
        da.attrs['units'] = metadata['units'][idx].values[0]
        da = da.where(da != metadata['missingdataflag'][idx].values[0])
        da.name = name
        return [da]
        
    def get_metadata(self):
        """Loads RCMED metadata.
        
        Returns
        -------
        meta : pandas.DataFrame
            RCMED metadata table.
        """
        info = requests.get(RCMED_QUERY_URL, params=dict(param_info='yes')).json()
        meta = pd.DataFrame(data=info['data'], columns=info['fields_name'])
        return meta


@register('source.esgf')
class ESGFSource(DataSource):
    """Earth System Grid (ESGF) data source"""
    _origin = 'esgf'
    def load(self, variable=None, project=None, node='JPL', **kwargs):
        """Builds an xarray.DataArray object from given parameters.

        Parameters
        ----------
        paths : str or list of str, optional
            Regex or (list of) full path(s) to the netcdf file(s) to open.
        variable : str, optional
            Variable Name.
        project : str, optional
            A project name that encapsulates all of the datasets to be loaded.
            This must be a valid key in the bcdp object registry, which includes
            `CMIP5`, `CORDEX`, and `obs4MIPS`. When set, this replaces the
            default behavior for defining the variable and dataset names. For
            this reason, this parameter should only be set if you are sure that
            all of the input filenames correctly conform to the conventions
            required by the given project.
        node : str, optional
            ESGF node to search from, default is JPL.
        **kwargs : dict, optional
            ESGF search parameters.
        Returns
        -------
        datasets : dict of xarray.DataArray
            xarray DataArray objects with origin information and CF-compliant
            coordinate variables, keyed by dataset names.
        """
        hostname = ESGF_NODES[node]
        raise NotImplementedError('')


load_local = LocalFileSource()
load_intake = IntakeSource()
load_intake_esm = IntakeESMSource()
load_rcmed = RCMEDSource()
