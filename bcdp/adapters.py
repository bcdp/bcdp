import warnings
import numpy as np
from .constants import ATTRS
from .registry import register

@register('adapter.basic')
class Adapter(object):
    """Basic Adapter to standardize dataset fields and metadata.

    This is mainly a convenience class, which is designed to ensure that all
    data loaded into the OCW ecosystem conforms to the same conventions for
    intercomparison purposes. By default, this just changes dimension names to
    conform to CF conventions if necessary.
    """
    def __call__(self, old, x=None, y=None, z=None, t=None):
        """Modify the fields of a DataArray to conform to a specific convention.

        Parameters
        ----------
        old : xarray.DataArray
            DataArray contents to be modified
        kwargs : dict, optional
            Coordinate dimension names for space(x, y, z) and time (t).
            Set these to override adapter defaults.

        Returns
        -------
        new : xarray.DataArray
            Modified DataArray
        """
        old = self.remap_coords(old, x, y, z, t)
        old = self.shift_latlons(old)
        new = self.post_process(old)
        return new

    def remap_coords(self, old, x=None, y=None, z=None, t=None):
        """Modify coordinate variable names and attributes of a DataArray.

        Parameters
        ----------
        old : xarray.DataArray
            DataArray contents to be modified

        Returns
        -------
        new : xarray.DataArray
            Modified DataArray
        """
        curvilinear = False
        coords = old.coords
        mapping = {}
        extra_labels = dict(X=x, Y=y, Z=z, T=t)

        # Adds CF Attributes to coordinates
        def add_attr(coord, axis):
            for attr, value in ATTRS[axis].items():
                ignored = ['name', 'gridname', 'labels']
                if attr == 'axis' and len(coord.dims) != 2:
                    coord.attrs[attr] = value

        for dim, coord in coords.items():
            if hasattr(coord, 'axis'):
                axis = coord.axis.upper()
            else:
                for axis, attrs in ATTRS.items():
                    if dim == extra_labels[axis] or dim in attrs['labels']:
                        break

            # Account for curvilinear grids
            if len(coord.dims) == 2 and (axis == 'X' or axis == 'Y'):
                mapping[dim] = ATTRS[axis]['gridname']
                curvilinear = True
            else:
                mapping[dim] = ATTRS[axis]['name']

            # Add attributes if necessary
            add_attr(coord, axis)
        new = old.rename(mapping)

        # Add grid coordinates for cartesian grids
        if not curvilinear:
            space_dims = ATTRS['Y']['name'], ATTRS['X']['name']
            lon, lat = new[space_dims[1]], new[space_dims[0]]
            grids = dict(zip(('X', 'Y'), np.meshgrid(lon, lat)))
            for axis in ['X', 'Y']:
                gridname = ATTRS[axis]['gridname']
                new.coords[gridname] = space_dims, grids[axis]
                add_attr(new.coords[gridname], axis)
        return new

    def shift_latlons(self, old):
        """Ensure grids have domain lat: [-90, 90] and lon: [-180, 180).

        Parameters
        ----------
        old : xarray.DataArray
            DataArray contents to be modified

        Returns
        -------
        new : xarray.DataArray
            Modified DataArray
        """
        # Ensure latitudes and longitudes are monotonically increasing.
        if 'x' in old.dims or 'y' in old.dims:
            for dim in ['x', 'y']:
                diff = np.diff(old[dim]) < 0
                if diff.all():
                    # Dimension is reversed
                    old = old.isel(**{dim: slice(None, None, -1)})
                elif diff.any():
                    warnings.warn(f'Dimension {dim} in {old.name} is not monotonic.')

        if old.lon.values.max() >= 180:
            # Convert lons from [0, 360) to [-180, 180)
            for coord in ['lon', 'x']:
                vals = old[coord].values.copy()
                dims = old[coord].dims
                vals[vals >= 180] -= 360
                old.coords.update({coord: (dims, vals)})

            # Shift grid in x-dimension so longitudes are monotonically increasing.
            # This approach might not work for all curvilinear grids, so we should
            # look into other ways to do this...
            idx = np.where(np.diff(old.lon[0, :]) < 0)[0]
            if not idx:
                return old
                
            if np.isscalar(idx):
                offset = idx + 1
            else:
                offset = idx[0] + 1
            new = old.roll(x=-offset)
        else:
            new = old
        return new

    def post_process(self, old):
        """Additional post-processing to be performed after loading the data.

        Override this method when creating your own Adapter subclass. Assume
        that coordinate variables are already remapped to conform to CF and
        bcdp conventions.

        Parameters
        ----------
        old : xarray.DataArray
            DataArray contents to be modified

        Returns
        -------
        new : xarray.DataArray
            Modified DataArray
        """
        return old
