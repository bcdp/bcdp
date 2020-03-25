import json
import cartopy.io.shapereader as shapereader
import matplotlib.path as mpath
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import unary_union
from .registry import register
from .utils import inherit_docs

class Domain(object):
    """One dimensional domain.

    Used to represent the domain (start and end values) for a given dimension.
    Internally, this is implemented as a tuple with enhanced features such
    as bounds checking and special handling for periodic dimensions.
    """
    def __init__(self, name, bnds, periodic=False, valid_range=None):
        """Domain constructor.

        Parameters
        ----------
        name : str
            Corresponding dimension name.
        bnds : tuple
            "Min" (start) and "max" (end) values for domain.
        periodic : bool
            Whether or not the dimension loops around.
        valid_range : tuple
            Domain start and end values
        """
        self._name = name
        self._periodic = periodic
        self._range = valid_range
        self.bnds = bnds

    def __repr__(self):
        return self.bnds.__repr__()

    @property
    def name(self):
        """Corresponding dimension name"""
        return self._name

    @property
    def periodic(self):
        """Whether domain loops around"""
        return self._periodic

    @property
    def range(self):
        """Minimum and maximum possible values for which domain is valid"""
        return self._range

    @property
    def bnds(self):
        """Domain start and end values"""
        return self._bnds

    @bnds.setter
    def bnds(self, value):
        # This section sets the bounds to the valid range by default.
        if not value:
            if self.range:
                value = self.range
            else:
                self._bnds = value
                return

        # Handle max > min
        new_min, new_max = value
        if new_max <= new_min:
            if not self.periodic:
                raise ValueError('Max value of non-periodic dim must exceed min')
            elif self.range:
                self._slices = [(new_min, self.range[1]),
                                (self.range[0], new_max)]
        else:
            self._slices = [value]

        # Check if input is within valid range
        if self.range:
            global_min, global_max = self.range
            if new_min < global_min or new_max > global_max:
                raise ValueError('Domain {} outside valid range: {}'
                                 .format(value, self.range))
        self._bnds = value

    @property
    def min(self):
        return self.bnds[0]

    @min.setter
    def min(self, value):
        self.bnds = (value, self.max)

    @property
    def max(self):
        return self.bnds[1]

    @max.setter
    def max(self, value):
        self.bnds = (self.min, value)

    def __getitem__(self, idx):
        return self.bnds[idx]

    def __setitem__(self, idx, value):
        if idx == 1:
            self.max = value
        else:
            self.min = value

    def to_dict(self):
        """Dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of this instance.
        """
        if not self.range:
            postfix = 'Start', 'End'
        else:
            postfix = 'Min', 'Max'
        kmin, kmax = self.name + postfix[0], self.name + postfix[1]
        return {kmin: self.min, kmax: self.max}


@register('bounds.bbox')
class Bounds(object):
    """A bbox representation of dimensional boundaries.

    This is the simplest boundary type, which requires only the mins and maxes
    of the lat/lon/time fields to be defined.
    """
    def __init__(self, lon_bnds=None, lat_bnds=None, time_bnds=None):
        """Bounds constructor.

        Parameters
        ----------
        lon_bnds, lat_bnds, time_bnds : tuple of float, optional
            Minimum and maximum bounding values for lon, lat, and
            time dimensions.
        """
        self._initialize_domains(lon_bnds=lon_bnds, lat_bnds=lat_bnds,
                                 time_bnds=time_bnds)

    def _initialize_domains(self, lon_bnds=None, lat_bnds=None, time_bnds=None):
        if lon_bnds is not None:
            self._lon_bnds = Domain('lon', lon_bnds,
                                    periodic=True, valid_range=(-180, 180))
        if lat_bnds is not None:
            self._lat_bnds = Domain('lat', lat_bnds, valid_range=(-90, 90))
        if time_bnds is not None:
            time_bnds = list(time_bnds)
            for i, time in enumerate(time_bnds):
                if isinstance(time, str):
                    time_bnds[i] = np.datetime64(time)
            self._time_bnds = Domain('time', tuple(time_bnds))

    @property
    def lon_bnds(self):
        return self._lon_bnds

    @lon_bnds.setter
    def lon_bnds(self, value):
        self._lon_bnds.bnds = value

    @property
    def lat_bnds(self):
        return self._lat_bnds

    @lat_bnds.setter
    def lat_bnds(self, value):
        self._lat_bnds.bnds = value

    @property
    def time_bnds(self):
        return self._time_bnds

    @time_bnds.setter
    def time_bnds(self, value):
        self._time_bnds.bnds = value

    def to_dict(self):
        """Dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of this instance.
        """
        obj = {}
        for bnds in [self.lon_bnds, self.lat_bnds, self.time_bnds]:
            obj.update(bnds.to_dict())
        return obj

    def contains(self, points):
        """Check if a set of points are inbounds.

        Parameters
        ----------
        points : array_like
            Array of point coordinates (x, y)

        Returns
        -------
        mask : array_like
            Mask defining which points are in the boundary.
        """
        mask = False
        for slc in self.lon_bnds._slices:
            ymin, ymax = self.lat_bnds
            xmin, xmax = slc
            x, y = points.T
            mask |= (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        return mask


@register('bounds.polygon')
@inherit_docs
class PolygonBounds(Bounds):
    """Boundary defined by polygon vertices.

    Particularly useful if your boundary contains a collection of states,
    provinces, or countries. Boundary data can be loaded directly from
    shapefile or GeoJSON formats.
    """
    def __init__(self, boundary_file, names=None, field='NAME',
                 fmt=None, time_bnds=None):
        """PolygonBounds constructor.

        Parameters
        ----------
        boundary_file : str
            Path to a boundary file (GeoJSON or shapefile).
        names : list of str, optional
            List of elements (eg, countries, states, provinces, or counties)
            to load from the boundary file. If None, all features are used
            to define the overall boundary mask.
        field : str, optional
            Name of field to load from boundary file, default: NAME
        fmt : {None, 'shp', 'json'}
            Boundary file type. Can be shp (shapefile), json (GeoJSON), or
            inferred from the extension.
        time_bnds : tuple of floats, optional
            Start and end times.
        """
        self._initialize_domains(lat_bnds=None, lon_bnds=None, time_bnds=time_bnds)
        fmt = boundary_file.split('.')[-1] if not fmt else fmt
        def geom_sel(name, names):
            if not names:
                return True
            else:
                return name in names
        if fmt == 'shp':
            reader = shapereader.Reader(boundary_file)
            features = reader.records()
            shapes = [feature.geometry for feature in features
                      if geom_sel(feature.attributes[field], names)]
        elif fmt == 'json':
            with open(boundary_file) as json_file:
                features = json.load(json_file)['features']
            shapes = [shape(feature['geometry']) for feature in features
                      if geom_sel(feature['properties'][field], names)]
        else:
            raise ValueError('Unrecognized boundary file format.')
        self.shapes = unary_union(shapes)
        if self.shapes.boundary.type == 'LineString':
            self.shapes = MultiPolygon([self.shapes])

    @property
    def shapes(self):
        """Polygon geometries"""
        return self._shapes

    @shapes.setter
    def shapes(self, value):
        self._set_bnds_from_shape(value)
        self._shapes = value

    def _set_bnds_from_shape(self, shp):
        xmin, ymin, xmax, ymax = shp.bounds
        self.lon_bnds = (xmin, xmax)
        self.lat_bnds = (ymin, ymax)

    @property
    def xy(self):
        """Polygon vertices"""
        xy = []
        for shp in self.shapes:
            xy.append(np.asarray(shp.boundary.coords.xy).T)
        return xy

    def contains(self, points):
        # Optimization: matplotlib's contain_points() is relatively slow,
        # so only use it to check points that are within the polyon's bbox.
        # Especially useful when the input dataset domain is much larger
        # than the boundary domain.
        mask = False
        for shp in self.shapes:
            self._set_bnds_from_shape(shp)
            polymask = super().contains(points)
            if shp.boundary.type != 'MultiLineString':
                boundaries = [shp.boundary]
            else:
                boundaries = shp.boundary
            for boundary in boundaries:
                path = mpath.Path(np.asarray(boundary.coords.xy).T)
                polymask[polymask] &= path.contains_points(points[polymask])
                mask |= polymask

        # Reset this instance's spatial attributes.
        self._set_bnds_from_shape(self.shapes)
        return mask
