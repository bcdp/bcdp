import abc
import numpy as np
import xarray as xr
from .registry import register

class Regridder(object):
    """Generic regridder interface."""
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_grid, output_grid, method=None, **kwargs):
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.method = method
        self._params = kwargs

    @property
    def input_grid(self):
        return self._input_grid

    @input_grid.setter
    def input_grid(self, value):
        self.cached = False
        self._input_grid = value

    @property
    def output_grid(self):
        return self._output_grid

    @output_grid.setter
    def output_grid(self, value):
        self.cached = False
        self._output_grid = value

    def setup(self):
        pass

    def regrid(self, data):
        pass

    def clean(self):
        pass

    def __call__(self, data):
        if ((self.output_grid.lat.shape == self.input_grid.lat.shape) and
            (self.output_grid.lon.shape == self.input_grid.lon.shape) and
            (self.output_grid.lat.values == self.input_grid.lat.values).all() and
            (self.output_grid.lon.values == self.input_grid.lon.values).all()):
                return data
        if not self.cached:
            self.setup()
            self.cached = True
        return self.regrid(data)


@register('regridder.esmf')
class XesmfRegridder(Regridder):
    """xESMF regridder"""
    def setup(self):
        import xesmf as xe
        self._regridder = xe.Regridder(self.input_grid, self.output_grid,
                                       self.method, **self._params)

    def regrid(self, data):
        attrs = data.attrs
        result = self._regridder(data)
        result.attrs.update(attrs)
        return result

    def clean(self):
        self._regridder.clean_weight_file()


@register('regridder.scipy')
class ScipyRegridder(Regridder):
    """Scipy regridder"""
    def regrid(self, data):
        return data.interp(x=self.output_grid.x, y=self.output_grid.y,
                           method=self.method)


@register('regridder.pyresample')
class PyresampleRegridder(Regridder):
    """Pyresample regridder"""
    def setup(self):
        from pyresample import geometry, kd_tree, bilinear
        input_def = geometry.SwathDefinition(lons=self.input_grid.lon.values,
                                             lats=self.input_grid.lat.values)
        output_def = geometry.SwathDefinition(lons=self.output_grid.lon.values,
                                              lats=self.output_grid.lat.values)
        if not self.method or self.method == 'nearest':
            # Set default neighbours used in stencil to 1. Normal default is
            # 8, which won't work if the input and output grids are similar in
            # size and resolution.
            self._params.setdefault('neighbours', 1)
            self._args = kd_tree.get_neighbour_info(input_def, output_def,
                                                    50000, **self._params)
            self._regridder = kd_tree.get_sample_from_neighbour_info
        else:
            raise NotImplementedError('Only nearest-neighbor regridding is '
                                      'currently supported for pyresample backend')

    def regrid(self, data):
        nt = len(data.time)
        shp = [nt] + list(self.output_grid.lon.shape)
        out = xr.DataArray(np.zeros(shp), dims=data.dims,
                           coords={'lon': self.output_grid.lon,
                                   'lat': self.output_grid.lat},
                           name=data.name, attrs=data.attrs)
        for i in range(nt):
            tstep = data.isel(time=i)
            if tstep.isnull().any():
                tstep = tstep.to_masked_array()
            else:
                tstep = tstep.values
            out[i] = self._regridder('nn', shp[1:], tstep, *self._args)
        return out
