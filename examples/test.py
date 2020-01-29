import matplotlib
matplotlib.use('Agg')
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import bcdp

# Create a file conventions template. Here we only need the model and variable
# names so the rest of the file template is filled in by wildcards.
project = 'CORDEX-Africa'
template = '*_{model}_*_{variable}.nc'
bcdp.build_extractor(project, template, name_field='model', index=[1, 6])

# Load the data. Because we have loaded the template, the loader now knows
# exactly how to extract the required informations from the filenames.
loader = bcdp.LocalFileSource()
paths = os.path.join(os.path.expanduser('~'), 'data/CORDEX_Africa/*clt*')
ens = loader(paths=paths, project=project)

# The loader returns an Ensemble object, which is essentially a collection
# of datasets and applies preprocessing operations to each of them. Here we
# will regrid the data to a coarser (0.88 degree) grid using ESMF's bilinear
# interpolation, and consider only the winter months (DJF).
output_grid = bcdp.utils.grid_from_res((0.88, 0.88), ens.overlap)
ens = ens.homogenize(freq='Y', season='DJF', backend='scipy', method='linear',
                     output_grid=output_grid, clean=False, reuse_weights=True)

# Now that the underlying datastructures are homogeneous (same grid and
# time step), we can convert it to an xarray dataarray which has dimensions
# (names, time, lat, lon). We can easily visualize the annual climatology with
# xarray's built-in plotting methods.
da = ens.bundle('CORDEX').add_mean('CORDEX').first.mean('time')
da.plot.pcolormesh('lon', 'lat', col='names', col_wrap=3)
plt.savefig('test.png')
