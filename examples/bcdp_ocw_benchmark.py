from contextlib import contextmanager
from datetime import datetime
import time
import os
import glob
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bcdp
import ocw.data_source.local as local
import ocw.dataset_processor as dsp
from ocw.dataset import Bounds as Bounds


@contextmanager
def time_block(results, name):
    print(name)
    t0 = time.time()
    yield
    t1 = time.time()
    results[name] = t1 - t0

def dt64_to_datetime(dt64):
    t = dt64.astype(int)*1e-9
    dt = datetime.utcfromtimestamp(t)
    return dt
    
# Dataset paths
paths = os.path.join(os.path.expanduser('~'), 'data/CORDEX_Africa/*clt*')

### BCDP SECTION
print('BCDP Benchmarks')
bcdp_results = {}
with time_block(bcdp_results, 'Dataset Loading'):
    project = 'CORDEX-Africa'
    template = '*_{model}_*_{variable}.nc'
    bcdp.build_extractor(project, template, name_field='model', index=[1, 6])
    loader = bcdp.LocalFileSource()
    ens = loader(paths=paths, project=project)
    
# Ouput grid info
domain = ens.overlap
output_grid = bcdp.utils.grid_from_res((0.88, 0.88), domain)
new_lats = output_grid.lat.values
new_lons = output_grid.lon.values
start_time = dt64_to_datetime(domain.time_bnds.min)
end_time = dt64_to_datetime(domain.time_bnds.max)
bnds = Bounds(lat_min=domain.lat_bnds.min, lat_max=domain.lat_bnds.max,
              lon_min=domain.lon_bnds.min, lon_max=domain.lon_bnds.max,
              start=start_time, end=end_time)


with time_block(bcdp_results, 'Domain Subsetting'):
    ens = ens.subset()
    
with time_block(bcdp_results, 'Seasonal Subsetting'):
    ens = ens.select_season(season='DJF')
    
with time_block(bcdp_results, 'Resampling'):
    ens = ens.resample(freq='Y')
    
with time_block(bcdp_results, 'Regridding'):
    ens.regrid(backend='scipy', method='linear', output_grid=output_grid)

print(f'BCDP Results: {bcdp_results}')

### OCW SECTION
print('OCW Benchmarks')
ocw_results = {}
with time_block(ocw_results, 'Dataset Loading'):
    datasets = local.load_multiple_files(paths, 'clt')

with time_block(ocw_results, 'Domain Subsetting'):
    for i, ds in enumerate(datasets):
        datasets[i] = dsp.subset(ds, bnds)

with time_block(ocw_results, 'Seasonal Subsetting'):
    for i, ds in enumerate(datasets):
        datasets[i] = dsp.temporal_subset(ds, 9, 11)
    
with time_block(ocw_results, 'Resampling'):
    for i, ds in enumerate(datasets):
        datasets[i] = dsp.temporal_rebin(ds, 'annual')
        
with time_block(ocw_results, 'Regridding'):
    for i, ds in enumerate(datasets):
        datasets[i] = dsp.spatial_regrid(ds, new_lats, new_lons)

print(f'OCW Results: {ocw_results}')

# Plot results
matplotlib.style.use('ggplot')
df = pd.DataFrame({'OCW': ocw_results, 'BCDP': bcdp_results})
df.plot.bar(logy=True, rot=12)
for p in ax.patches:
    val = np.round(p.get_height(), decimals=2)
    ax.annotate(str(val), (p.get_x() + .02, p.get_height()), size=9.5)

plt.ylabel('Running Time [s]')
plt.savefig('bcdp_ocw_benchmarks.png')
