# Default values of coordinate names. Primarily used by instances of
# bcdp.Adapter when remapping coordinate variables
X_LABELS = ['x', 'lon', 'lons', 'longitude', 'longitudes', 'rlon']
Y_LABELS = ['y', 'lat', 'lats', 'latitude', 'latitudes', 'rlat']
Z_LABELS = ['lev', 'plev', 'level']
T_LABELS = ['time', 'times', 'date', 'dates', 'julian']

# CF-compliant coordinate variable attribute defaults
X_ATTR = dict(name='x', gridname='lon', labels=X_LABELS,
              units='degrees_east', axis='X', standard_name='longitude',
              long_name='longitude')
Y_ATTR = dict(name='y', gridname='lat', labels=Y_LABELS,
              units='degrees_north', axis='Y', standard_name='latitude',
              long_name='latitude')
Z_ATTR = dict(name='lev', labels=Z_LABELS, axis='Z')
T_ATTR = dict(name='time', labels=T_LABELS, axis='T',
              standard_name='time', long_name='time')
ATTRS = dict(X=X_ATTR, Y=Y_ATTR, Z=Z_ATTR, T=T_ATTR)

# RCMED URL
RCMED_QUERY_URL = 'http://rcmes.jpl.nasa.gov/query-api/query.php?'

# ESGF Index/Search nodes
ESGF_NODES = {'LLNL': 'esgf-node.llnl.gov',
              'JPL': 'esgf-node.jpl.nasa.gov',
              'CEDA': 'esgf-index1.ceda.ac.uk',
              'DKRZ': 'esgf-data.dkrz.de',
              'LIU': 'esg-dn1.nsc.liu.se',
              'IPSL': 'esgf-node.ipsl.upmc.fr',
              'NCI': 'esgf.nci.org.au',
              'GFDL': 'esgdata.gfdl.noaa.gov',
              'GSFC': 'esgf.nccs.nasa.gov'}
