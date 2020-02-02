# bcdp
Big Climate Data Pipeline

This library is intended to be a drop in replacement for Apache [OCW](climate.apache.org), a climate model evaluation processing pipeline.

**NOTE**: This package is still essentially in alpha is missing some features. The current API is subject to major changes.

# Installation
`bcdp` requires Python 3.6+. Currently available via conda:
```
conda install -c conda-forge bcdp
```
or `pip`:
```
pip install bcdp
```

## Dependencies
Assuming you have the [conda](https://conda.io/miniconda.html) package manager installed:
```
conda install -c conda-forge xesmf pyresample cartopy
```

# Quick Example
For this example, you'll need to download some sample [CORDEX Africa simulation data](https://rcmes.jpl.nasa.gov/RCMES_Turtorial_data/CORDEX-Africa_data.tar.gz).

To run, edit the value of the `paths` variable in `examples/test.py`, for example:
```python
paths = os.path.join(os.path.expanduser('~'), 'data/CORDEX_Africa/*clt*')
```

Then:
```
cd bcdp/examples/scripts
python test.py
```
