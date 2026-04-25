"""Verify GCP access to Pangeo's LLC4320 zarrs and discover their schema.

Run AFTER you've done:
    gcloud auth application-default login
    gcloud config set project YOUR_PROJECT_ID

Run as: python preprocess/probe_gcp_llc4320.py YOUR_PROJECT_ID

What this does (no large downloads):
  - Opens .zmetadata for grid, sst, ssu, ssv (a few KB each)
  - Prints dimensions, coords, and chunk shapes
  - Reads a tiny corner of the grid lat/lon arrays so we know face layout
  - Estimates per-snapshot byte size for the user's target region

Once this works, we use the discovered schema to write the real fetcher.
"""
import sys
import gc
import xarray as xr
import gcsfs
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python preprocess/probe_gcp_llc4320.py YOUR_PROJECT_ID", file=sys.stderr)
    sys.exit(1)

PROJECT = sys.argv[1]
import os
os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT
LAT_BOX = (25.0, 42.0)
LON_BOX = (-80.0, -50.0)

fs = gcsfs.GCSFileSystem(token='google_default', project=PROJECT, requester_pays=True)


def probe(name):
    print(f'\n========== {name} ==========')
    store = fs.get_mapper(f'pangeo-ecco-llc4320/{name}')
    ds = xr.open_zarr(store, consolidated=True)
    print(f'Dimensions: {dict(ds.sizes)}')
    print(f'Coords: {list(ds.coords)}')
    print(f'Data vars: {list(ds.data_vars)}')
    for v in ds.data_vars:
        a = ds[v]
        chunks = getattr(a.data, 'chunksize', '?')
        print(f'  {v}: shape={a.shape}, dtype={a.dtype}, chunks={chunks}')
    return ds


# Grid first — gives us lat/lon for cell centers
ds_grid = probe('grid')

# Identify the lat/lon vars in the grid (typical names: XC, YC for tracer points)
lat_name = next((n for n in ('YC', 'lat', 'latitude') if n in ds_grid), None)
lon_name = next((n for n in ('XC', 'lon', 'longitude') if n in ds_grid), None)
print(f'\nLat var: {lat_name}, Lon var: {lon_name}')
if lat_name and lon_name:
    YC = ds_grid[lat_name]
    XC = ds_grid[lon_name]
    print(f'Lat full range: {float(YC.min())} to {float(YC.max())}')
    print(f'Lon full range: {float(XC.min())} to {float(XC.max())}')
    # Find which face(s) intersect our target box
    if 'face' in YC.dims:
        for f in range(YC.sizes['face']):
            yf = YC.isel(face=f)
            xf = XC.isel(face=f)
            ymin, ymax = float(yf.min()), float(yf.max())
            xmin, xmax = float(xf.min()), float(xf.max())
            hits = (ymax > LAT_BOX[0] and ymin < LAT_BOX[1]
                    and xmax > LON_BOX[0] and xmin < LON_BOX[1])
            tag = '<<< HIT' if hits else ''
            print(f'  face {f}: lat [{ymin:6.1f},{ymax:6.1f}], '
                  f'lon [{xmin:6.1f},{xmax:6.1f}]  {tag}')

# SST/SSU/SSV
for name in ('sst', 'ssu', 'ssv'):
    ds = probe(name)
    if 'time' in ds.sizes:
        print(f'  time range: {ds.time.values[0]} to {ds.time.values[-1]}')
        dt = ds.time.values[1] - ds.time.values[0]
        print(f'  step: {dt}')

print('\nProbe complete.  If you saw the four datasets and their dimensions, GCP is wired up.')

# Tear down gcsfs/aiohttp before interpreter exit (Python 3.14 asyncio cleanup)
del fs
gc.collect()
