"""Center-crop the bigger LLC4320 pilot (944x1666) to 256x256 for end-to-end
training validation on MPS. Keeps full time axis.

Run: python preprocess/crop_llc_bigger.py
Output: data/llc_train_256.nc
"""
import os
import xarray as xr
import numpy as np
from netCDF4 import Dataset as NCDataset

SRC = 'data/llc_pangeo_bigger_2012-04-08_to_2012-04-21.nc'
DST = 'data/llc_train_256.nc'
N = 256


def main():
    ds = xr.open_dataset(SRC, decode_times=False)
    print(f'Source: {dict(ds.sizes)}')

    # Center crop indices
    yc = ds.sizes['lat'] // 2
    xc = ds.sizes['lon'] // 2
    j0, j1 = yc - N // 2, yc + N // 2
    i0, i1 = xc - N // 2, xc + N // 2
    print(f'  cropping lat[{j0}:{j1}] lon[{i0}:{i1}]')
    print(f'  lat range: {float(ds.lat[j0]):.2f} - {float(ds.lat[j1-1]):.2f}')
    print(f'  lon range: {float(ds.lon[i0]):.2f} - {float(ds.lon[i1-1]):.2f}')

    sub = ds.isel(lat=slice(j0, j1), lon=slice(i0, i1))
    nt = sub.sizes['time']
    print(f'  output shape: ({nt}, {N}, {N})')

    with NCDataset(DST, 'w') as nc:
        nc.createDimension('time', nt)
        nc.createDimension('lat', N)
        nc.createDimension('lon', N)
        v = nc.createVariable('time', 'f8', ('time',))
        v.units = sub.time.attrs.get('units', 'seconds since 2000-01-01 00:00:00')
        v[:] = sub.time.values
        nc.createVariable('lat', 'f4', ('lat',))[:] = sub.lat.values
        nc.createVariable('lon', 'f4', ('lon',))[:] = sub.lon.values
        for name in ('loggrad_T', 'U', 'V'):
            arr = sub[name].values.astype('f4')
            # nan_to_num: SSTDataset only nan_to_nums the input, not U/V targets.
            # Land/edge NaNs would otherwise poison the L1 loss.
            arr = np.nan_to_num(arr, nan=0.0)
            var = nc.createVariable(name, 'f4', ('time', 'lat', 'lon'),
                                    zlib=True, complevel=4)
            var[:] = arr
            var.long_name = sub[name].attrs.get('long_name', name)
            for k in ('units', 'normalization_range'):
                if k in sub[name].attrs:
                    setattr(var, k, sub[name].attrs[k])
        nc.source = f'Center 256x256 crop of {os.path.basename(SRC)}'
    print(f'\nWrote {DST}: {os.path.getsize(DST)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
