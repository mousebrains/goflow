"""Adapter: convert the published Zenodo testset NetCDF into the schema
train_goflow.py / dataSST.SSTDataset expects (loggrad_T, U, V).

This is a stand-in for a full Pangeo LLC4320 pull while we get GCP set up.
The Zenodo testset is real LLC4320 data — just a 41-snapshot, 256x256 slice.

Run: python preprocess/llc_from_testset.py
Output: data/llc_from_testset.nc
"""
import os
import numpy as np
import xarray as xr
from netCDF4 import Dataset as NCDataset

SRC = 'data/testset_llc4320_0.2.nc'
DST = 'data/llc_from_testset.nc'

# The Zenodo file's gradT is already normalized to [0,1] from raw log range
# [lgtMin=-19, lgtMax=0]. We want to store RAW log values so SSTDataset can
# re-normalize via its standard formula (x - lgtMin) / (lgtMax - lgtMin).
LGT_MIN, LGT_MAX = -19.0, 0.0


def main():
    src = xr.open_dataset(SRC)
    print(f'Input: {SRC}')
    print(f'  shape: {dict(src.sizes)}')

    # Recover raw log gradient values
    loggrad_T = src.gradT.values * (LGT_MAX - LGT_MIN) + LGT_MIN

    # Truth velocities → U, V (ground truth is in *_inp variables)
    U = src.U_inp.values
    V = src.V_inp.values

    # Time/lat/lon coords are missing in the Zenodo file; synthesize evenly
    # spaced placeholders. The GOFLOW dataset code only uses time as an index.
    nt, ny, nx = U.shape
    time_synth = np.arange(nt, dtype='f8') * 3600.0  # 1 hour spacing
    lat_synth = np.linspace(35.0, 40.3, ny)  # 5.3° box, paper-style
    lon_synth = np.linspace(-70.0, -64.7, nx)

    print(f'  loggrad_T raw range: {loggrad_T.min():.2f} to {loggrad_T.max():.2f}')
    print(f'  U range: {U.min():.3f} to {U.max():.3f} m/s')
    print(f'  V range: {V.min():.3f} to {V.max():.3f} m/s')

    print(f'\nWriting {DST}')
    with NCDataset(DST, 'w') as nc:
        nc.createDimension('time', nt)
        nc.createDimension('lat', ny)
        nc.createDimension('lon', nx)
        nc.createVariable('time', 'f8', ('time',))[:] = time_synth
        nc.createVariable('lat', 'f4', ('lat',))[:] = lat_synth
        nc.createVariable('lon', 'f4', ('lon',))[:] = lon_synth

        v = nc.createVariable('loggrad_T', 'f4', ('time', 'lat', 'lon'),
                              zlib=True, complevel=4)
        v[:] = loggrad_T.astype('f4')
        v.long_name = 'log of SST gradient magnitude'
        v.normalization_range = f'[{LGT_MIN}, {LGT_MAX}]'

        for name, arr in [('U', U), ('V', V)]:
            v = nc.createVariable(name, 'f4', ('time', 'lat', 'lon'),
                                  zlib=True, complevel=4)
            v[:] = arr.astype('f4')
            v.long_name = f'LLC4320 surface {name} velocity'
            v.units = 'm/s'

        nc.source = (
            f'Repackaged from {os.path.basename(SRC)} (Zenodo 15815704). '
            'gradT denormalized to raw log range, used as loggrad_T.'
        )

    out_size_mb = os.path.getsize(DST) / 1e6
    print(f'Done. Output: {nt} snapshots × {ny}×{nx}, {out_size_mb:.1f} MB')


if __name__ == '__main__':
    main()
