"""GOES-16 ABI Band 14 → NetCDF for GOFLOW inference, 1-day pilot.

Pulls 24 hourly Channel-14 CONUS files from the public NOAA AWS bucket,
projects each into a regular lat/lon grid, computes log|grad BT|, and writes
a NetCDF in the schema dataSST.SatelliteDataset expects (BT, log_gradT, mask).

Run: python preprocess/goes_pilot.py
Output: data/goes_pilot_<date>.nc

Tunable globals at top of file: DATE, LAT/LON box, target grid size.
"""
import os
import sys
import gc
from datetime import datetime
import numpy as np
import xarray as xr
import s3fs
import pyproj
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset as NCDataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Pilot configuration
# ---------------------------------------------------------------------------
DATE_START = '2023-04-13'            # YYYY-MM-DD (paper Fig. 2 = day 1)
DATE_END   = '2023-04-14'            # inclusive; 2 days = 48 frames
LAT_MIN, LAT_MAX = 35.0, 40.3        # 5.3 deg box (paper convention)
LON_MIN, LON_MAX = -70.0, -64.7
NLAT, NLON = 256, 256                # output grid resolution

BUCKET = 'noaa-goes16'
SECTOR = 'ABI-L1b-RadC'              # CONUS sector
CHANNEL = 14                          # Band 14, 11.2 micron longwave IR

OUT_DIR = 'data'
CACHE_DIR = 'data/_goes_cache'
LGT_MIN, LGT_MAX = -19.0, 0.0        # SSTDataset normalization expects this range


def doy(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d').timetuple().tm_yday


def list_hour_files(fs, year, day, hour):
    prefix = f'{BUCKET}/{SECTOR}/{year}/{day:03d}/{hour:02d}'
    files = fs.ls(prefix, detail=False)
    return sorted(f for f in files if f'C{CHANNEL:02d}' in f)


def download_first_per_hour(fs, year, day):
    cached = []
    for h in range(24):
        files = list_hour_files(fs, year, day, h)
        if not files:
            print(f'  WARN no Band {CHANNEL} files for hour {h}')
            continue
        src = files[0]
        local = os.path.join(CACHE_DIR, os.path.basename(src))
        if not os.path.exists(local):
            fs.get(src, local)
        cached.append(local)
    return cached


def planck_bt(rad, fk1, fk2, bc1, bc2):
    """ABI L1b radiance -> brightness temperature (K)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return (fk2 / np.log((fk1 / rad) + 1.0) - bc1) / bc2


def project_to_latlon(ds, target_lats, target_lons):
    """Resample one ABI L1b scene from geostationary scan angles to a regular lat/lon grid."""
    p = ds.goes_imager_projection.attrs
    geos = pyproj.Proj(
        proj='geos',
        h=p['perspective_point_height'],
        lon_0=p['longitude_of_projection_origin'],
        sweep=p['sweep_angle_axis'],
        a=p['semi_major_axis'],
        b=p['semi_minor_axis'],
    )
    # Forward-transform target lat/lon -> scan-angle radians (matching ds.x, ds.y)
    LL, LA = np.meshgrid(target_lons, target_lats)
    x_m, y_m = geos(LL, LA)
    h = p['perspective_point_height']
    x_rad = x_m / h
    y_rad = y_m / h

    bt = planck_bt(
        ds.Rad.values,
        float(ds.planck_fk1), float(ds.planck_fk2),
        float(ds.planck_bc1), float(ds.planck_bc2),
    )

    # ds.y is decreasing; RegularGridInterpolator wants increasing axes
    y_axis = ds.y.values
    x_axis = ds.x.values
    if y_axis[0] > y_axis[-1]:
        y_axis = y_axis[::-1]
        bt = bt[::-1, :]

    rgi = RegularGridInterpolator(
        (y_axis, x_axis), bt,
        bounds_error=False, fill_value=np.nan, method='linear',
    )
    pts = np.column_stack([y_rad.ravel(), x_rad.ravel()])
    return rgi(pts).reshape(LA.shape)


def compute_log_gradient(field, dx_m, dy_m):
    gy, gx = np.gradient(field, dy_m, dx_m)
    mag = np.hypot(gx, gy)
    valid = np.isfinite(mag) & (mag > 0)
    log_mag = np.full_like(mag, np.nan)
    log_mag[valid] = np.log(mag[valid])
    return np.clip(log_mag, LGT_MIN, LGT_MAX)


def daterange(start, end):
    from datetime import timedelta
    d0 = datetime.strptime(start, '%Y-%m-%d')
    d1 = datetime.strptime(end, '%Y-%m-%d')
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    fs = s3fs.S3FileSystem(anon=True)
    days = list(daterange(DATE_START, DATE_END))
    print(f'GOES preprocessor pilot: {DATE_START} to {DATE_END}  ({len(days)} days)')
    print(f'  region:  lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}')
    print(f'  grid:    {NLAT} x {NLON}')

    target_lats = np.linspace(LAT_MIN, LAT_MAX, NLAT)
    target_lons = np.linspace(LON_MIN, LON_MAX, NLON)
    mid_lat = 0.5 * (LAT_MIN + LAT_MAX)
    dy_m = (target_lats[1] - target_lats[0]) * 111e3
    dx_m = (target_lons[1] - target_lons[0]) * 111e3 * np.cos(np.deg2rad(mid_lat))
    print(f'  grid spacing: dy={dy_m:.0f} m, dx={dx_m:.0f} m at midlat')

    print('\nDownloading hourly Band-14 files...')
    files = []
    for d in days:
        files += download_first_per_hour(fs, d.year, d.timetuple().tm_yday)
    print(f'  cached {len(files)} files in {CACHE_DIR}')

    print('\nReprojecting + computing log|grad BT|...')
    bt_t, lg_t, time_t = [], [], []
    for f in tqdm(files):
        ds = xr.open_dataset(f)
        bt = project_to_latlon(ds, target_lats, target_lons)
        lg = compute_log_gradient(bt, dx_m, dy_m)
        bt_t.append(bt)
        lg_t.append(lg)
        time_t.append(float(ds.t.values))   # central scene time, sec since 2000-01-01 12:00 UTC
        ds.close()

    bt_arr = np.stack(bt_t).astype(np.float32)
    lg_arr = np.stack(lg_t).astype(np.float32)
    time_arr = np.array(time_t, dtype=np.float64)

    # Crude cloud screen: BT > 270 K is the open-ocean criterion
    mask = (np.isfinite(bt_arr) & (bt_arr > 270.0)).astype(np.uint8)

    tag = f'{DATE_START}_to_{DATE_END}' if DATE_START != DATE_END else DATE_START
    out = os.path.join(OUT_DIR, f'goes_pilot_{tag}.nc')
    print(f'\nWriting {out}')
    with NCDataset(out, 'w') as nc:
        nc.createDimension('time', len(time_arr))
        nc.createDimension('lat', NLAT)
        nc.createDimension('lon', NLON)
        v = nc.createVariable('time', 'f8', ('time',))
        v.units = 'seconds since 2000-01-01 12:00:00'
        v[:] = time_arr
        nc.createVariable('lat', 'f4', ('lat',))[:] = target_lats
        nc.createVariable('lon', 'f4', ('lon',))[:] = target_lons
        for name, arr, attrs in [
            ('BT', bt_arr,
             dict(long_name='Brightness temperature, ABI Band 14 (11.2 um)', units='K')),
            ('log_gradT', lg_arr,
             dict(long_name='log of |grad BT|', normalization_range=f'[{LGT_MIN}, {LGT_MAX}]')),
            ('mask', mask,
             dict(long_name='1=open-ocean valid, 0=cloud/land/missing')),
        ]:
            dtype = 'u1' if name == 'mask' else 'f4'
            var = nc.createVariable(name, dtype, ('time', 'lat', 'lon'),
                                    zlib=True, complevel=4)
            var[:] = arr
            for k, val in attrs.items():
                setattr(var, k, val)
        nc.source = f'GOES-16 ABI L1b-RadC channel {CHANNEL}, hourly subset of {tag}'
        nc.region = f'lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}'

    valid_mean = mask.mean()
    valid_bt = bt_arr[mask == 1]
    print(f'\nResults:')
    print(f'  shape:          {bt_arr.shape}')
    print(f'  output size:    {os.path.getsize(out)/1e6:.1f} MB')
    print(f'  valid coverage: {valid_mean*100:.1f}%')
    if valid_bt.size:
        print(f'  BT (valid):     {valid_bt.min():.1f} - {valid_bt.max():.1f} K')
    print(f'  log_gradT range: {np.nanmin(lg_arr):.2f} to {np.nanmax(lg_arr):.2f}')

    # Tear down s3fs/aiohttp before interpreter exit
    del fs
    gc.collect()


if __name__ == '__main__':
    main()
