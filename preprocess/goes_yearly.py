"""GOES-16 ABI Band 14 -> NetCDF for GOFLOW inference, full year + full region.

Generalizes goes_pilot.py: configurable date range + region, concurrent S3
downloads, streaming write so memory stays bounded regardless of total span.

Run:
    python preprocess/goes_yearly.py
    python preprocess/goes_yearly.py --start 2023-01-01 --end 2023-12-31
    python preprocess/goes_yearly.py --lat-min 25 --lat-max 42 \
                                     --lon-min -80 --lon-max -50

Output (default): data/goes_2023_full.nc
Cache (default): data/_goes_cache_year/  (~44 GB for full year, can delete after)
"""
import os
import sys
import gc
import time
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import xarray as xr
import s3fs
import pyproj
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset as NCDataset


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--start', default='2023-01-01', help='YYYY-MM-DD inclusive')
parser.add_argument('--end',   default='2023-12-31', help='YYYY-MM-DD inclusive')
parser.add_argument('--lat-min', type=float, default=25.0)
parser.add_argument('--lat-max', type=float, default=42.0)
parser.add_argument('--lon-min', type=float, default=-80.0)
parser.add_argument('--lon-max', type=float, default=-50.0)
parser.add_argument('--output', default='data/goes_2023_full.nc')
parser.add_argument('--cache',  default='data/_goes_cache_year')
parser.add_argument('--workers', type=int, default=8,
                    help='Concurrent S3 download threads')
parser.add_argument('--per-hour', type=int, default=1,
                    help='Files per hour to keep: 1 = first only (hourly cadence), '
                         '0 = all (5-min cadence, ~12x data)')
ARGS = parser.parse_args()

LAT_MIN, LAT_MAX = ARGS.lat_min, ARGS.lat_max
LON_MIN, LON_MAX = ARGS.lon_min, ARGS.lon_max
DLAT = DLON = 2.0 / 111.0   # ~2 km

target_lats = np.arange(LAT_MIN, LAT_MAX + DLAT/2, DLAT)
target_lons = np.arange(LON_MIN, LON_MAX + DLON/2, DLON)
NLAT, NLON = target_lats.size, target_lons.size
LL, LA = np.meshgrid(target_lons, target_lats)

mid_lat = 0.5 * (LAT_MIN + LAT_MAX)
dy_m = DLAT * 111e3
dx_m = DLON * 111e3 * np.cos(np.deg2rad(mid_lat))

BUCKET, SECTOR, CHANNEL = 'noaa-goes16', 'ABI-L1b-RadC', 14


def daterange(start, end):
    d0 = datetime.strptime(start, '%Y-%m-%d')
    d1 = datetime.strptime(end, '%Y-%m-%d')
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def list_per_hour(fs, year, day, per_hour=1):
    """Return Channel-14 RadC files for the given day.

    per_hour: 1 = first file each hour (hourly cadence, ~24/day),
              0 = all files (5-min cadence, ~288/day).
    """
    out = []
    for h in range(24):
        prefix = f'{BUCKET}/{SECTOR}/{year}/{day:03d}/{h:02d}'
        try:
            files = fs.ls(prefix, detail=False)
            c14 = sorted(f for f in files if f'C{CHANNEL:02d}' in f)
            if per_hour == 1:
                out.append(c14[0] if c14 else None)
            else:
                out.extend(c14)
        except Exception as e:
            print(f'  list ERR {prefix}: {e}')
            if per_hour == 1:
                out.append(None)
    return out


def download(fs, key):
    if key is None:
        return None
    local = os.path.join(ARGS.cache, os.path.basename(key))
    if os.path.exists(local) and os.path.getsize(local) > 1000:
        return local
    try:
        fs.get(key, local)
        return local
    except Exception as e:
        print(f'  dl ERR {key}: {e}')
        return None


def planck_bt(rad, fk1, fk2, bc1, bc2):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (fk2 / np.log((fk1 / rad) + 1.0) - bc1) / bc2


_geo_cache = {}


def project_to_latlon(local_path):
    """Open one ABI L1b file, return (BT_K, log_grad, mask_u1, time_sec_2000)."""
    ds = xr.open_dataset(local_path)
    p = ds.goes_imager_projection.attrs
    cache_key = (float(p['perspective_point_height']),
                 float(p['longitude_of_projection_origin']),
                 p['sweep_angle_axis'])
    if cache_key not in _geo_cache:
        h = p['perspective_point_height']
        geos = pyproj.Proj(
            proj='geos', h=h,
            lon_0=p['longitude_of_projection_origin'],
            sweep=p['sweep_angle_axis'],
            a=p['semi_major_axis'], b=p['semi_minor_axis'])
        x_m, y_m = geos(LL, LA)
        _geo_cache[cache_key] = (y_m / h, x_m / h)
    y_rad, x_rad = _geo_cache[cache_key]

    bt = planck_bt(ds.Rad.values,
                   float(ds.planck_fk1), float(ds.planck_fk2),
                   float(ds.planck_bc1), float(ds.planck_bc2))
    y_axis = ds.y.values
    x_axis = ds.x.values
    if y_axis[0] > y_axis[-1]:
        y_axis = y_axis[::-1]
        bt = bt[::-1, :]
    rgi = RegularGridInterpolator(
        (y_axis, x_axis), bt,
        bounds_error=False, fill_value=np.nan, method='linear')
    pts = np.column_stack([y_rad.ravel(), x_rad.ravel()])
    bt_grid = rgi(pts).reshape(NLAT, NLON).astype(np.float32)

    gy, gx = np.gradient(bt_grid, dy_m, dx_m)
    mag = np.hypot(gx, gy)
    log_grad = np.full_like(mag, np.nan, dtype=np.float32)
    valid = np.isfinite(mag) & (mag > 0)
    log_grad[valid] = np.log(mag[valid])
    log_grad = np.clip(log_grad, -19, 0)
    mask = (np.isfinite(bt_grid) & (bt_grid > 270.0)).astype(np.uint8)
    # ABI L1b stores t as "seconds since 2000-01-01 12:00:00" (J2000), but xarray
    # auto-decodes it to datetime64[ns]. float() on that gives ns-since-1970.
    # Convert back to seconds since J2000 to match the file's units attribute.
    J2000 = np.datetime64('2000-01-01T12:00:00')
    t_sec = float((ds.t.values - J2000) / np.timedelta64(1, 's'))
    ds.close()
    return bt_grid, log_grad, mask, t_sec


def init_netcdf(path):
    nc = NCDataset(path, 'w')
    nc.createDimension('time', None)
    nc.createDimension('lat', NLAT)
    nc.createDimension('lon', NLON)
    nc.createVariable('time', 'f8', ('time',)).units = 'seconds since 2000-01-01 12:00:00'
    nc.createVariable('lat', 'f4', ('lat',))[:] = target_lats
    nc.createVariable('lon', 'f4', ('lon',))[:] = target_lons
    for name, dtype, attrs in [
        ('BT', 'f4', dict(units='K',
                          long_name='Brightness temperature, ABI Band 14 (11.2 um)')),
        ('log_gradT', 'f4', dict(long_name='log|grad BT|',
                                 normalization_range='[-19, 0]')),
        ('mask', 'u1', dict(long_name='1=valid open-ocean, 0=cloud/land/missing')),
    ]:
        v = nc.createVariable(name, dtype, ('time', 'lat', 'lon'),
                              zlib=True, complevel=4,
                              chunksizes=(24, 64, 64))
        for k, val in attrs.items():
            setattr(v, k, val)
    nc.source = (f'GOES-16 ABI L1b-{SECTOR.split("-")[-1]} channel {CHANNEL}, '
                 f'1 hourly file per day')
    nc.region = f'lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}'
    return nc


def main():
    os.makedirs(os.path.dirname(ARGS.output) or '.', exist_ok=True)
    os.makedirs(ARGS.cache, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)
    days = list(daterange(ARGS.start, ARGS.end))
    frames_per_day = 24 if ARGS.per_hour == 1 else 288
    target_total = frames_per_day * len(days)
    print(f'GOES yearly: {ARGS.start} -> {ARGS.end} ({len(days)} days, '
          f'~{target_total} target frames @ {frames_per_day}/day)')
    print(f'  region:  lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}')
    print(f'  grid:    {NLAT} x {NLON}  (~{DLAT*111:.1f} km)')
    print(f'  output:  {ARGS.output}')
    print(f'  workers: {ARGS.workers}')
    print()

    nc = init_netcdf(ARGS.output)
    write_t = 0
    t_overall = time.time()

    for day in days:
        year = day.year
        doy  = day.timetuple().tm_yday
        keys = list_per_hour(fs, year, doy, per_hour=ARGS.per_hour)
        with ThreadPoolExecutor(max_workers=ARGS.workers) as ex:
            local_paths = list(ex.map(lambda k: download(fs, k), keys))

        bt_list, lg_list, mk_list, t_list = [], [], [], []
        for lp in local_paths:
            if lp is None:
                continue
            try:
                bt, lg, mk, ts = project_to_latlon(lp)
                bt_list.append(bt); lg_list.append(lg)
                mk_list.append(mk); t_list.append(ts)
            except Exception as e:
                print(f'  proj ERR {lp}: {e}')
        if not bt_list:
            print(f'  {day.date()}: no valid frames; skipping')
            continue
        bt_arr = np.stack(bt_list)
        lg_arr = np.stack(lg_list)
        mk_arr = np.stack(mk_list)
        t_arr  = np.array(t_list, dtype='f8')
        nt = bt_arr.shape[0]
        nc.variables['time'][write_t:write_t+nt]      = t_arr
        nc.variables['BT'][write_t:write_t+nt]        = bt_arr
        nc.variables['log_gradT'][write_t:write_t+nt] = lg_arr
        nc.variables['mask'][write_t:write_t+nt]      = mk_arr
        nc.sync()
        write_t += nt
        elapsed = time.time() - t_overall
        rate = write_t / elapsed if elapsed > 0 else 0
        eta = (target_total - write_t) / rate / 60 if rate > 0 else 0
        print(f'  {day.date()}: +{nt}  total {write_t}/{target_total}  '
              f'rate {rate:.1f} fr/s  ETA {eta:.0f} min')
        del bt_list, lg_list, mk_list, t_list, bt_arr, lg_arr, mk_arr, t_arr
        gc.collect()

    nc.close()
    print(f'\nDone in {(time.time()-t_overall)/60:.1f} min. {write_t} frames.')
    print(f'Output: {ARGS.output}  ({os.path.getsize(ARGS.output)/1e9:.2f} GB)')
    del fs
    gc.collect()


if __name__ == '__main__':
    main()
