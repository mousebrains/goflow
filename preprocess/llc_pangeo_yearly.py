"""LLC4320 -> NetCDF: chunked-time fetcher for full year (or any range).

Streaming version of llc_pangeo_bigger.py.  Processes time in CHUNK_DAYS
slices with a BUFFER_DAYS overlap at each end (to absorb Butterworth filter
edge effects), trims the buffer, and APPENDS to a single growing NetCDF.

Memory bounded to ~10-15 GB regardless of total time span.

Run: python preprocess/llc_pangeo_yearly.py YOUR_PROJECT_ID

Config below; defaults pull a 2-month test scope (April-May 2012).
Set FULL_YEAR=True to switch to 2011-09-15 -> 2012-09-21 (1 year).
"""
import sys
import os
import gc
import time
import numpy as np
import xarray as xr
import gcsfs
from scipy.signal import butter, filtfilt
from netCDF4 import Dataset as NCDataset
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info


import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('project', help='GCP project ID for requester-pays billing')
parser.add_argument('--full-year', action='store_true',
                    help='Fetch full LLC4320 year (2011-09-15 to 2012-09-21).  '
                         'Default is the 2-month test window (April-May 2012).')
parser.add_argument('--start', help='Override start date (YYYY-MM-DDTHH:MM:SS)')
parser.add_argument('--end',   help='Override end date (inclusive)')
parser.add_argument('--output', help='Override output NetCDF path')
ARGS = parser.parse_args()
PROJECT = ARGS.project
os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT

# ---------------- config ----------------
LAT_MIN, LAT_MAX = 25.0, 42.0
LON_MIN, LON_MAX = -80.0, -50.0
DLAT = DLON = 2.0 / 111.0   # ~2 km

if ARGS.full_year:
    WINDOW_START = ARGS.start  or '2011-09-15T00:00:00'
    WINDOW_END   = ARGS.end    or '2012-09-21T23:00:00'
    OUT_PATH     = ARGS.output or 'data/llc_pangeo_2011-09_to_2012-09.nc'
else:
    WINDOW_START = ARGS.start  or '2012-04-01T00:00:00'
    WINDOW_END   = ARGS.end    or '2012-05-31T23:00:00'
    OUT_PATH     = ARGS.output or 'data/llc_pangeo_2012-04_to_2012-05.nc'

CHUNK_DAYS  = 14
BUFFER_DAYS = 3       # Butterworth filtfilt needs ~2.5 days of context
CANDIDATE_FACES = (7, 10)

BUTTER_CUTOFF_HOURS = 18.0
BUTTER_ORDER = 5
# ----------------------------------------


def find_face_indices(grid, face, lat_min, lat_max, lon_min, lon_max):
    XC = grid.XC.isel(face=face).values
    YC = grid.YC.isel(face=face).values
    in_box = (YC >= lat_min) & (YC <= lat_max) & (XC >= lon_min) & (XC <= lon_max)
    if not in_box.any():
        return None
    j_inds, i_inds = np.where(in_box)
    j_min, j_max = int(j_inds.min()), int(j_inds.max())
    i_min, i_max = int(i_inds.min()), int(i_inds.max())
    j_slice = slice(j_min, j_max + 1)
    i_slice = slice(i_min, i_max + 1)
    return dict(
        face=face,
        j_slice=j_slice, i_slice=i_slice,
        i_g_slice=slice(i_min, min(i_max + 2, grid.sizes['i'])),
        j_g_slice=slice(j_min, min(j_max + 2, grid.sizes['j'])),
        XC=XC[j_slice, i_slice],
        YC=YC[j_slice, i_slice],
        cell_count=int(in_box.sum()),
    )


def chunk_windows(start, end, chunk_days, buffer_days):
    """Yield (chunk_start, chunk_end, output_start, output_end) datetimes.

    Each yielded chunk pulls [chunk_start, chunk_end] hourly inclusive
    (with buffer) and after Butterworth + trim emits valid output for
    [output_start, output_end].
    """
    s = np.datetime64(start)
    e = np.datetime64(end)
    one_day = np.timedelta64(1, 'D')
    one_h   = np.timedelta64(1, 'h')
    cur = s
    while cur <= e:
        out_end = min(cur + chunk_days * one_day - one_h, e)
        chunk_start = cur - buffer_days * one_day
        chunk_end   = out_end + buffer_days * one_day
        yield chunk_start, chunk_end, cur, out_end
        cur = out_end + one_h


def init_netcdf(path, target_lats, target_lons):
    nc = NCDataset(path, 'w')
    nc.createDimension('time', None)   # unlimited
    nc.createDimension('lat', target_lats.size)
    nc.createDimension('lon', target_lons.size)
    nc.createVariable('time', 'f8', ('time',)).units = 'seconds since 2000-01-01 00:00:00'
    nc.createVariable('lat',  'f4', ('lat',))[:]  = target_lats
    nc.createVariable('lon',  'f4', ('lon',))[:]  = target_lons
    for name, attrs in [
        ('loggrad_T', dict(long_name='log|grad SST|', normalization_range='[-19, 0]')),
        ('U', dict(long_name='LLC4320 surface U, cell-centered, 18hr LP', units='m/s')),
        ('V', dict(long_name='LLC4320 surface V, cell-centered, 18hr LP', units='m/s')),
    ]:
        v = nc.createVariable(name, 'f4', ('time', 'lat', 'lon'),
                              zlib=True, complevel=4, chunksizes=(24, 64, 64))
        for k, val in attrs.items():
            setattr(v, k, val)
    nc.source = 'LLC4320 chunked yearly fetch via pangeo-ecco-llc4320'
    nc.region = f'lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}'
    nc.recipe = (
        f'pyresample kd-tree (1 NN), 18-hr Butterworth low-pass on U,V '
        f'order {BUTTER_ORDER}, chunk size {CHUNK_DAYS} days with '
        f'{BUFFER_DAYS}-day buffers'
    )
    return nc


def process_chunk(chunk_start, chunk_end, output_start, output_end,
                  sst, ssu, ssv, face_info,
                  target_def, valid_in, valid_out, idx_array,
                  target_lats, target_lons, butter_b, butter_a):
    src_sst, src_u, src_v, src_lons, src_lats = [], [], [], [], []
    for info in face_info:
        f = info['face']
        sst_sub = sst.SST.sel(time=slice(str(chunk_start), str(chunk_end))).isel(
            face=f, j=info['j_slice'], i=info['i_slice'])
        ssu_sub = ssu.U.sel(time=slice(str(chunk_start), str(chunk_end))).isel(
            face=f, j=info['j_slice'], i_g=info['i_g_slice'])
        ssv_sub = ssv.V.sel(time=slice(str(chunk_start), str(chunk_end))).isel(
            face=f, j_g=info['j_g_slice'], i=info['i_slice'])
        sst_arr = sst_sub.values
        u_arr   = ssu_sub.values
        v_arr   = ssv_sub.values
        if u_arr.shape[2] == sst_arr.shape[2] + 1:
            u_arr = 0.5 * (u_arr[..., :-1] + u_arr[..., 1:])
        if v_arr.shape[1] == sst_arr.shape[1] + 1:
            v_arr = 0.5 * (v_arr[:, :-1, :] + v_arr[:, 1:, :])
        nt, ny, nx = sst_arr.shape
        src_sst.append(sst_arr.reshape(nt, ny * nx))
        src_u.append(u_arr.reshape(nt, ny * nx))
        src_v.append(v_arr.reshape(nt, ny * nx))
        src_lons.append(info['XC'].ravel())
        src_lats.append(info['YC'].ravel())
        time_axis = sst_sub.time.values

    # Free large source arrays once concatenated
    src_sst = np.concatenate(src_sst, axis=1)
    src_u   = np.concatenate(src_u,   axis=1)
    src_v   = np.concatenate(src_v,   axis=1)
    NLAT, NLON = target_lats.size, target_lons.size

    def resample(arr_t):
        out = np.empty((arr_t.shape[0], NLAT, NLON), dtype=np.float32)
        for t in range(arr_t.shape[0]):
            out[t] = get_sample_from_neighbour_info(
                'nn', target_def.shape, arr_t[t],
                valid_in, valid_out, idx_array, fill_value=np.nan)
        return out

    sst_grid = resample(src_sst); del src_sst
    u_grid   = resample(src_u);   del src_u
    v_grid   = resample(src_v);   del src_v

    # Butterworth on the WHOLE chunk (incl. buffer), then trim
    u_filt = filtfilt(butter_b, butter_a, u_grid, axis=0)
    v_filt = filtfilt(butter_b, butter_a, v_grid, axis=0)
    del u_grid, v_grid

    valid = (time_axis >= np.datetime64(output_start)) & \
            (time_axis <= np.datetime64(output_end))
    sst_grid = np.nan_to_num(sst_grid[valid].astype(np.float32))
    u_filt   = np.nan_to_num(u_filt[valid].astype(np.float32))
    v_filt   = np.nan_to_num(v_filt[valid].astype(np.float32))
    times    = time_axis[valid]

    # log|grad SST|
    mid_lat = 0.5 * (LAT_MIN + LAT_MAX)
    dy_m = DLAT * 111e3
    dx_m = DLON * 111e3 * np.cos(np.deg2rad(mid_lat))
    log_grad_t = np.empty_like(sst_grid)
    for t in range(sst_grid.shape[0]):
        gy, gx = np.gradient(sst_grid[t], dy_m, dx_m)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_grad_t[t] = np.clip(np.log(np.hypot(gx, gy)), -19, 0)
    log_grad_t = np.nan_to_num(log_grad_t, nan=-19.0).astype(np.float32)
    return log_grad_t, u_filt, v_filt, times


def main():
    fs = gcsfs.GCSFileSystem(token='google_default', project=PROJECT, requester_pays=True)
    print('Opening LLC4320 zarrs...')
    grid = xr.open_zarr(fs.get_mapper('pangeo-ecco-llc4320/grid'), consolidated=True)
    sst  = xr.open_zarr(fs.get_mapper('pangeo-ecco-llc4320/sst'),  consolidated=True)
    ssu  = xr.open_zarr(fs.get_mapper('pangeo-ecco-llc4320/ssu'),  consolidated=True)
    ssv  = xr.open_zarr(fs.get_mapper('pangeo-ecco-llc4320/ssv'),  consolidated=True)

    print(f'Target box: lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}')
    face_info = []
    for f in CANDIDATE_FACES:
        info = find_face_indices(grid, f, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        if info is not None:
            print(f'  face {f}: {info["cell_count"]:,} cells in box')
            face_info.append(info)
    if not face_info:
        sys.exit('No face has cells in box.')

    target_lats = np.arange(LAT_MIN, LAT_MAX + DLAT/2, DLAT)
    target_lons = np.arange(LON_MIN, LON_MAX + DLON/2, DLON)
    LL, LA = np.meshgrid(target_lons, target_lats)
    NLAT, NLON = target_lats.size, target_lons.size
    print(f'Target grid: {NLAT} x {NLON}')

    src_lons = np.concatenate([info['XC'].ravel() for info in face_info])
    src_lats = np.concatenate([info['YC'].ravel() for info in face_info])
    src_def    = SwathDefinition(lons=src_lons, lats=src_lats)
    target_def = SwathDefinition(lons=LL, lats=LA)
    print('Building pyresample kd-tree (once)...')
    valid_in, valid_out, idx_array, _ = get_neighbour_info(
        src_def, target_def, radius_of_influence=5000.0, neighbours=1)

    sample_rate_hz = 1.0 / 3600.0
    cutoff_hz      = 1.0 / (BUTTER_CUTOFF_HOURS * 3600.0)
    butter_b, butter_a = butter(BUTTER_ORDER, cutoff_hz / (0.5 * sample_rate_hz), btype='low')

    nc = init_netcdf(OUT_PATH, target_lats, target_lons)
    write_t = 0
    ref = np.datetime64('2000-01-01T00:00:00')

    chunks = list(chunk_windows(WINDOW_START, WINDOW_END, CHUNK_DAYS, BUFFER_DAYS))
    print(f'\n{len(chunks)} chunks of {CHUNK_DAYS} days (+{BUFFER_DAYS}-day buffer each side)\n')
    overall_start = time.time()
    for ci, (cs, ce, os_, oe) in enumerate(chunks):
        t0 = time.time()
        print(f'Chunk {ci+1}/{len(chunks)}: fetch [{cs}..{ce}] -> output [{os_}..{oe}]')
        log_grad_t, u_filt, v_filt, times = process_chunk(
            cs, ce, os_, oe,
            sst, ssu, ssv, face_info,
            target_def, valid_in, valid_out, idx_array,
            target_lats, target_lons, butter_b, butter_a)
        nt = times.size
        time_sec = (times - ref) / np.timedelta64(1, 's')
        nc.variables['time'][write_t:write_t + nt]      = time_sec
        nc.variables['loggrad_T'][write_t:write_t + nt] = log_grad_t
        nc.variables['U'][write_t:write_t + nt]         = u_filt
        nc.variables['V'][write_t:write_t + nt]         = v_filt
        nc.sync()
        write_t += nt
        del log_grad_t, u_filt, v_filt, times, time_sec
        gc.collect()
        dt = time.time() - t0
        print(f'  wrote {nt} steps in {dt:.1f}s; total written: {write_t}')

    nc.close()
    print(f'\nTotal time: {(time.time()-overall_start)/60:.1f} min')
    print(f'Output: {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1e9:.2f} GB)')

    del grid, sst, ssu, ssv, fs
    gc.collect()


if __name__ == '__main__':
    main()
