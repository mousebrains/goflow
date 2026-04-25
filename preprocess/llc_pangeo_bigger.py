"""LLC4320 -> NetCDF: 2-week, full-region bigger pilot.

Improvements over llc_pangeo_pilot.py:
  - Multi-face coverage check (faces 7 and 10 over the Atlantic)
  - pyresample kd_tree resampler (KDTree built once, reused per snapshot)
  - 18-hour Butterworth low-pass on U and V (paper Methods),
    with extra buffer days to absorb filter edge effects
  - Configurable region and date window (defaults to user's 25-42N, 80-50W,
    14 days in April 2012 + 2-day buffer at each end)

Run AFTER GCP setup:
    python preprocess/llc_pangeo_bigger.py YOUR_PROJECT_ID

Output: data/llc_pangeo_bigger_<start>_to_<end>.nc
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
from tqdm import tqdm
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info


if len(sys.argv) < 2:
    print('Usage: python preprocess/llc_pangeo_bigger.py YOUR_PROJECT_ID', file=sys.stderr)
    sys.exit(1)

PROJECT = sys.argv[1]
os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT

# Full target region
LAT_MIN, LAT_MAX = 25.0, 42.0
LON_MIN, LON_MAX = -80.0, -50.0

# Output regular-grid spacing (~2 km matches LLC4320 native)
DLAT = DLON = 2.0 / 111.0  # ~ 2 km in degrees

# Time window: keep BUFFER_DAYS at each end for Butterworth filter edge effects
WINDOW_START = '2012-04-08T00:00:00'  # output validity start
WINDOW_END   = '2012-04-21T23:00:00'  # output validity end
BUFFER_DAYS  = 2                        # padding for filter

CANDIDATE_FACES = (7, 10)  # both overlap our lat band; probe will pick

OUT_DIR = 'data'

# Butterworth low-pass: 18-hour cutoff (paper Methods)
BUTTER_CUTOFF_HOURS = 18.0
BUTTER_ORDER = 5


def expand_window(start, end, buffer_days):
    s = np.datetime64(start) - np.timedelta64(buffer_days, 'D')
    e = np.datetime64(end)   + np.timedelta64(buffer_days, 'D')
    return str(s), str(e)


def find_face_indices(grid, face, lat_min, lat_max, lon_min, lon_max):
    """Return (j_slice, i_slice, XC_sub, YC_sub) for cells in the box on this face,
    or None if face has no cells in the box.
    """
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
    return {
        'face': face,
        'j_slice': j_slice,
        'i_slice': i_slice,
        'i_g_slice': slice(i_min, min(i_max + 2, grid.sizes['i'])),
        'j_g_slice': slice(j_min, min(j_max + 2, grid.sizes['j'])),
        'XC': XC[j_slice, i_slice],
        'YC': YC[j_slice, i_slice],
        'cell_count': int(in_box.sum()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
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
        if info is None:
            print(f'  face {f}: no cells in box')
        else:
            print(f'  face {f}: {info["cell_count"]:,} cells in box  '
                  f'(j[{info["j_slice"].start}:{info["j_slice"].stop}] '
                  f'i[{info["i_slice"].start}:{info["i_slice"].stop}])')
            face_info.append(info)
    if not face_info:
        print('ERROR: no face has cells in the box.', file=sys.stderr)
        sys.exit(1)

    full_start, full_end = expand_window(WINDOW_START, WINDOW_END, BUFFER_DAYS)
    print(f'\nTime window (with {BUFFER_DAYS}-day buffer): {full_start} to {full_end}')

    # Pull SST/U/V per face, concatenating scattered points across faces
    print('\nDownloading from GCS...')
    t0 = time.time()
    src_lons, src_lats = [], []
    src_sst, src_u, src_v = [], [], []
    for info in face_info:
        f = info['face']
        print(f'  face {f}...')
        sst_sub = sst.SST.sel(time=slice(full_start, full_end)).isel(
            face=f, j=info['j_slice'], i=info['i_slice'])
        ssu_sub = ssu.U.sel(time=slice(full_start, full_end)).isel(
            face=f, j=info['j_slice'], i_g=info['i_g_slice'])
        ssv_sub = ssv.V.sel(time=slice(full_start, full_end)).isel(
            face=f, j_g=info['j_g_slice'], i=info['i_slice'])
        sst_arr = sst_sub.values
        u_arr   = ssu_sub.values
        v_arr   = ssv_sub.values

        # Cell-center U, V
        if u_arr.shape[2] == sst_arr.shape[2] + 1:
            u_arr = 0.5 * (u_arr[..., :-1] + u_arr[..., 1:])
        if v_arr.shape[1] == sst_arr.shape[1] + 1:
            v_arr = 0.5 * (v_arr[:, :-1, :] + v_arr[:, 1:, :])

        src_lons.append(info['XC'].ravel())
        src_lats.append(info['YC'].ravel())
        # Stack as (time, points) for each variable
        nt, ny, nx = sst_arr.shape
        src_sst.append(sst_arr.reshape(nt, ny * nx))
        src_u.append(u_arr.reshape(nt, ny * nx))
        src_v.append(v_arr.reshape(nt, ny * nx))
        time_axis = sst_sub.time.values

    src_lons = np.concatenate(src_lons)
    src_lats = np.concatenate(src_lats)
    src_sst  = np.concatenate(src_sst, axis=1)
    src_u    = np.concatenate(src_u,   axis=1)
    src_v    = np.concatenate(src_v,   axis=1)
    dt = time.time() - t0
    total_mb = (src_sst.nbytes + src_u.nbytes + src_v.nbytes) / 1e6
    print(f'  pulled {total_mb:.1f} MB ({src_sst.shape[0]} timesteps, '
          f'{src_lons.size:,} source cells) in {dt:.1f} s '
          f'({total_mb/dt:.1f} MB/s)')

    # Build target regular lat/lon grid
    target_lats = np.arange(LAT_MIN, LAT_MAX + DLAT/2, DLAT)
    target_lons = np.arange(LON_MIN, LON_MAX + DLON/2, DLON)
    NLAT, NLON = target_lats.size, target_lons.size
    LL, LA = np.meshgrid(target_lons, target_lats)
    print(f'\nTarget grid: {NLAT} x {NLON} (~{DLAT*111:.1f} km)')

    # pyresample: build kd-tree neighbour info ONCE, apply many times
    print('Building pyresample kd-tree...')
    t0 = time.time()
    src_def    = SwathDefinition(lons=src_lons, lats=src_lats)
    target_def = SwathDefinition(lons=LL, lats=LA)
    valid_in, valid_out, idx_array, dist = get_neighbour_info(
        src_def, target_def, radius_of_influence=5000.0, neighbours=1)
    print(f'  built in {time.time()-t0:.1f} s')

    def resample_series(data_t):
        """Resample a (time, points) array to (time, lat, lon)."""
        out = np.empty((data_t.shape[0], NLAT, NLON), dtype=np.float32)
        for t in range(data_t.shape[0]):
            out[t] = get_sample_from_neighbour_info(
                'nn', target_def.shape, data_t[t],
                valid_in, valid_out, idx_array,
                fill_value=np.nan)
        return out

    print('Resampling SST/U/V (pyresample)...')
    t0 = time.time()
    sst_grid = resample_series(src_sst)
    u_grid   = resample_series(src_u)
    v_grid   = resample_series(src_v)
    print(f'  resampled {3*src_sst.shape[0]} fields in {time.time()-t0:.1f} s')

    # 18-hr Butterworth low-pass on U, V along the time axis
    print(f'Applying {BUTTER_CUTOFF_HOURS}-hr Butterworth low-pass to U, V...')
    sample_rate_hz = 1.0 / 3600.0
    cutoff_hz = 1.0 / (BUTTER_CUTOFF_HOURS * 3600.0)
    nyq = 0.5 * sample_rate_hz
    b, a = butter(BUTTER_ORDER, cutoff_hz / nyq, btype='low')
    u_filt = filtfilt(b, a, u_grid, axis=0)
    v_filt = filtfilt(b, a, v_grid, axis=0)

    # Trim back to validity window (drop the buffer days)
    valid_mask = (time_axis >= np.datetime64(WINDOW_START)) & \
                 (time_axis <= np.datetime64(WINDOW_END))
    sst_grid = sst_grid[valid_mask].astype(np.float32)
    u_filt   = u_filt[valid_mask].astype(np.float32)
    v_filt   = v_filt[valid_mask].astype(np.float32)
    time_axis = time_axis[valid_mask]
    nt = time_axis.size
    print(f'  output: {nt} timesteps after trimming buffer')

    # log|grad SST|
    print('Computing log|grad SST|...')
    mid_lat = 0.5 * (LAT_MIN + LAT_MAX)
    dy_m = DLAT * 111e3
    dx_m = DLON * 111e3 * np.cos(np.deg2rad(mid_lat))
    log_grad_t = np.empty_like(sst_grid)
    for t in range(nt):
        gy, gx = np.gradient(sst_grid[t], dy_m, dx_m)
        mag = np.hypot(gx, gy)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_grad_t[t] = np.clip(np.log(mag), -19, 0)

    # Time axis (seconds since 2000-01-01)
    ref = np.datetime64('2000-01-01T00:00:00')
    time_sec = (time_axis - ref) / np.timedelta64(1, 's')

    out_path = os.path.join(
        OUT_DIR,
        f'llc_pangeo_bigger_{WINDOW_START[:10]}_to_{WINDOW_END[:10]}.nc')
    print(f'\nWriting {out_path}')
    with NCDataset(out_path, 'w') as nc:
        nc.createDimension('time', nt)
        nc.createDimension('lat',  NLAT)
        nc.createDimension('lon',  NLON)
        v = nc.createVariable('time', 'f8', ('time',))
        v.units = 'seconds since 2000-01-01 00:00:00'
        v[:] = time_sec
        nc.createVariable('lat', 'f4', ('lat',))[:] = target_lats
        nc.createVariable('lon', 'f4', ('lon',))[:] = target_lons
        for name, arr, attrs in [
            ('loggrad_T', log_grad_t,
             dict(long_name='log|grad SST|', normalization_range='[-19, 0]')),
            ('U', u_filt,
             dict(long_name='LLC4320 surface U, cell-centered, 18hr LP',
                  units='m/s')),
            ('V', v_filt,
             dict(long_name='LLC4320 surface V, cell-centered, 18hr LP',
                  units='m/s')),
        ]:
            var = nc.createVariable(name, 'f4', ('time', 'lat', 'lon'),
                                    zlib=True, complevel=4)
            var[:] = arr
            for k, val in attrs.items():
                setattr(var, k, val)
        nc.source = (
            f'LLC4320 faces {sorted(i["face"] for i in face_info)} '
            f'subset, resampled via pyresample kd-tree, '
            f'U/V Butterworth-filtered (cutoff {BUTTER_CUTOFF_HOURS} hr, '
            f'order {BUTTER_ORDER}, with {BUFFER_DAYS}-day buffer)')
        nc.region = f'lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}'

    print('\nResults:')
    print(f'  shape:           {sst_grid.shape}')
    print(f'  output size:     {os.path.getsize(out_path)/1e6:.1f} MB')
    print(f'  SST:             {np.nanmin(sst_grid):.2f} to {np.nanmax(sst_grid):.2f} C')
    print(f'  U (18hr LP):     {np.nanmin(u_filt):.3f} to {np.nanmax(u_filt):.3f} m/s')
    print(f'  V (18hr LP):     {np.nanmin(v_filt):.3f} to {np.nanmax(v_filt):.3f} m/s')
    print(f'  log|grad SST|:   {np.nanmin(log_grad_t):.2f} to {np.nanmax(log_grad_t):.2f}')

    del grid, sst, ssu, ssv, fs
    gc.collect()


if __name__ == '__main__':
    main()
