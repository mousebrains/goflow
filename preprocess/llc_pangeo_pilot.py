"""LLC4320 Pangeo zarr -> NetCDF for GOFLOW training, 1-day, 1-face, 1-tile pilot.

Reads 24 hourly snapshots of SST, U, V from the requester-pays Pangeo zarrs
on GCS (pangeo-ecco-llc4320) for face 10 (Atlantic), restricted to a 5.3 deg
box inside your target region (25-42N, 80-50W). Interpolates U/V from
staggered to cell-centered grid, resamples the curvilinear LLC face to a
regular lat/lon grid, computes log|grad SST|, writes NetCDF in the schema
dataSST.SSTDataset expects.

Run AFTER GCP setup:
    python preprocess/llc_pangeo_pilot.py YOUR_PROJECT_ID

Output: data/llc_pangeo_pilot_<date>.nc

Cost: ~5.4 GB GCS read (24 timesteps x 3 vars at 1 face x 4320^2 chunk granularity)
       ~$0.65 in egress on requester-pays
"""
import sys
import os
import gc
import time
import numpy as np
import xarray as xr
import gcsfs
from scipy.interpolate import griddata
from netCDF4 import Dataset as NCDataset
from tqdm import tqdm


if len(sys.argv) < 2:
    print('Usage: python preprocess/llc_pangeo_pilot.py YOUR_PROJECT_ID', file=sys.stderr)
    sys.exit(1)

PROJECT = sys.argv[1]
# Make google-auth see our project regardless of shell env
os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT

# Pilot tile: 5.3 deg box matching paper convention, inside user's target region
LAT_MIN, LAT_MAX = 35.0, 40.3
LON_MIN, LON_MAX = -70.0, -64.7

# Output regular-grid resolution (paper convention: 256x256 ~= 2 km)
NLAT, NLON = 256, 256

# Time slice in LLC4320's coverage window (Sep 2011 - Sep 2012)
TIME_START = '2012-04-13T00:00:00'
TIME_END   = '2012-04-13T23:00:00'

# LLC face containing our Atlantic box (verified via probe_gcp_llc4320.py)
FACE = 10

OUT_DIR = 'data'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fs = gcsfs.GCSFileSystem(token='google_default', project=PROJECT, requester_pays=True)

    print('Opening LLC4320 zarrs...')
    grid = xr.open_zarr(fs.get_mapper('pangeo-ecco-llc4320/grid'), consolidated=True)
    sst  = xr.open_zarr(fs.get_mapper('pangeo-ecco-llc4320/sst'),  consolidated=True)
    ssu  = xr.open_zarr(fs.get_mapper('pangeo-ecco-llc4320/ssu'),  consolidated=True)
    ssv  = xr.open_zarr(fs.get_mapper('pangeo-ecco-llc4320/ssv'),  consolidated=True)

    print(f'Loading face {FACE} grid coords (XC, YC)...')
    XC = grid.XC.isel(face=FACE).load()
    YC = grid.YC.isel(face=FACE).load()

    in_box = (
        (YC.values >= LAT_MIN) & (YC.values <= LAT_MAX) &
        (XC.values >= LON_MIN) & (XC.values <= LON_MAX)
    )
    if not in_box.any():
        print(f'ERROR: no cells in face {FACE} fall within '
              f'lat[{LAT_MIN},{LAT_MAX}] x lon[{LON_MIN},{LON_MAX}]', file=sys.stderr)
        sys.exit(1)

    j_inds, i_inds = np.where(in_box)
    j_min, j_max = int(j_inds.min()), int(j_inds.max())
    i_min, i_max = int(i_inds.min()), int(i_inds.max())
    print(f'  bbox in face coords: j[{j_min}:{j_max+1}] i[{i_min}:{i_max+1}]')
    print(f'  source size:         {j_max-j_min+1} x {i_max-i_min+1} cells')

    j_slice = slice(j_min, j_max + 1)
    i_slice = slice(i_min, i_max + 1)
    # Staggered grids: pull one extra cell so we can average to centers
    NJ_MAX = grid.sizes['j']  # 4320
    NI_MAX = grid.sizes['i']
    i_slice_g = slice(i_min, min(i_max + 2, NI_MAX))
    j_slice_g = slice(j_min, min(j_max + 2, NJ_MAX))

    XC_sub = XC.isel(j=j_slice, i=i_slice).values
    YC_sub = YC.isel(j=j_slice, i=i_slice).values

    print(f'\nDownloading from GCS for {TIME_START} to {TIME_END}...')
    t0 = time.time()
    sst_sub = sst.SST.sel(time=slice(TIME_START, TIME_END)).isel(face=FACE, j=j_slice, i=i_slice)
    ssu_sub = ssu.U.sel(time=slice(TIME_START, TIME_END)).isel(face=FACE, j=j_slice, i_g=i_slice_g)
    ssv_sub = ssv.V.sel(time=slice(TIME_START, TIME_END)).isel(face=FACE, j_g=j_slice_g, i=i_slice)
    sst_arr = sst_sub.values
    u_arr   = ssu_sub.values
    v_arr   = ssv_sub.values
    dt = time.time() - t0
    total_mb = (sst_arr.nbytes + u_arr.nbytes + v_arr.nbytes) / 1e6
    print(f'  pulled {total_mb:.1f} MB in {dt:.1f} s ({total_mb/dt:.1f} MB/s effective)')
    print(f'  SST shape: {sst_arr.shape}, U shape: {u_arr.shape}, V shape: {v_arr.shape}')

    # Interpolate U from i_g (west-face) to i (cell center)
    if u_arr.shape[2] == sst_arr.shape[2] + 1:
        u_centered = 0.5 * (u_arr[..., :-1] + u_arr[..., 1:])
    else:
        u_centered = u_arr  # at edge of face; lose 1 column
    # Interpolate V from j_g (south-face) to j (cell center)
    if v_arr.shape[1] == sst_arr.shape[1] + 1:
        v_centered = 0.5 * (v_arr[:, :-1, :] + v_arr[:, 1:, :])
    else:
        v_centered = v_arr

    # Resample curvilinear face -> regular lat/lon
    target_lats = np.linspace(LAT_MIN, LAT_MAX, NLAT)
    target_lons = np.linspace(LON_MIN, LON_MAX, NLON)
    LL, LA = np.meshgrid(target_lons, target_lats)
    pts        = np.column_stack([XC_sub.ravel(), YC_sub.ravel()])
    target_pts = np.column_stack([LL.ravel(),     LA.ravel()])

    nt = sst_arr.shape[0]
    sst_grid = np.empty((nt, NLAT, NLON), dtype=np.float32)
    u_grid   = np.empty((nt, NLAT, NLON), dtype=np.float32)
    v_grid   = np.empty((nt, NLAT, NLON), dtype=np.float32)
    print('Resampling curvilinear -> regular lat/lon...')
    for t in tqdm(range(nt)):
        sst_grid[t] = griddata(pts, sst_arr[t].ravel(),    target_pts, method='linear').reshape(NLAT, NLON)
        u_grid[t]   = griddata(pts, u_centered[t].ravel(), target_pts, method='linear').reshape(NLAT, NLON)
        v_grid[t]   = griddata(pts, v_centered[t].ravel(), target_pts, method='linear').reshape(NLAT, NLON)

    # log|grad SST|
    print('Computing log|grad SST|...')
    mid_lat = 0.5 * (LAT_MIN + LAT_MAX)
    dy_m = (target_lats[1] - target_lats[0]) * 111e3
    dx_m = (target_lons[1] - target_lons[0]) * 111e3 * np.cos(np.deg2rad(mid_lat))
    log_grad_t = np.empty_like(sst_grid)
    for t in range(nt):
        gy, gx = np.gradient(sst_grid[t], dy_m, dx_m)
        mag = np.hypot(gx, gy)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_mag = np.log(mag)
        log_grad_t[t] = np.clip(log_mag, -19, 0)

    # Time axis (seconds since 2000-01-01 to match GOES pilot convention)
    ref = np.datetime64('2000-01-01T00:00:00')
    time_sec = (sst_sub.time.values - ref) / np.timedelta64(1, 's')

    out = os.path.join(OUT_DIR, f'llc_pangeo_pilot_{TIME_START[:10]}.nc')
    print(f'\nWriting {out}')
    with NCDataset(out, 'w') as nc:
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
            ('U', u_grid,
             dict(long_name='LLC4320 surface zonal velocity, cell-centered', units='m/s')),
            ('V', v_grid,
             dict(long_name='LLC4320 surface meridional velocity, cell-centered', units='m/s')),
        ]:
            var = nc.createVariable(name, 'f4', ('time', 'lat', 'lon'),
                                    zlib=True, complevel=4)
            var[:] = arr
            for k, val in attrs.items():
                setattr(var, k, val)
        nc.source = (f'LLC4320 face {FACE}, {TIME_START} to {TIME_END}, '
                     f'resampled to {NLAT}x{NLON} regular lat/lon')
        nc.region = f'lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}'

    print('\nResults:')
    print(f'  shape:           {sst_grid.shape}')
    print(f'  output size:     {os.path.getsize(out)/1e6:.1f} MB')
    print(f'  SST:             {np.nanmin(sst_grid):.2f} to {np.nanmax(sst_grid):.2f} C')
    print(f'  U:               {np.nanmin(u_grid):.3f} to {np.nanmax(u_grid):.3f} m/s')
    print(f'  V:               {np.nanmin(v_grid):.3f} to {np.nanmax(v_grid):.3f} m/s')
    print(f'  log|grad SST|:   {np.nanmin(log_grad_t):.2f} to {np.nanmax(log_grad_t):.2f}')

    # Tear down gcsfs/aiohttp before interpreter exit
    del grid, sst, ssu, ssv, fs
    gc.collect()


if __name__ == '__main__':
    main()
