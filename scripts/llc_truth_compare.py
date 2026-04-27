"""LLC4320 truth vs model vs AVISO in the GS band overlap.

LLC training file covers lat 34-45 N, lon -80 to -60 W (paper file).
Model valid range covers lon -63.82 to -57.51 W.
AVISO covers full domain.

Overlap region (used for the comparison): lat 38-41 N, lon -63.82 to -60.0 W.

LLC time = 2011-09 to 2012-09 (343 days).
Model = 2023.
AVISO = 2023.

Different time periods -> we expect inter-annual variability ~10-20%, but the
time-mean magnitude should be in the same ballpark if neither has bias.

LLC reading is downsampled (every 8th frame) for speed; that is still ~1000
frames over 343 days, plenty to nail the time-mean.
"""
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta

LLC = '/Users/pat/tpw/goflow/data/paper/llcGoes_gradT_trunc.nc'
MODEL_NPZ = '/Users/pat/tpw/goflow/data/run_paper/climatology_v2_0.2cs.npz'
AVISO = '/Users/pat/tpw/goflow/data/aviso/aviso_l4_duacs_2023_25-42N_80-58W.nc'

GS_LAT = (38.0, 41.0)
GS_LON = (-63.82, -60.0)

CHUNK = 100


def llc_mean_speed():
    print(f'Loading LLC {LLC}')
    ds = nc.Dataset(LLC, 'r')
    lat = np.asarray(ds.variables['lat'][:])
    lon = np.asarray(ds.variables['lon'][:])
    nt = ds.dimensions['time'].size
    in_lat = (lat >= GS_LAT[0]) & (lat <= GS_LAT[1])
    in_lon = (lon >= GS_LON[0]) & (lon <= GS_LON[1])
    iy0, iy1 = np.where(in_lat)[0][[0, -1]]
    ix0, ix1 = np.where(in_lon)[0][[0, -1]]
    iy1 += 1; ix1 += 1
    print(f'  LLC nt={nt}, GS-band slice y[{iy0}:{iy1}] x[{ix0}:{ix1}] '
          f'(lat {lat[iy0]:.2f}-{lat[iy1-1]:.2f}, lon {lon[ix0]:.2f}-{lon[ix1-1]:.2f})')

    sum_speed = 0.0
    sum_u = 0.0
    sum_v = 0.0
    sum_sq_u = 0.0
    sum_sq_v = 0.0
    n_frames = 0

    monthly_speed = np.zeros(12)
    monthly_u_sum = np.zeros(12)
    monthly_v_sum = np.zeros(12)
    monthly_u_sq = np.zeros(12)
    monthly_v_sq = np.zeros(12)
    monthly_n = np.zeros(12, dtype=np.int64)

    epoch = datetime(2011, 9, 13)
    stride = 8
    for c0 in range(0, nt, CHUNK):
        c1 = min(nt, c0 + CHUNK)
        idx = list(range(c0, c1, stride))
        if not idx:
            continue
        u = np.asarray(ds.variables['U'][idx, iy0:iy1, ix0:ix1])
        v = np.asarray(ds.variables['V'][idx, iy0:iy1, ix0:ix1])
        sp = np.hypot(u, v)
        sum_speed += sp.sum()
        sum_u += u.sum()
        sum_v += v.sum()
        sum_sq_u += (u * u).sum()
        sum_sq_v += (v * v).sum()
        n_frames += sp.size

        for k, ii in enumerate(idx):
            t_real = epoch + timedelta(hours=ii)
            m = t_real.month - 1
            u_k = u[k]; v_k = v[k]
            monthly_speed[m] += np.hypot(u_k, v_k).sum()
            monthly_u_sum[m] += u_k.sum()
            monthly_v_sum[m] += v_k.sum()
            monthly_u_sq[m] += (u_k * u_k).sum()
            monthly_v_sq[m] += (v_k * v_k).sum()
            monthly_n[m] += u_k.size

        if c0 % 800 == 0:
            print(f'    {c0}/{nt} frames')
    ds.close()

    spm = sum_speed / n_frames
    u_mean = sum_u / n_frames
    v_mean = sum_v / n_frames
    u_var = sum_sq_u / n_frames - u_mean ** 2
    v_var = sum_sq_v / n_frames - v_mean ** 2
    MKE = 0.5 * (u_mean ** 2 + v_mean ** 2)
    EKE = 0.5 * (u_var + v_var)

    monthly_sp = monthly_speed / np.maximum(monthly_n, 1)
    monthly_u = monthly_u_sum / np.maximum(monthly_n, 1)
    monthly_v = monthly_v_sum / np.maximum(monthly_n, 1)
    monthly_uvar = monthly_u_sq / np.maximum(monthly_n, 1) - monthly_u ** 2
    monthly_vvar = monthly_v_sq / np.maximum(monthly_n, 1) - monthly_v ** 2
    monthly_MKE = 0.5 * (monthly_u ** 2 + monthly_v ** 2)
    monthly_EKE = 0.5 * (monthly_uvar + monthly_vvar)

    return dict(speed=spm, u_mean=u_mean, v_mean=v_mean, MKE=MKE, EKE=EKE,
                monthly_speed=monthly_sp, monthly_MKE=monthly_MKE,
                monthly_EKE=monthly_EKE)


def model_mean_speed():
    A = dict(np.load(MODEL_NPZ))
    lat, lon = A['lat'], A['lon']
    in_lat = (lat >= GS_LAT[0]) & (lat <= GS_LAT[1])
    in_lon = (lon >= GS_LON[0]) & (lon <= GS_LON[1])
    LL, LO = np.ix_(np.where(in_lat)[0], np.where(in_lon)[0])
    sp_mean = np.nanmean(A['speed_mean_true'][LL, LO])
    u_mean = np.nanmean(A['U_mean'][LL, LO])
    v_mean = np.nanmean(A['V_mean'][LL, LO])
    MKE = np.nanmean(A['MKE'][LL, LO])
    EKE = np.nanmean(A['EKE'][LL, LO])
    monthly_speed = np.zeros(12)
    monthly_MKE = np.zeros(12)
    monthly_EKE = np.zeros(12)
    for m in range(12):
        valid = A['monthly_count'][m][LL, LO] > 0
        monthly_speed[m] = np.nanmean(np.where(valid, A['monthly_speed'][m][LL, LO], np.nan))
        monthly_MKE[m] = np.nanmean(np.where(valid, A['monthly_MKE'][m][LL, LO], np.nan))
        monthly_EKE[m] = np.nanmean(np.where(valid, A['monthly_EKE'][m][LL, LO], np.nan))
    return dict(speed=sp_mean, u_mean=u_mean, v_mean=v_mean, MKE=MKE, EKE=EKE,
                monthly_speed=monthly_speed, monthly_MKE=monthly_MKE,
                monthly_EKE=monthly_EKE)


def aviso_mean_speed():
    ds = nc.Dataset(AVISO, 'r')
    lat = np.asarray(ds.variables['latitude'][:])
    lon = np.asarray(ds.variables['longitude'][:])
    t = np.asarray(ds.variables['time'][:])
    u = np.asarray(ds.variables['ugos'][:])
    v = np.asarray(ds.variables['vgos'][:])
    ds.close()
    in_lat = (lat >= GS_LAT[0]) & (lat <= GS_LAT[1])
    in_lon = (lon >= GS_LON[0]) & (lon <= GS_LON[1])
    LL, LO = np.ix_(np.where(in_lat)[0], np.where(in_lon)[0])
    u = u[:, LL, LO]
    v = v[:, LL, LO]
    sp = np.hypot(u, v)
    sp_mean = np.nanmean(sp)
    u_mean = np.nanmean(u)
    v_mean = np.nanmean(v)
    MKE = 0.5 * (u_mean ** 2 + v_mean ** 2)
    EKE = 0.5 * (np.nanvar(u, axis=0).mean() + np.nanvar(v, axis=0).mean())

    epoch = datetime(1950, 1, 1)
    months = np.array([(epoch + timedelta(days=int(td))).month - 1 for td in t])
    monthly_speed = np.zeros(12)
    monthly_MKE = np.zeros(12)
    monthly_EKE = np.zeros(12)
    for m in range(12):
        sel = months == m
        u_m = u[sel]; v_m = v[sel]
        if len(u_m) == 0:
            continue
        monthly_speed[m] = np.nanmean(np.hypot(u_m, v_m))
        u_clim = np.nanmean(u_m); v_clim = np.nanmean(v_m)
        monthly_MKE[m] = 0.5 * (u_clim ** 2 + v_clim ** 2)
        monthly_EKE[m] = 0.5 * (np.nanvar(u_m) + np.nanvar(v_m))

    return dict(speed=sp_mean, u_mean=u_mean, v_mean=v_mean, MKE=MKE, EKE=EKE,
                monthly_speed=monthly_speed, monthly_MKE=monthly_MKE,
                monthly_EKE=monthly_EKE)


def main():
    print(f'\n=== Region: lat {GS_LAT[0]}-{GS_LAT[1]} N, lon {GS_LON[0]} to {GS_LON[1]} W ===\n')

    print('Computing AVISO ...')
    a = aviso_mean_speed()
    print('Computing MODEL ...')
    m = model_mean_speed()
    print('Computing LLC ...')
    l = llc_mean_speed()

    print()
    print('Time-mean stats in GS overlap region:')
    print(f'  source        | <|U|>  | <U>      <V>     | MKE      EKE      | EKE/MKE')
    for name, d in [('LLC truth (paper)', l), ('MODEL (c_spec=0.2)', m), ('AVISO L4 1/8d', a)]:
        ratio = d['EKE'] / max(d['MKE'], 1e-9)
        print(f'  {name:18s} | {d["speed"]:.3f}  | {d["u_mean"]:+.3f}   {d["v_mean"]:+.3f}  | '
              f'{d["MKE"]:.4f}  {d["EKE"]:.4f}  | {ratio:.2f}')

    print()
    print('Monthly mean speed:')
    months_lbl = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    print('             ', '  '.join([f'{x:>5s}' for x in months_lbl]))
    print('LLC truth:   ', '  '.join([f'{x:.3f}' for x in l['monthly_speed']]))
    print('MODEL:       ', '  '.join([f'{x:.3f}' for x in m['monthly_speed']]))
    print('AVISO:       ', '  '.join([f'{x:.3f}' for x in a['monthly_speed']]))
    print()
    print('Monthly MKE:')
    print('LLC truth:   ', '  '.join([f'{x:.4f}' for x in l['monthly_MKE']]))
    print('MODEL:       ', '  '.join([f'{x:.4f}' for x in m['monthly_MKE']]))
    print('AVISO:       ', '  '.join([f'{x:.4f}' for x in a['monthly_MKE']]))
    print()
    print('Monthly EKE:')
    print('LLC truth:   ', '  '.join([f'{x:.4f}' for x in l['monthly_EKE']]))
    print('MODEL:       ', '  '.join([f'{x:.4f}' for x in m['monthly_EKE']]))
    print('AVISO:       ', '  '.join([f'{x:.4f}' for x in a['monthly_EKE']]))


if __name__ == '__main__':
    main()
