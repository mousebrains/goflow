"""AVISO/CMEMS L4 daily geostrophic vs c_spec=0.2 model climatology.

Loads:
  - data/aviso/aviso_l4_duacs_2023_25-42N_80-58W.nc  (1/8 deg daily, 2023)
  - data/run_paper/climatology_v2_0.2cs.npz          (model 2023 climatology cache)

Computes the same climatology metrics on AVISO (time-mean |U|, MKE, EKE,
monthly seasonal cycle in a Gulf Stream lat band) and produces a side-by-side
comparison with the model.

Caveats:
  - AVISO is geostrophic-only at 1/8 deg and merged from multiple altimeters
    -> ageostrophic / sub-mesoscale energy is missing.
  - Comparison is on time-mean and seasonal cycle, not point-wise correlation.
  - GS-band stats are evaluated over a fixed lat box [38, 41] N, lon
    [-67, -58.06] W (the model-cropped longitude window).
"""
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

AVISO = '/Users/pat/tpw/goflow/data/aviso/aviso_l4_duacs_2023_25-42N_80-58W.nc'
MODEL_NPZ = '/Users/pat/tpw/goflow/data/run_paper/climatology_v2_0.2cs.npz'
OUT_PNG = '/Users/pat/tpw/goflow/data/run_paper/aviso_vs_model_0.2cs.png'

GS_LAT = (38.0, 41.0)
# Model predictions only cover lon -63.82 to -50 W (inference x-range x898-1666).
# Within that, we cropped to lon < -57.5 W to drop the GOES east-sector mask=0
# stripe. So the model's valid lon range is roughly -63.82 to -57.51 W.
GS_LON = (-63.82, -57.51)


def load_aviso():
    ds = nc.Dataset(AVISO, 'r')
    lat = np.asarray(ds.variables['latitude'][:])
    lon = np.asarray(ds.variables['longitude'][:])
    t = np.asarray(ds.variables['time'][:])
    u = np.asarray(ds.variables['ugos'][:])
    v = np.asarray(ds.variables['vgos'][:])
    ds.close()
    epoch = datetime(1950, 1, 1)
    months = np.array([(epoch + timedelta(days=int(td))).month for td in t])
    return lat, lon, months, u, v


def aviso_stats(lat, lon, months, u, v):
    speed = np.hypot(u, v)
    mask_invalid = np.ma.getmaskarray(np.ma.masked_invalid(u))
    valid = (~mask_invalid).astype(np.float32)
    n = valid.sum(axis=0)
    cnt = np.maximum(n, 1)

    u_mean = np.where(np.isnan(u), 0, u).sum(axis=0) / cnt
    v_mean = np.where(np.isnan(v), 0, v).sum(axis=0) / cnt
    sp_mean = np.where(np.isnan(speed), 0, speed).sum(axis=0) / cnt
    u_var = np.where(np.isnan(u), 0, (u - u_mean) ** 2).sum(axis=0) / cnt
    v_var = np.where(np.isnan(v), 0, (v - v_mean) ** 2).sum(axis=0) / cnt
    bad = n == 0
    for f in [u_mean, v_mean, sp_mean, u_var, v_var]:
        f[bad] = np.nan

    MKE = 0.5 * (u_mean ** 2 + v_mean ** 2)
    EKE = 0.5 * (u_var + v_var)

    monthly_MKE = np.full(12, np.nan)
    monthly_EKE = np.full(12, np.nan)
    monthly_speed = np.full(12, np.nan)
    in_lat = (lat >= GS_LAT[0]) & (lat <= GS_LAT[1])
    in_lon = (lon >= GS_LON[0]) & (lon <= GS_LON[1])
    sel_lat = np.where(in_lat)[0]
    sel_lon = np.where(in_lon)[0]
    LL, LO = np.ix_(sel_lat, sel_lon)
    for m in range(1, 13):
        sel = months == m
        u_m = u[sel][:, LL, LO]
        v_m = v[sel][:, LL, LO]
        u_clim = np.nanmean(u_m, axis=0)
        v_clim = np.nanmean(v_m, axis=0)
        u_var_m = np.nanvar(u_m, axis=0)
        v_var_m = np.nanvar(v_m, axis=0)
        monthly_MKE[m - 1] = np.nanmean(0.5 * (u_clim ** 2 + v_clim ** 2))
        monthly_EKE[m - 1] = np.nanmean(0.5 * (u_var_m + v_var_m))
        monthly_speed[m - 1] = np.nanmean(np.hypot(u_m, v_m))

    return dict(u_mean=u_mean, v_mean=v_mean, sp_mean=sp_mean,
                MKE=MKE, EKE=EKE,
                monthly_MKE=monthly_MKE, monthly_EKE=monthly_EKE,
                monthly_speed=monthly_speed,
                lat=lat, lon=lon)


def model_stats(npz):
    A = dict(np.load(npz))
    lat, lon = A['lat'], A['lon']
    in_lat = (lat >= GS_LAT[0]) & (lat <= GS_LAT[1])
    in_lon = (lon >= GS_LON[0]) & (lon <= GS_LON[1])
    sel_lat = np.where(in_lat)[0]
    sel_lon = np.where(in_lon)[0]
    LL, LO = np.ix_(sel_lat, sel_lon)
    monthly_MKE = np.full(12, np.nan)
    monthly_EKE = np.full(12, np.nan)
    monthly_speed = np.full(12, np.nan)
    for m in range(12):
        valid = A['monthly_count'][m][LL, LO] > 0
        mke = A['monthly_MKE'][m][LL, LO]
        eke = A['monthly_EKE'][m][LL, LO]
        sp = A['monthly_speed'][m][LL, LO]
        monthly_MKE[m] = np.nanmean(np.where(valid, mke, np.nan))
        monthly_EKE[m] = np.nanmean(np.where(valid, eke, np.nan))
        monthly_speed[m] = np.nanmean(np.where(valid, sp, np.nan))
    return dict(lat=lat, lon=lon,
                u_mean=A['U_mean'], v_mean=A['V_mean'],
                sp_mean=A['speed_mean_true'],
                MKE=A['MKE'], EKE=A['EKE'],
                monthly_MKE=monthly_MKE,
                monthly_EKE=monthly_EKE,
                monthly_speed=monthly_speed)


def crop_aviso(aviso):
    lat, lon = aviso['lat'], aviso['lon']
    in_lon = (lon >= GS_LON[0]) & (lon <= GS_LON[1])
    sel = np.where(in_lon)[0]
    out = dict(aviso)
    out['lon'] = lon[sel]
    for k in ['u_mean', 'v_mean', 'sp_mean', 'MKE', 'EKE']:
        out[k] = aviso[k][:, sel]
    return out


def plot(model, aviso):
    aviso_crop = crop_aviso(aviso)
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)

    def add_map(ax, lat, lon, field, title, cmap, vmin, vmax):
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        im = ax.imshow(field, origin='lower', extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.axhline(GS_LAT[0], color='lime', lw=0.8, ls='--', alpha=0.6)
        ax.axhline(GS_LAT[1], color='lime', lw=0.8, ls='--', alpha=0.6)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04)

    add_map(fig.add_subplot(gs[0, 0]), model['lat'], model['lon'],
            model['sp_mean'], 'MODEL mean |U| (m/s)', 'turbo', 0, 0.5)
    add_map(fig.add_subplot(gs[0, 1]), aviso_crop['lat'], aviso_crop['lon'],
            aviso_crop['sp_mean'], 'AVISO mean |U_geo| (m/s)', 'turbo', 0, 0.5)
    add_map(fig.add_subplot(gs[0, 2]), model['lat'], model['lon'],
            model['MKE'], 'MODEL MKE', 'viridis', 0, 0.05)
    add_map(fig.add_subplot(gs[0, 3]), aviso_crop['lat'], aviso_crop['lon'],
            aviso_crop['MKE'], 'AVISO MKE', 'viridis', 0, 0.3)

    add_map(fig.add_subplot(gs[1, 0]), model['lat'], model['lon'],
            np.hypot(model['u_mean'], model['v_mean']),
            r'MODEL $|\langle U\rangle|$', 'turbo', 0, 0.5)
    add_map(fig.add_subplot(gs[1, 1]), aviso_crop['lat'], aviso_crop['lon'],
            np.hypot(aviso['u_mean'], aviso['v_mean']),
            r'AVISO $|\langle U_{geo}\rangle|$', 'turbo', 0, 0.5)
    add_map(fig.add_subplot(gs[1, 2]), model['lat'], model['lon'],
            model['EKE'], 'MODEL EKE', 'inferno', 0, 0.1)
    add_map(fig.add_subplot(gs[1, 3]), aviso_crop['lat'], aviso_crop['lon'],
            aviso_crop['EKE'], 'AVISO EKE', 'inferno', 0, 0.3)

    months = np.arange(1, 13)
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(months, model['monthly_speed'], '-o', lw=2, color='tab:blue', label='MODEL |U|')
    ax2 = ax.twinx()
    ax2.plot(months, aviso['monthly_speed'], '-s', lw=2, color='tab:orange', label='AVISO |U_geo|')
    ax.set_xticks(months); ax.set_xticklabels(month_labels, rotation=45)
    ax.set_ylabel('MODEL mean speed (m/s)', color='tab:blue')
    ax2.set_ylabel('AVISO mean speed (m/s)', color='tab:orange')
    ax.set_title(f'Mean speed in GS band {GS_LAT[0]}-{GS_LAT[1]}N')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(months, model['monthly_MKE'], '-o', lw=2, color='tab:green', label='MODEL')
    ax2 = ax.twinx()
    ax2.plot(months, aviso['monthly_MKE'], '-s', lw=2, color='tab:olive', label='AVISO')
    ax.set_xticks(months); ax.set_xticklabels(month_labels, rotation=45)
    ax.set_ylabel('MODEL MKE', color='tab:green')
    ax2.set_ylabel('AVISO MKE', color='tab:olive')
    ax.set_title('MKE seasonal cycle')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='y', labelcolor='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:olive')

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(months, model['monthly_EKE'], '-o', lw=2, color='tab:red', label='MODEL')
    ax2 = ax.twinx()
    ax2.plot(months, aviso['monthly_EKE'], '-s', lw=2, color='tab:purple', label='AVISO')
    ax.axvline(5, color='gray', ls=':', alpha=0.5, label='Kang+ May')
    ax.axvline(9, color='gray', ls=':', alpha=0.4, label='Kang+ Sep')
    ax.set_xticks(months); ax.set_xticklabels(month_labels, rotation=45)
    ax.set_ylabel('MODEL EKE', color='tab:red')
    ax2.set_ylabel('AVISO EKE', color='tab:purple')
    ax.set_title('EKE seasonal cycle')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    ax.legend(loc='upper left', fontsize=8)

    ax = fig.add_subplot(gs[2, 3])
    m_ratio = model['monthly_EKE'] / np.maximum(model['monthly_MKE'], 1e-9)
    a_ratio = aviso['monthly_EKE'] / np.maximum(aviso['monthly_MKE'], 1e-9)
    ax.plot(months, m_ratio, '-o', lw=2, color='tab:blue', label='MODEL')
    ax.plot(months, a_ratio, '-s', lw=2, color='tab:orange', label='AVISO')
    ax.set_xticks(months); ax.set_xticklabels(month_labels, rotation=45)
    ax.set_ylabel('EKE / MKE')
    ax.set_title('EKE/MKE seasonal cycle')
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle(f'AVISO 1/8deg vs c_spec=0.2 model — 2023 climatology in {GS_LAT[0]}-{GS_LAT[1]}N, '
                 f'{GS_LON[0]} to {GS_LON[1]}W',
                 fontsize=12)
    fig.savefig(OUT_PNG, dpi=120)
    print(f'Saved {OUT_PNG}')


def main():
    print('Loading AVISO ...')
    lat, lon, months, u, v = load_aviso()
    print(f'  AVISO shape: u={u.shape}, lat range [{lat.min():.2f},{lat.max():.2f}], '
          f'lon range [{lon.min():.2f},{lon.max():.2f}]')
    print('Computing AVISO climatology ...')
    aviso = aviso_stats(lat, lon, months, u, v)
    print('Loading model climatology ...')
    model = model_stats(MODEL_NPZ)

    print()
    print(f'=== GS band {GS_LAT[0]}-{GS_LAT[1]}N, lon {GS_LON[0]} to {GS_LON[1]}W ===')
    months_arr = np.arange(1, 13)
    print('             ', '  '.join([f'{m:>5d}' for m in months_arr]))
    print('MODEL |U|:   ', '  '.join([f'{x:.3f}' for x in model['monthly_speed']]))
    print('AVISO |U|:   ', '  '.join([f'{x:.3f}' for x in aviso['monthly_speed']]))
    print('MODEL MKE:   ', '  '.join([f'{x:.4f}' for x in model['monthly_MKE']]))
    print('AVISO MKE:   ', '  '.join([f'{x:.4f}' for x in aviso['monthly_MKE']]))
    print('MODEL EKE:   ', '  '.join([f'{x:.4f}' for x in model['monthly_EKE']]))
    print('AVISO EKE:   ', '  '.join([f'{x:.4f}' for x in aviso['monthly_EKE']]))
    print()
    print(f'MODEL EKE peak month: {months_arr[np.argmax(model["monthly_EKE"])]}, '
          f'secondary: {months_arr[np.argsort(model["monthly_EKE"])[-2]]}')
    print(f'AVISO EKE peak month: {months_arr[np.argmax(aviso["monthly_EKE"])]}, '
          f'secondary: {months_arr[np.argsort(aviso["monthly_EKE"])[-2]]}')
    print(f'MODEL MKE peak month: {months_arr[np.argmax(model["monthly_MKE"])]}')
    print(f'AVISO MKE peak month: {months_arr[np.argmax(aviso["monthly_MKE"])]}')
    print()

    plot(model, aviso)


if __name__ == '__main__':
    main()
