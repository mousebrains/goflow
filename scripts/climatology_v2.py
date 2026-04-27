"""Corrected climatology for c_spec=0.2 inference: cropped to valid west half,
with proper mean speed and EKE accumulation, monthly seasonal binning, and
cross-front profiles defined relative to the climatological thermal front.

Fixes from v1:
  - Cropped to lon < -57.5 W (where GOES-16 RadC sector has data)
  - Mean speed = <|U|>, accumulated directly (not |<U>|)
  - MKE and EKE computed in standard form
  - Monthly seasonal cycle (12 bins)
  - Cross-front profile: trace climatological log_gradT ridge by longitude,
    sample U, V at lat offsets, average across lons -> across-front profile

Outputs:
  data/run_paper/climatology_v2_0.2cs.png
  data/run_paper/climatology_v2_0.2cs.npz
"""
import os
import time
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

SOUTH = '/Users/pat/tpw/goflow/data/run_paper/preds_lgt_unet16_1_3_0.2cs_goes_2023_full.nc'
NORTH = '/Users/pat/tpw/goflow/data/run_paper/preds_lgt_unet16_1_3_0.2cs_goes_2023_full_y512-944_x898-1666.nc'
OUT_PNG = '/Users/pat/tpw/goflow/data/run_paper/climatology_v2_0.2cs.png'
OUT_NPZ = '/Users/pat/tpw/goflow/data/run_paper/climatology_v2_0.2cs.npz'

# Valid lon cutoff: GOES-16 RadC sector -> mask=0 east of about -57.5 W.
LON_VALID_MAX = -57.5
CHUNK = 100

# 2000-01-01 00:00:00 UTC epoch (predictions inherit this from goes_yearly.py;
# our fix in goes_yearly.py uses J2000 = 2000-01-01 12:00:00 but the prediction
# files were generated before that fix and use the buggy ns-since-1970-as-float
# encoding). Use the time-axis ordering only, infer month from frame index.
def month_of_frame(t_idx, frames_per_day=24):
    """Map hourly frame index to month-of-year [0..11], assuming Jan 1 at idx 0."""
    day_of_year = t_idx // frames_per_day  # 0..364
    # 2023 is non-leap. Cumulative days at start of each month:
    cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    for m in range(12):
        if cum[m] <= day_of_year < cum[m + 1]:
            return m
    return 11


def stream_file(path, lon_max):
    """Stream a prediction file, accumulating per-pixel U, V, speed, and
    log_gradT statistics, plus monthly bins. Cropped to lon <= lon_max."""
    ds = nc.Dataset(path, 'r')
    lat = np.asarray(ds.variables['lat'][:])
    lon = np.asarray(ds.variables['lon'][:])
    nt = ds.dimensions['time'].size

    keep_mask = lon <= lon_max
    lon_kept = lon[keep_mask]
    nx = lon_kept.size
    ny = lat.size
    print(f'  shape: nt={nt}, ny={ny}, nx_kept={nx} (cropped from {lon.size})')

    # Per-pixel accumulators (full-year)
    sum_U = np.zeros((ny, nx), dtype=np.float64)
    sum_V = np.zeros((ny, nx), dtype=np.float64)
    sum_sq_U = np.zeros((ny, nx), dtype=np.float64)
    sum_sq_V = np.zeros((ny, nx), dtype=np.float64)
    sum_speed = np.zeros((ny, nx), dtype=np.float64)
    sum_sq_speed = np.zeros((ny, nx), dtype=np.float64)
    sum_lgt = np.zeros((ny, nx), dtype=np.float64)
    count = np.zeros((ny, nx), dtype=np.int64)

    # Monthly bins
    monthly_speed_sum = np.zeros((12, ny, nx), dtype=np.float64)
    monthly_count = np.zeros((12, ny, nx), dtype=np.int64)
    # MKE/EKE: accumulate per-month sum_U, sum_V, sum_U^2, sum_V^2
    monthly_sum_U = np.zeros((12, ny, nx), dtype=np.float64)
    monthly_sum_V = np.zeros((12, ny, nx), dtype=np.float64)
    monthly_sum_sq_U = np.zeros((12, ny, nx), dtype=np.float64)
    monthly_sum_sq_V = np.zeros((12, ny, nx), dtype=np.float64)

    t0 = time.time()
    for c0 in range(0, nt, CHUNK):
        c1 = min(nt, c0 + CHUNK)
        u = np.asarray(ds.variables['U'][c0:c1, :, keep_mask])
        v = np.asarray(ds.variables['V'][c0:c1, :, keep_mask])
        lgt = np.asarray(ds.variables['loggrad_BT'][c0:c1, :, keep_mask])
        valid = (u != 0) | (v != 0)
        speed = np.hypot(u, v)

        # Full-year accumulators
        sum_U += np.where(valid, u, 0).sum(axis=0)
        sum_V += np.where(valid, v, 0).sum(axis=0)
        sum_sq_U += np.where(valid, u * u, 0).sum(axis=0)
        sum_sq_V += np.where(valid, v * v, 0).sum(axis=0)
        sum_speed += np.where(valid, speed, 0).sum(axis=0)
        sum_sq_speed += np.where(valid, speed * speed, 0).sum(axis=0)
        sum_lgt += np.where(valid, lgt, 0).sum(axis=0)
        count += valid.sum(axis=0)

        # Monthly bins
        for i in range(c1 - c0):
            m = month_of_frame(c0 + i)
            v_i = valid[i]
            monthly_speed_sum[m] += np.where(v_i, speed[i], 0)
            monthly_count[m] += v_i
            monthly_sum_U[m] += np.where(v_i, u[i], 0)
            monthly_sum_V[m] += np.where(v_i, v[i], 0)
            monthly_sum_sq_U[m] += np.where(v_i, u[i] * u[i], 0)
            monthly_sum_sq_V[m] += np.where(v_i, v[i] * v[i], 0)

        if c0 % 1000 == 0:
            elapsed = time.time() - t0
            rate = c1 / elapsed if elapsed else 0
            eta = (nt - c1) / max(rate, 1) / 60
            print(f'    {c1}/{nt} fr, {rate:.0f} fr/s, ETA {eta:.1f} min')

    ds.close()
    cnt = np.maximum(count, 1)
    out = {
        'lat': lat, 'lon': lon_kept,
        'U_mean': sum_U / cnt,
        'V_mean': sum_V / cnt,
        'speed_mean_true': sum_speed / cnt,         # <|U|>
        'speed_var': np.maximum(sum_sq_speed / cnt - (sum_speed / cnt) ** 2, 0),
        'EKE': 0.5 * (np.maximum(sum_sq_U / cnt - (sum_U / cnt) ** 2, 0)
                      + np.maximum(sum_sq_V / cnt - (sum_V / cnt) ** 2, 0)),
        'MKE': 0.5 * ((sum_U / cnt) ** 2 + (sum_V / cnt) ** 2),
        'lgt_mean': sum_lgt / cnt,
        'count': count,
    }
    bad = count == 0
    for k in ('U_mean', 'V_mean', 'speed_mean_true', 'speed_var',
              'EKE', 'MKE', 'lgt_mean'):
        out[k][bad] = np.nan

    # Monthly-mean speed & monthly EKE
    m_cnt = np.maximum(monthly_count, 1)
    out['monthly_speed'] = monthly_speed_sum / m_cnt
    out['monthly_count'] = monthly_count
    monthly_U_mean = monthly_sum_U / m_cnt
    monthly_V_mean = monthly_sum_V / m_cnt
    monthly_var_U = np.maximum(monthly_sum_sq_U / m_cnt - monthly_U_mean ** 2, 0)
    monthly_var_V = np.maximum(monthly_sum_sq_V / m_cnt - monthly_V_mean ** 2, 0)
    out['monthly_EKE'] = 0.5 * (monthly_var_U + monthly_var_V)
    out['monthly_MKE'] = 0.5 * (monthly_U_mean ** 2 + monthly_V_mean ** 2)
    return out


def stitch_strips(s, n):
    """Concatenate south + north strip outputs along latitude."""
    out = {'lat': np.concatenate([s['lat'], n['lat']]),
           'lon': s['lon']}
    for key in ('U_mean', 'V_mean', 'speed_mean_true', 'speed_var',
                'EKE', 'MKE', 'lgt_mean'):
        out[key] = np.concatenate([s[key], n[key]], axis=0)
    out['count'] = np.concatenate([s['count'], n['count']], axis=0)
    out['monthly_speed'] = np.concatenate([s['monthly_speed'], n['monthly_speed']], axis=1)
    out['monthly_EKE'] = np.concatenate([s['monthly_EKE'], n['monthly_EKE']], axis=1)
    out['monthly_MKE'] = np.concatenate([s['monthly_MKE'], n['monthly_MKE']], axis=1)
    out['monthly_count'] = np.concatenate([s['monthly_count'], n['monthly_count']], axis=1)
    return out


def front_trajectory(lgt_mean, lat, lat_search=(33.0, 42.0)):
    """Find the lat of climatological log_gradT max at each lon (smoothed),
    restricted to a plausible GS-axis search range to suppress spurious peaks."""
    in_range = (lat >= lat_search[0]) & (lat <= lat_search[1])
    if not in_range.any():
        raise ValueError(f'No latitudes in search range {lat_search}')
    # argmax in restricted band, then map back to full-array index
    band_idx = np.where(in_range)[0]
    band = lgt_mean[band_idx, :]
    front_in_band = np.nanargmax(band, axis=0)
    front_lat_idx = band_idx[front_in_band]
    front_lat = lat[front_lat_idx]
    # Smooth across lon
    win = 11
    kernel = np.ones(win) / win
    front_lat_smooth = np.convolve(front_lat, kernel, mode='same')
    return front_lat_idx, front_lat_smooth


def cross_front_profile(field, front_lat_idx, lat, offsets_deg):
    """Sample field at lat offsets from a per-lon front position. Returns
    array of shape (n_offsets,) -- lon-averaged across-front profile."""
    ny, nx = field.shape
    dlat = lat[1] - lat[0]
    profiles = np.full((len(offsets_deg), nx), np.nan)
    for i_off, off in enumerate(offsets_deg):
        d_idx = int(round(off / dlat))
        for ix in range(nx):
            target = front_lat_idx[ix] + d_idx
            if 0 <= target < ny:
                profiles[i_off, ix] = field[target, ix]
    return np.nanmean(profiles, axis=1), profiles


def main():
    print('Streaming SOUTH strip (cropped)...')
    s = stream_file(SOUTH, LON_VALID_MAX)
    print('Streaming NORTH strip (cropped)...')
    n = stream_file(NORTH, LON_VALID_MAX)
    A = stitch_strips(s, n)

    np.savez_compressed(OUT_NPZ, **A)
    print(f'Saved {OUT_NPZ}')

    # Climatological front
    front_idx, front_lat_sm = front_trajectory(A['lgt_mean'], A['lat'])
    print(f'Front lat (smoothed): {front_lat_sm.min():.2f} to {front_lat_sm.max():.2f} N')

    # Cross-front profile of mean speed (with offsets +/-2 deg in steps of 0.1 deg)
    offsets = np.arange(-2, 2.05, 0.1)
    cf_speed_mean, _ = cross_front_profile(A['speed_mean_true'],
                                            front_idx, A['lat'], offsets)
    cf_eke, _ = cross_front_profile(A['EKE'], front_idx, A['lat'], offsets)
    cf_mke, _ = cross_front_profile(A['MKE'], front_idx, A['lat'], offsets)

    # Plot
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    extent = [A['lon'].min(), A['lon'].max(),
              A['lat'].min(), A['lat'].max()]

    def add_map(ax, field, title, cmap, vmin=None, vmax=None, overlay_front=False):
        if vmin is None:
            vmin = np.nanpercentile(field, 1)
        if vmax is None:
            vmax = np.nanpercentile(field, 99)
        im = ax.imshow(field, origin='lower', extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        if overlay_front:
            ax.plot(A['lon'], front_lat_sm, color='cyan', lw=1.5,
                    alpha=0.8, label='climatological front')
            ax.legend(loc='lower right', fontsize=8)
        ax.axhline(34.21, color='lime', lw=1.0, ls='--', alpha=0.6)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.04)

    add_map(fig.add_subplot(gs[0, 0]), A['speed_mean_true'],
            r'time-mean $\langle|U|\rangle$ (m/s)', 'turbo', 0, 0.5,
            overlay_front=True)
    add_map(fig.add_subplot(gs[0, 1]), A['MKE'],
            'MKE (m^2/s^2) -- 0.5(<U>^2+<V>^2)', 'viridis', 0, 0.05)
    add_map(fig.add_subplot(gs[0, 2]), A['EKE'],
            "EKE (m^2/s^2) -- 0.5(var U + var V)", 'inferno', 0, 0.1)
    add_map(fig.add_subplot(gs[0, 3]), np.sqrt(A['speed_var']),
            'std(|U|) (m/s)', 'plasma', 0, 0.3)

    # Climatological front position + log_gradT mean
    add_map(fig.add_subplot(gs[1, 0]), A['lgt_mean'],
            'time-mean log|grad BT| (normalized)', 'gray_r', 0.7, 0.95,
            overlay_front=True)

    # Across-front profile
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(offsets, cf_speed_mean, '-o', lw=2, ms=4, color='tab:blue', label=r'$\langle|U|\rangle$')
    ax.set_xlabel('offset from front (deg lat, +N)')
    ax.set_ylabel('mean speed (m/s)')
    ax.axvline(0, color='cyan', ls='--', alpha=0.6, label='front')
    ax.set_title('Across-front profile: mean speed')
    ax.grid(alpha=0.3); ax.legend()

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(offsets, cf_eke, '-o', lw=2, ms=4, color='tab:red', label='EKE')
    ax.plot(offsets, cf_mke, '-s', lw=2, ms=4, color='tab:green', label='MKE')
    ax.set_xlabel('offset from front (deg lat, +N)')
    ax.set_ylabel('KE (m^2/s^2)')
    ax.axvline(0, color='cyan', ls='--', alpha=0.6, label='front')
    ax.set_title('Across-front profile: MKE and EKE')
    ax.grid(alpha=0.3); ax.legend()

    # Total KE = MKE + EKE (across-front)
    ax = fig.add_subplot(gs[1, 3])
    ratio = cf_eke / np.maximum(cf_mke, 1e-6)
    ax.plot(offsets, ratio, '-o', lw=2, ms=4, color='tab:purple')
    ax.set_xlabel('offset from front (deg lat, +N)')
    ax.set_ylabel('EKE / MKE')
    ax.axvline(0, color='cyan', ls='--', alpha=0.6)
    ax.axhline(1, color='gray', ls=':', alpha=0.5)
    ax.set_title('Across-front profile: EKE/MKE ratio')
    ax.grid(alpha=0.3)

    # Seasonal cycle
    months = np.arange(1, 13)
    ax = fig.add_subplot(gs[2, 0:2])
    # Spatial-mean monthly MKE and EKE in the Stream region (within +/- 0.5 deg of front)
    # Sample only valid pixels in that band
    front_band_mask = np.zeros_like(A['lgt_mean'], dtype=bool)
    dlat = A['lat'][1] - A['lat'][0]
    band_idx = int(round(0.5 / dlat))
    for ix in range(len(A['lon'])):
        i0 = max(0, front_idx[ix] - band_idx)
        i1 = min(len(A['lat']), front_idx[ix] + band_idx + 1)
        front_band_mask[i0:i1, ix] = True
    monthly_mke_band = []
    monthly_eke_band = []
    monthly_speed_band = []
    for m in range(12):
        valid_band = front_band_mask & (A['monthly_count'][m] > 0)
        monthly_mke_band.append(np.nanmean(A['monthly_MKE'][m][valid_band]))
        monthly_eke_band.append(np.nanmean(A['monthly_EKE'][m][valid_band]))
        monthly_speed_band.append(np.nanmean(A['monthly_speed'][m][valid_band]))
    ax.plot(months, monthly_mke_band, '-o', lw=2, color='tab:green', label='MKE in front band (+/-0.5 deg)')
    ax.plot(months, monthly_eke_band, '-s', lw=2, color='tab:red', label='EKE in front band (+/-0.5 deg)')
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylabel('KE (m^2/s^2)')
    ax.set_title('Seasonal cycle: MKE and EKE within 0.5 deg of climatological front')
    ax.grid(alpha=0.3); ax.legend()

    ax = fig.add_subplot(gs[2, 2:])
    ax.plot(months, monthly_speed_band, '-o', lw=2, color='tab:blue',
            label='mean |U| in front band')
    # Compare to two regions: south of front (Sargasso) and north (slope)
    south_mask = A['lat'][:, None].repeat(len(A['lon']), axis=1) < A['lat'][front_idx][None, :] - 1.5
    north_mask = A['lat'][:, None].repeat(len(A['lon']), axis=1) > A['lat'][front_idx][None, :] + 1.5
    monthly_s = []
    monthly_n = []
    for m in range(12):
        v_s = south_mask & (A['monthly_count'][m] > 0)
        v_n = north_mask & (A['monthly_count'][m] > 0)
        monthly_s.append(np.nanmean(A['monthly_speed'][m][v_s]) if v_s.any() else np.nan)
        monthly_n.append(np.nanmean(A['monthly_speed'][m][v_n]) if v_n.any() else np.nan)
    ax.plot(months, monthly_s, '-^', lw=1.5, color='tab:cyan',
            label='south of front (Sargasso)')
    ax.plot(months, monthly_n, '-v', lw=1.5, color='tab:orange',
            label='north of front (slope/extension)')
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylabel('mean speed (m/s)')
    ax.set_title('Seasonal cycle: mean speed by region')
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle('c_spec=0.2 climatology v2 -- cropped to lon < -57.5 W (valid GOES-16 RadC) '
                 '-- with cross-front + seasonal analysis', fontsize=13)
    fig.savefig(OUT_PNG, dpi=120)
    print(f'Saved {OUT_PNG}')

    # Print summary
    print()
    print('=== Summary ===')
    print(f'Climatological front: lat {front_lat_sm.min():.2f} to {front_lat_sm.max():.2f} N (smoothed)')
    print(f'Front-band mean speed peak month: {months[np.argmax(monthly_speed_band)]} (value {max(monthly_speed_band):.4f} m/s)')
    print(f'Front-band EKE peak month: {months[np.argmax(monthly_eke_band)]}')
    print(f'Front-band MKE peak month: {months[np.argmax(monthly_mke_band)]}')


if __name__ == '__main__':
    main()
