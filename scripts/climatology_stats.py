"""Time-mean climatology and zonal-mean profiles for c_spec=0.2 inference.

Reads the per-strip prediction NetCDFs (north + south) for the full year 2023,
streams through frames to accumulate per-pixel mean and variance, and plots:

  1. Combined N+S climatology maps: mean U, mean V, mean speed, valid fraction
  2. Zonal-mean profiles: mean |U| vs latitude (model)
  3. Time series of spatial-mean speed (daily aggregates)

Treats output==0 as cloud/land masked (the inference script multiplies by mask
before writing). Per-pixel statistics use only valid frames.

Outputs:
  data/run_paper/climatology_0.2cs.png       — climatology + zonal panels
  data/run_paper/climatology_0.2cs.npz       — raw arrays for downstream use
"""
import os
import time
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

SOUTH = '/Users/pat/tpw/goflow/data/run_paper/preds_lgt_unet16_1_3_0.2cs_goes_2023_full.nc'
NORTH = '/Users/pat/tpw/goflow/data/run_paper/preds_lgt_unet16_1_3_0.2cs_goes_2023_full_y512-944_x898-1666.nc'
OUT_PNG = '/Users/pat/tpw/goflow/data/run_paper/climatology_0.2cs.png'
OUT_NPZ = '/Users/pat/tpw/goflow/data/run_paper/climatology_0.2cs.npz'

CHUNK = 100  # frames per read


def stream_stats(path, var_names):
    """Stream through a NetCDF file and accumulate per-pixel sum, sum_sq, count.

    Returns dict[var] = (mean, var, count) plus lat, lon, time arrays.
    Treats value == 0 as masked (consistent with how predictions are written).
    """
    print(f'  reading {path}')
    ds = nc.Dataset(path, 'r')
    lat = np.asarray(ds.variables['lat'][:])
    lon = np.asarray(ds.variables['lon'][:])
    time_arr = np.asarray(ds.variables['time'][:])
    nt, ny, nx = (ds.dimensions['time'].size,
                  ds.dimensions['lat'].size,
                  ds.dimensions['lon'].size)
    print(f'  shape: nt={nt}, ny={ny}, nx={nx}')

    accum = {v: {'sum': np.zeros((ny, nx), dtype=np.float64),
                 'sumsq': np.zeros((ny, nx), dtype=np.float64),
                 'count': np.zeros((ny, nx), dtype=np.int64)}
             for v in var_names}
    # Daily-mean speed time series (aggregating across spatial dim per frame
    # with cloud mask). We build per-frame, then aggregate to daily later.
    speed_per_frame = np.zeros(nt, dtype=np.float32)
    valid_frac_per_frame = np.zeros(nt, dtype=np.float32)

    t0 = time.time()
    for c0 in range(0, nt, CHUNK):
        c1 = min(nt, c0 + CHUNK)
        # Build a single mask from U: zero in U usually means masked.
        u_chunk = np.asarray(ds.variables['U'][c0:c1])  # (chunk, ny, nx)
        v_chunk = np.asarray(ds.variables['V'][c0:c1])
        valid = (u_chunk != 0) | (v_chunk != 0)
        speed_chunk = np.hypot(u_chunk, v_chunk)
        # Per-frame summaries (used later)
        valid_count_per_frame = valid.reshape(c1 - c0, -1).sum(axis=1)
        speed_sum_per_frame = (speed_chunk * valid).reshape(c1 - c0, -1).sum(axis=1)
        speed_per_frame[c0:c1] = np.where(valid_count_per_frame > 0,
                                           speed_sum_per_frame / np.maximum(valid_count_per_frame, 1),
                                           0)
        valid_frac_per_frame[c0:c1] = valid_count_per_frame / (ny * nx)

        for v in var_names:
            arr = np.asarray(ds.variables[v][c0:c1])
            valid_v = (arr != 0)
            accum[v]['sum'] += np.where(valid_v, arr, 0).sum(axis=0)
            accum[v]['sumsq'] += np.where(valid_v, arr * arr, 0).sum(axis=0)
            accum[v]['count'] += valid_v.sum(axis=0)
        if c0 % 1000 == 0:
            elapsed = time.time() - t0
            rate = (c1 / elapsed) if elapsed > 0 else 0
            eta = (nt - c1) / max(rate, 1) / 60
            print(f'    {c1}/{nt} frames, {rate:.0f} fr/s, ETA {eta:.1f} min')
    ds.close()

    out = {}
    for v in var_names:
        cnt = np.maximum(accum[v]['count'], 1)
        mean = accum[v]['sum'] / cnt
        meansq = accum[v]['sumsq'] / cnt
        var = np.maximum(meansq - mean * mean, 0)
        # Mask cells with zero valid frames
        bad = accum[v]['count'] == 0
        mean[bad] = np.nan
        var[bad] = np.nan
        out[v] = {'mean': mean, 'var': var, 'count': accum[v]['count']}
    return out, lat, lon, time_arr, speed_per_frame, valid_frac_per_frame


def main():
    var_names = ['U', 'V', 'Vorticity']
    print('SOUTH strip (extension, 25-34 N)')
    s_stats, s_lat, s_lon, s_time, s_speed, s_valfrac = stream_stats(SOUTH, var_names)
    print('NORTH strip (paper-overlap, 34-42 N)')
    n_stats, n_lat, n_lon, n_time, n_speed, n_valfrac = stream_stats(NORTH, var_names)

    # Stitch
    lat_all = np.concatenate([s_lat, n_lat])
    lon = s_lon
    fields = {}
    for v in var_names:
        fields[f'{v}_mean'] = np.concatenate([s_stats[v]['mean'],
                                              n_stats[v]['mean']], axis=0)
        fields[f'{v}_std'] = np.concatenate([np.sqrt(s_stats[v]['var']),
                                             np.sqrt(n_stats[v]['var'])], axis=0)
        fields[f'{v}_count'] = np.concatenate([s_stats[v]['count'],
                                               n_stats[v]['count']], axis=0)
    speed_mean = np.hypot(fields['U_mean'], fields['V_mean'])
    valid_frac = fields['U_count'] / 8743.0  # nt = 8743 from earlier

    # Save raw arrays
    np.savez_compressed(
        OUT_NPZ, lat=lat_all, lon=lon,
        U_mean=fields['U_mean'], V_mean=fields['V_mean'],
        speed_mean=speed_mean,
        Vorticity_mean=fields['Vorticity_mean'],
        U_std=fields['U_std'], V_std=fields['V_std'],
        Vorticity_std=fields['Vorticity_std'],
        valid_frac=valid_frac,
        s_speed_per_frame=s_speed, n_speed_per_frame=n_speed,
        s_valfrac_per_frame=s_valfrac, n_valfrac_per_frame=n_valfrac,
    )
    print(f'Saved {OUT_NPZ}')

    # ===== PLOT =====
    fig = plt.figure(figsize=(20, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    extent = [lon.min(), lon.max(), lat_all.min(), lat_all.max()]

    def add_map(ax, field, title, cmap, vmin, vmax, train_line=True):
        im = ax.imshow(field, origin='lower', extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect='auto')
        if train_line:
            ax.axhline(34.21, color='lime', lw=1.2, ls='--', alpha=0.7,
                       label='paper southern boundary')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.04)

    add_map(fig.add_subplot(gs[0, 0]), fields['U_mean'],
            'time-mean U (m/s)', 'RdBu_r', -0.8, 0.8)
    add_map(fig.add_subplot(gs[0, 1]), fields['V_mean'],
            'time-mean V (m/s)', 'RdBu_r', -0.5, 0.5)
    add_map(fig.add_subplot(gs[0, 2]), speed_mean,
            'time-mean |U| (m/s)', 'turbo', 0, 1.0)
    add_map(fig.add_subplot(gs[0, 3]), valid_frac,
            'valid-frame fraction (1 = clear-sky)', 'viridis', 0, 1)

    add_map(fig.add_subplot(gs[1, 0]), fields['U_std'],
            'std(U) (m/s) — eddy variance', 'inferno', 0, 0.5)
    add_map(fig.add_subplot(gs[1, 1]), fields['V_std'],
            'std(V) (m/s) — eddy variance', 'inferno', 0, 0.5)
    f0 = 2 * 7.2921e-5 * np.sin(np.deg2rad(34))
    add_map(fig.add_subplot(gs[1, 2]), fields['Vorticity_mean'],
            'time-mean Vorticity (raw units)', 'RdBu_r', -0.5, 0.5)
    add_map(fig.add_subplot(gs[1, 3]), fields['Vorticity_std'],
            'std(Vorticity) — eddy intensity', 'inferno', 0, 1.0)

    # Zonal-mean profile
    ax = fig.add_subplot(gs[2, 0:2])
    zonal_speed = np.nanmean(speed_mean, axis=1)
    ax.plot(zonal_speed, lat_all, lw=2, color='tab:blue', label='time-mean |U|')
    ax.axhline(34.21, color='lime', lw=1.2, ls='--', alpha=0.7,
               label='paper southern boundary')
    # Common Gulf Stream axis literature: ~36 N at -65 W
    ax.axhline(36, color='gold', lw=1.2, ls=':', alpha=0.7,
               label='~GS axis literature (36 N)')
    ax.set_xlabel('mean speed (m/s)')
    ax.set_ylabel('latitude (N)')
    ax.set_title('Zonal mean speed (longitude-averaged) vs latitude — '
                 '2023 climatology')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(lat_all.min(), lat_all.max())

    # Time series: daily mean speed (aggregate hourly to daily)
    ax = fig.add_subplot(gs[2, 2:])
    # Crude daily aggregation: 24 hourly frames per day
    nt = len(s_speed)
    days = nt // 24
    s_daily = s_speed[:days * 24].reshape(days, 24).mean(axis=1)
    n_daily = n_speed[:days * 24].reshape(days, 24).mean(axis=1)
    s_validfrac_daily = s_valfrac[:days * 24].reshape(days, 24).mean(axis=1)
    day_axis = np.arange(days)
    # Mask days with very low valid fraction
    s_daily_m = np.where(s_validfrac_daily > 0.2, s_daily, np.nan)
    n_daily_m = np.where(s_validfrac_daily > 0.2, n_daily, np.nan)
    ax.plot(day_axis, s_daily_m, lw=1, alpha=0.7, color='tab:blue',
            label='SOUTH (25-34 N)')
    ax.plot(day_axis, n_daily_m, lw=1, alpha=0.7, color='tab:orange',
            label='NORTH (34-42 N)')
    ax.set_xlabel('day of 2023')
    ax.set_ylabel('daily-mean spatial-mean |U| (m/s)')
    ax.set_title('Daily mean of spatial-mean speed (clear-sky weighted)')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle('c_spec=0.2 model 2023 climatology — N+S strips combined',
                 fontsize=14)
    fig.savefig(OUT_PNG, dpi=120)
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
