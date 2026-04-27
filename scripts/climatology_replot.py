"""Replot climatology v2 from cached npz with fixed front detection.

Reuses data/run_paper/climatology_v2_0.2cs.npz (already streamed), but restricts
the front-trajectory argmax search to lat in [33, 42] N to suppress spurious
peaks at the southern edge of the domain.
"""
import numpy as np
import matplotlib.pyplot as plt

NPZ = '/Users/pat/tpw/goflow/data/run_paper/climatology_v2_0.2cs.npz'
OUT_PNG = '/Users/pat/tpw/goflow/data/run_paper/climatology_v2_fixed_0.2cs.png'


def front_trajectory(lgt_mean, lat, lat_search=(33.0, 42.0)):
    in_range = (lat >= lat_search[0]) & (lat <= lat_search[1])
    band_idx = np.where(in_range)[0]
    band = lgt_mean[band_idx, :]
    front_in_band = np.nanargmax(band, axis=0)
    front_lat_idx = band_idx[front_in_band]
    win = 11
    half = win // 2
    front_lat_raw = lat[front_lat_idx]
    padded = np.pad(front_lat_raw, half, mode='edge')
    kernel = np.ones(win) / win
    front_lat_smooth = np.convolve(padded, kernel, mode='valid')
    return front_lat_idx, front_lat_smooth


def cross_front_profile(field, front_lat_idx, lat, offsets_deg):
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
    A = dict(np.load(NPZ))
    lat, lon = A['lat'], A['lon']
    front_idx, front_lat_sm = front_trajectory(A['lgt_mean'], lat)
    print(f'Fixed front lat range: {front_lat_sm.min():.2f} to {front_lat_sm.max():.2f} N')
    print(f'  median front lat: {np.median(front_lat_sm):.2f} N')

    offsets = np.arange(-2, 2.05, 0.1)
    cf_speed_mean, _ = cross_front_profile(A['speed_mean_true'],
                                            front_idx, lat, offsets)
    cf_eke, _ = cross_front_profile(A['EKE'], front_idx, lat, offsets)
    cf_mke, _ = cross_front_profile(A['MKE'], front_idx, lat, offsets)

    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    def add_map(ax, field, title, cmap, vmin=None, vmax=None, overlay_front=False):
        if vmin is None:
            vmin = np.nanpercentile(field, 1)
        if vmax is None:
            vmax = np.nanpercentile(field, 99)
        im = ax.imshow(field, origin='lower', extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        if overlay_front:
            ax.plot(lon, front_lat_sm, color='cyan', lw=1.8,
                    alpha=0.9, label='climatological front')
            ax.legend(loc='lower right', fontsize=8)
        ax.axhline(34.21, color='lime', lw=1.0, ls='--', alpha=0.6)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.04)

    add_map(fig.add_subplot(gs[0, 0]), A['speed_mean_true'],
            r'time-mean $\langle|U|\rangle$ (m/s)', 'turbo', 0, 0.5,
            overlay_front=True)
    add_map(fig.add_subplot(gs[0, 1]), A['MKE'],
            'MKE (m^2/s^2)', 'viridis', 0, 0.05)
    add_map(fig.add_subplot(gs[0, 2]), A['EKE'],
            'EKE (m^2/s^2)', 'inferno', 0, 0.1)
    add_map(fig.add_subplot(gs[0, 3]), np.sqrt(A['speed_var']),
            'std(|U|) (m/s)', 'plasma', 0, 0.3)

    add_map(fig.add_subplot(gs[1, 0]), A['lgt_mean'],
            'time-mean log|grad BT| (normed)', 'gray_r', 0.7, 0.95,
            overlay_front=True)

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
    ax.axvline(0, color='cyan', ls='--', alpha=0.6)
    ax.set_title('Across-front profile: MKE and EKE')
    ax.grid(alpha=0.3); ax.legend()

    ax = fig.add_subplot(gs[1, 3])
    ratio = cf_eke / np.maximum(cf_mke, 1e-6)
    ax.plot(offsets, ratio, '-o', lw=2, ms=4, color='tab:purple')
    ax.set_xlabel('offset from front (deg lat, +N)')
    ax.set_ylabel('EKE / MKE')
    ax.axvline(0, color='cyan', ls='--', alpha=0.6)
    ax.axhline(1, color='gray', ls=':', alpha=0.5)
    ax.set_title('Across-front profile: EKE/MKE ratio')
    ax.grid(alpha=0.3)

    months = np.arange(1, 13)
    dlat = lat[1] - lat[0]
    band_idx = int(round(0.5 / dlat))

    front_band_mask = np.zeros_like(A['lgt_mean'], dtype=bool)
    for ix in range(len(lon)):
        i0 = max(0, front_idx[ix] - band_idx)
        i1 = min(len(lat), front_idx[ix] + band_idx + 1)
        front_band_mask[i0:i1, ix] = True

    monthly_mke_band, monthly_eke_band, monthly_speed_band = [], [], []
    monthly_s, monthly_n = [], []
    for m in range(12):
        valid_band = front_band_mask & (A['monthly_count'][m] > 0)
        monthly_mke_band.append(np.nanmean(A['monthly_MKE'][m][valid_band]))
        monthly_eke_band.append(np.nanmean(A['monthly_EKE'][m][valid_band]))
        monthly_speed_band.append(np.nanmean(A['monthly_speed'][m][valid_band]))
        # South / north of front bands
        front_lat_arr = lat[front_idx][None, :]
        lat_2d = lat[:, None].repeat(len(lon), axis=1)
        s_mask = (lat_2d < front_lat_arr - 1.5) & (A['monthly_count'][m] > 0)
        n_mask = (lat_2d > front_lat_arr + 1.5) & (A['monthly_count'][m] > 0)
        monthly_s.append(np.nanmean(A['monthly_speed'][m][s_mask]) if s_mask.any() else np.nan)
        monthly_n.append(np.nanmean(A['monthly_speed'][m][n_mask]) if n_mask.any() else np.nan)

    ax = fig.add_subplot(gs[2, 0:2])
    ax.plot(months, monthly_mke_band, '-o', lw=2, color='tab:green', label='MKE in front band (+/-0.5 deg)')
    ax.plot(months, monthly_eke_band, '-s', lw=2, color='tab:red', label='EKE in front band')
    # Kang+ 2016 reported peaks: EKE in May (off-coast: dominant), September (secondary)
    ax.axvline(5, color='red', ls=':', alpha=0.4, label='Kang+ 2016 EKE primary (May)')
    ax.axvline(9, color='red', ls=':', alpha=0.3, label='Kang+ 2016 EKE secondary (Sep)')
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylabel('KE (m^2/s^2)')
    ax.set_title('Seasonal cycle: MKE / EKE in front band (lat search [33, 42] N)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 2:])
    ax.plot(months, monthly_speed_band, '-o', lw=2, color='tab:blue', label='mean |U| in front band')
    ax.plot(months, monthly_s, '-^', lw=1.5, color='tab:cyan', label='south of front')
    ax.plot(months, monthly_n, '-v', lw=1.5, color='tab:orange', label='north of front')
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylabel('mean speed (m/s)')
    ax.set_title('Seasonal cycle: mean speed by region')
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle('c_spec=0.2 climatology v2 (FIXED front detection: argmax restricted to lat in [33,42] N)',
                 fontsize=13)
    fig.savefig(OUT_PNG, dpi=120)
    print(f'Saved {OUT_PNG}')

    print()
    print('=== Summary ===')
    print(f'Front lat: {front_lat_sm.min():.2f} - {front_lat_sm.max():.2f} N (median {np.median(front_lat_sm):.2f})')
    print(f'Front-band mean speed peak month: {months[np.argmax(monthly_speed_band)]} '
          f'(value {max(monthly_speed_band):.4f} m/s)')
    print(f'Front-band EKE peak month: {months[np.argmax(monthly_eke_band)]} '
          f'(value {max(monthly_eke_band):.4f})')
    print(f'Front-band EKE secondary peak: {months[np.argsort(monthly_eke_band)[-2]]}')
    print(f'Front-band MKE peak month: {months[np.argmax(monthly_mke_band)]}')


if __name__ == '__main__':
    main()
