"""Combined N+S strip view for the c_spec=0.2 model on our extended region.

The south strip (default valid_inds) and the north strip (--valid_inds 512 944 898 1666)
share the same lon range and abut at lat 34.21 N. We concatenate them along lat and
plot the combined 944 x 768 field, with the green dashed line marking the boundary
between the paper's training region (north) and the extension (south).
"""
import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

SOUTH = '/Users/pat/tpw/goflow/data/run_paper/preds_lgt_unet16_1_3_0.2cs_goes_2023_full.nc'
NORTH = '/Users/pat/tpw/goflow/data/run_paper/preds_lgt_unet16_1_3_0.2cs_goes_2023_full_y512-944_x898-1666.nc'
OUT_PNG = '/Users/pat/tpw/goflow/data/run_paper/quicklook_strips_0.2cs.png'


def load_slice(path, t_idx, var):
    ds = nc.Dataset(path, 'r')
    a = ds.variables[var][t_idx]
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    ds.close()
    return a, lat, lon


def stitch(t_idx):
    """Return (combined_field_dict, combined_lat, lon) for time t_idx."""
    out = {}
    for var in ['loggrad_BT', 'U', 'V', 'Vorticity']:
        s_a, s_lat, s_lon = load_slice(SOUTH, t_idx, var)
        n_a, n_lat, n_lon = load_slice(NORTH, t_idx, var)
        # Stitch along lat (axis 0)
        combined = np.concatenate([s_a, n_a], axis=0)
        out[var] = combined
    combined_lat = np.concatenate([s_lat, n_lat])
    return out, combined_lat, s_lon


def main():
    # Sample three time slices from the year
    nt_south = nc.Dataset(SOUTH, 'r').dimensions['time'].size
    t_idxs = [nt_south // 4, nt_south // 2, 3 * nt_south // 4]
    labels = [f't={ti} (~day {int(ti/(nt_south-1)*365):d})' for ti in t_idxs]

    f0 = 2 * 7.2921e-5 * np.sin(np.deg2rad(34))

    fig, axes = plt.subplots(len(t_idxs), 4, figsize=(20, 5.2 * len(t_idxs)),
                             constrained_layout=True)
    extent = None
    for row, (ti, lbl) in enumerate(zip(t_idxs, labels)):
        fields, lat, lon = stitch(ti)
        u, v = fields['U'], fields['V']
        speed = np.hypot(u, v)
        vor_n = fields['Vorticity'] / f0
        if extent is None:
            extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        ax = axes[row, 0]
        im = ax.imshow(fields['loggrad_BT'], origin='lower', extent=extent,
                       cmap='gray_r', vmin=0.5, vmax=1.0, aspect='auto')
        ax.axhline(34.21, color='lime', lw=1.2, ls='--', alpha=0.7)
        ax.set_title(f'{lbl}: input log|grad BT|')
        plt.colorbar(im, ax=ax, fraction=0.04)

        ax = axes[row, 1]
        im = ax.imshow(speed, origin='lower', extent=extent, cmap='turbo',
                       vmin=0, vmax=1.5, aspect='auto')
        ax.axhline(34.21, color='lime', lw=1.2, ls='--', alpha=0.7)
        ax.set_title(f'{lbl}: |U| (m/s)')
        plt.colorbar(im, ax=ax, fraction=0.04)

        ax = axes[row, 2]
        im = ax.imshow(vor_n, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=-2, vmax=2, aspect='auto')
        ax.axhline(34.21, color='lime', lw=1.2, ls='--', alpha=0.7)
        ax.set_title(f'{lbl}: vorticity / f')
        plt.colorbar(im, ax=ax, fraction=0.04)

        ax = axes[row, 3]
        skip_y = max(20, len(lat) // 30)
        skip_x = max(20, len(lon) // 30)
        ax.imshow(speed, origin='lower', extent=extent, cmap='Greys',
                  vmin=0, vmax=1.5, aspect='auto')
        yy, xx = np.meshgrid(lat[::skip_y], lon[::skip_x], indexing='ij')
        ax.quiver(xx, yy, u[::skip_y, ::skip_x], v[::skip_y, ::skip_x],
                  color='steelblue', scale=20)
        ax.axhline(34.21, color='lime', lw=1.2, ls='--', alpha=0.7)
        ax.set_title(f'{lbl}: U,V quiver')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    fig.suptitle('c_spec=0.2 model on N+S strips: 25-42 N, 64-50 W; '
                 'green dashed line = paper southern boundary at 34.21 N. '
                 'Above = within paper training latitudes; below = extension.',
                 fontsize=13)
    fig.savefig(OUT_PNG, dpi=120)
    print(f'Saved {OUT_PNG}')

    # Quick stats per strip (using first time slice)
    print(f'\n=== Statistics at t={t_idxs[0]} ===')
    for label, path, slc in [('SOUTH (extension, 25-34 N)', SOUTH, slice(0, 512)),
                              ('NORTH (paper-overlap, 34-42 N)', NORTH, slice(0, 432))]:
        ds = nc.Dataset(path, 'r')
        u = np.asarray(ds.variables['U'][t_idxs[0]])
        v = np.asarray(ds.variables['V'][t_idxs[0]])
        vor = np.asarray(ds.variables['Vorticity'][t_idxs[0]])
        speed = np.hypot(u, v)
        m = speed > 1e-6  # exclude masked-zero pixels
        if m.any():
            print(f'  {label:40s} |U| mean={speed[m].mean():.3f} m/s, '
                  f'|U| 95%={np.percentile(speed[m], 95):.3f}, '
                  f'vor_n std={(vor[m] / f0).std():.3f}')
        ds.close()


if __name__ == '__main__':
    main()
