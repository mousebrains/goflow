"""Quicklook of c_spec=0.2 model applied to the extended GOES region (25-42 N, 80-50 W).

The default valid_inds in inf_llc_stage1.py crops to the bottom-right 512 x 768,
which on our hourly file covers roughly 25-34 N, 64-50 W (Sargasso-Caribbean).
This is well south of the paper's 34-45 N training band — a true out-of-training
spatial test.

Plots three representative time slices (early-year, mid-year, late-year) showing
input log|grad BT|, predicted speed, vorticity/f, and a U,V quiver overlay.
"""
import sys
import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

PRED = '/Users/pat/tpw/goflow/data/run_paper/preds_lgt_unet16_1_3_0.2cs_goes_2023_full.nc'
OUT_PNG = '/Users/pat/tpw/goflow/data/run_paper/quicklook_extended_0.2cs.png'

if not os.path.exists(PRED):
    print(f'ERROR: {PRED} not found')
    sys.exit(1)

ds = nc.Dataset(PRED, 'r')
lat = ds.variables['lat'][:]
lon = ds.variables['lon'][:]
nt = ds.dimensions['time'].size
extent = [lon.min(), lon.max(), lat.min(), lat.max()]
print(f'Predictions span: lat {lat.min():.2f}-{lat.max():.2f}, lon {lon.min():.2f}-{lon.max():.2f}')
print(f'Time steps: {nt}')

# Pick three representative time slices: 25%, 50%, 75% through the year
t_idxs = [nt // 4, nt // 2, 3 * nt // 4]
labels = [f't={ti} (~{int(ti/(nt-1)*365):d} days)' for ti in t_idxs]

f0 = 2 * 7.2921e-5 * np.sin(np.deg2rad(40))

fig, axes = plt.subplots(len(t_idxs), 4, figsize=(20, 4 * len(t_idxs)),
                         constrained_layout=True)
for row, (ti, lbl) in enumerate(zip(t_idxs, labels)):
    lg = ds.variables['loggrad_BT'][ti]
    u = ds.variables['U'][ti]
    v = ds.variables['V'][ti]
    vor = ds.variables['Vorticity'][ti]
    speed = np.hypot(u, v)
    vor_n = vor / f0

    ax = axes[row, 0]
    im = ax.imshow(lg, origin='lower', extent=extent, cmap='gray_r',
                   vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_title(f'{lbl}: input log|grad BT|')
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[row, 1]
    im = ax.imshow(speed, origin='lower', extent=extent, cmap='turbo',
                   vmin=0, vmax=1.5, aspect='auto')
    # Annotate paper training band (34-45 N) — useful only if y-axis covers it
    if lat.max() >= 34:
        ax.axhline(34, color='lime', lw=1, ls='--', alpha=0.6)
    ax.set_title(f'{lbl}: |U| (m/s)')
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[row, 2]
    im = ax.imshow(vor_n, origin='lower', extent=extent, cmap='RdBu_r',
                   vmin=-2, vmax=2, aspect='auto')
    if lat.max() >= 34:
        ax.axhline(34, color='lime', lw=1, ls='--', alpha=0.6)
    ax.set_title(f'{lbl}: vorticity / f')
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[row, 3]
    skip = 30
    ax.imshow(speed, origin='lower', extent=extent, cmap='Greys',
              vmin=0, vmax=1.5, aspect='auto')
    yy, xx = np.meshgrid(lat[::skip], lon[::skip], indexing='ij')
    ax.quiver(xx, yy, u[::skip, ::skip], v[::skip, ::skip],
              color='steelblue', scale=20)
    if lat.max() >= 34:
        ax.axhline(34, color='lime', lw=1, ls='--', alpha=0.6)
    ax.set_title(f'{lbl}: U,V quiver')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

ds.close()

fig.suptitle('c_spec=0.2 model on extended region (paper-trained on 34-45 N; '
             'green dashed = paper southern boundary if visible)', fontsize=13)
fig.savefig(OUT_PNG, dpi=120)
print(f'Saved {OUT_PNG}')
