"""Quicklook of train/test tile placement overlaid on real LLC4320 SST gradient.

Validates the physics-driven 5-tile layout in train_goflow.py: do the tiles land
on the regimes we intended (Sargasso, slope water, Gulf Stream main jet, Stream
extension, south-wall mix)? Plots a single LLC timestep with the 6 boxes drawn
on top.

Run (after a 944x1666 LLC fetch is available):
    python preprocess/tile_overlay_quicklook.py
    python preprocess/tile_overlay_quicklook.py --llc data/llc_pangeo_2011-09_to_2012-09.nc
    python preprocess/tile_overlay_quicklook.py --time-idx 100 --out data/tiles_t100.png
"""
import argparse
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature


TILES = [
    ('T1 Sargasso SW',    'tab:blue',   0,   256, 256,  512),
    ('T2 Sargasso E',     'tab:blue',   0,   256, 1100, 1356),
    ('T3 NW slope',       'tab:green',  600, 856, 0,    256),
    ('T4 GS main jet',    'tab:red',    600, 856, 550,  806),
    ('T5 GS extension',   'tab:orange', 600, 856, 1100, 1356),
    ('TEST south-wall',   'magenta',    300, 556, 550,  806),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--llc', default='data/llc_pangeo_bigger_2012-04-08_to_2012-04-21.nc')
    ap.add_argument('--time-idx', type=int, default=None,
                    help='Time index to plot (default: middle)')
    ap.add_argument('--out', default='data/tile_overlay_quicklook.png')
    args = ap.parse_args()

    ds = xr.open_dataset(args.llc, decode_times=False)
    ny, nx = ds.sizes['lat'], ds.sizes['lon']
    if ny < 800 or nx < 1500:
        raise SystemExit(
            f'LLC file is {ny}x{nx}; physics-driven layout needs >=800x1500.')
    t = args.time_idx if args.time_idx is not None else ds.sizes['time'] // 2
    print(f'LLC: {os.path.basename(args.llc)}  shape={ds.sizes}  t={t}')

    lgt = ds['loggrad_T'].isel(time=t).values
    lat = ds['lat'].values
    lon = ds['lon'].values

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection=proj)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=proj)

    # SST gradient as colormap
    pcm = ax.pcolormesh(
        lon, lat, lgt, cmap='inferno', vmin=-19, vmax=0,
        shading='auto', transform=proj)
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('log|grad T|')

    ax.add_feature(cfeature.LAND, facecolor='0.85', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.top_labels = gl.right_labels = False

    for label, color, j0, j1, i0, i1 in TILES:
        la0, la1 = lat[j0], lat[j1 - 1]
        lo0, lo1 = lon[i0], lon[i1 - 1]
        rect = patches.Rectangle(
            (lo0, la0), lo1 - lo0, la1 - la0,
            linewidth=2.0, edgecolor=color, facecolor='none',
            zorder=4, transform=proj)
        ax.add_patch(rect)
        ax.text(lo0 + 0.2, la1 - 0.6, label, color=color, fontsize=8,
                fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7),
                transform=proj)

    src = os.path.basename(args.llc)
    ax.set_title(f'GOFLOW training tiles on LLC4320 log|grad T|\n'
                 f'{src}  t={t} of {ds.sizes["time"]}',
                 fontsize=11)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {args.out}  ({os.path.getsize(args.out)/1e3:.0f} KB)')


if __name__ == '__main__':
    main()
