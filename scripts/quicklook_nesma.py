"""Stage-0 vs stage-1 NESMA quicklook: side-by-side input + predicted fields."""
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PATHS = {
    'Stage 0 (L1 only)': '/Users/pat/tpw/goflow/data/run_paper/preds_lgt_unet16_1_3_0.0cs_goes_nesma.nc',
    'Stage 1 (L1 + 0.5 spec)': '/Users/pat/tpw/goflow/data/run_paper/preds_lgt_unet16_1_3_0.5cs_goes_nesma.nc',
}
OUTPATH = '/Users/pat/tpw/goflow/data/run_paper/quicklook_nesma_stage_compare.png'
T_IDX = 300  # middle of NESMA window

fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
extent = None
for row, (label, path) in enumerate(PATHS.items()):
    ds = nc.Dataset(path, 'r')
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    if extent is None:
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    lg = ds.variables['loggrad_BT'][T_IDX]
    u  = ds.variables['U'][T_IDX]
    v  = ds.variables['V'][T_IDX]
    vor = ds.variables['Vorticity'][T_IDX]
    speed = np.hypot(u, v)
    # Coriolis-normalized vorticity at 40N: f = 9.4e-5 s^-1
    f0 = 2*7.2921e-5*np.sin(np.deg2rad(40))
    vor_n = vor / f0

    if row == 0:
        ax = axes[0, 0]; im = ax.imshow(lg, origin='lower', extent=extent, cmap='gray_r', vmin=0.5, vmax=1.0, aspect='auto')
        ax.set_title('Input: log|grad BT| (normed)'); plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[row, 1]; im = ax.imshow(speed, origin='lower', extent=extent, cmap='turbo', vmin=0, vmax=1.5, aspect='auto')
    ax.set_title(f'{label}: |U| (m/s)'); plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[row, 2]; im = ax.imshow(vor_n, origin='lower', extent=extent, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_title(f'{label}: vorticity / f'); plt.colorbar(im, ax=ax, fraction=0.04)

    # Quiver overlay on speed for stage 1 only (cleanest)
    ax = axes[row, 3]
    skip = 30
    ax.imshow(speed, origin='lower', extent=extent, cmap='Greys', vmin=0, vmax=1.5, aspect='auto')
    yy, xx = np.meshgrid(lat[::skip], lon[::skip], indexing='ij')
    ax.quiver(xx, yy, u[::skip, ::skip], v[::skip, ::skip], color='steelblue', scale=20)
    ax.set_title(f'{label}: U,V quiver')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ds.close()

# Hide unused top-row axes (we only filled (0,0) once, but we put input only in row 0)
axes[1, 0].axis('off')

fig.suptitle(f'NESMA inference, t={T_IDX}/597 (~25 hr into 50-hr window)', fontsize=14)
fig.savefig(OUTPATH, dpi=120)
print(f'Saved {OUTPATH}')
