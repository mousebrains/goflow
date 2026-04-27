"""3-way comparison of c_spec stage-2 weights on the same NESMA inference.

Compares stage-0 (c_spec=0), stage-2 c_spec=0.2, and stage-2 c_spec=0.5.
Reads the per-stage prediction NetCDFs that train_goflow.py writes when
satellite inference is enabled (default).
"""
import json
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

CSPECS = [0.0, 0.2, 0.5]
PRED_TPL = '/Users/pat/tpw/goflow/data/run_paper/preds_lgt_unet16_1_3_{c}cs_goes_nesma.nc'
METRICS_TPL = {
    0.0: None,
    0.2: '/Users/pat/tpw/goflow/data/run_paper/run_metrics_c0.2.json',
    0.5: '/Users/pat/tpw/goflow/data/run_paper/run_metrics.json',
}
OUT_PNG = '/Users/pat/tpw/goflow/data/run_paper/cspec_compare.png'
T_IDX = 300


def load_best(path):
    if path is None:
        return None
    with open(path) as f:
        m = json.load(f)
    s = m.get('summary', {})
    return {
        'velocity_r2': s.get('best_velocity_r2'),
        'gradient_r2': s.get('best_gradient_r2'),
        'spec_loss': s.get('best_spec_loss'),
        'selected_velocity_r2': s.get('selected_velocity_r2'),
        'selected_gradient_r2': s.get('selected_gradient_r2'),
        'selected_spec_loss': s.get('selected_spec_loss'),
        'selected_epoch': s.get('selected_epoch'),
    }


def main():
    fig, axes = plt.subplots(len(CSPECS), 3, figsize=(15, 4 * len(CSPECS)),
                             constrained_layout=True)
    metrics_rows = []
    extent = None
    for row, c in enumerate(CSPECS):
        path = PRED_TPL.format(c=c)
        ds = nc.Dataset(path, 'r')
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:]
        if extent is None:
            extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        u = ds.variables['U'][T_IDX]
        v = ds.variables['V'][T_IDX]
        vor = ds.variables['Vorticity'][T_IDX]
        speed = np.hypot(u, v)
        f0 = 2 * 7.2921e-5 * np.sin(np.deg2rad(40))
        vor_n = vor / f0

        ax = axes[row, 0]
        im = ax.imshow(speed, origin='lower', extent=extent, cmap='turbo',
                       vmin=0, vmax=1.5, aspect='auto')
        ax.set_title(f'c_spec={c}: |U| (m/s)')
        plt.colorbar(im, ax=ax, fraction=0.04)

        ax = axes[row, 1]
        im = ax.imshow(vor_n, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=-2, vmax=2, aspect='auto')
        ax.set_title(f'c_spec={c}: vorticity / f')
        plt.colorbar(im, ax=ax, fraction=0.04)

        ax = axes[row, 2]
        skip = 30
        ax.imshow(speed, origin='lower', extent=extent, cmap='Greys',
                  vmin=0, vmax=1.5, aspect='auto')
        yy, xx = np.meshgrid(lat[::skip], lon[::skip], indexing='ij')
        ax.quiver(xx, yy, u[::skip, ::skip], v[::skip, ::skip],
                  color='steelblue', scale=20)
        ax.set_title(f'c_spec={c}: U,V quiver')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # spec spectrum on speed (1D radial)
        # (skipped — heavy; we already have spec_loss in metrics)
        ds.close()

        # Best metrics
        best = load_best(METRICS_TPL[c])
        if best is None:
            metrics_rows.append((c, '-', '-', '-'))
        else:
            metrics_rows.append((
                c,
                f"{best.get('velocity_r2', '-'):.4f}" if isinstance(best.get('velocity_r2'), float) else '-',
                f"{best.get('gradient_r2', '-'):.4f}" if isinstance(best.get('gradient_r2'), float) else '-',
                f"{best.get('spec_loss', '-'):.4f}" if isinstance(best.get('spec_loss'), float) else '-',
            ))

    fig.suptitle(f'NESMA inference at t={T_IDX}/597 — 3-way c_spec comparison',
                 fontsize=14)
    fig.savefig(OUT_PNG, dpi=120)
    print(f'Saved {OUT_PNG}')
    print()
    print(f"{'c_spec':>8s}  {'velR2':>10s}  {'gradR2':>10s}  {'specLoss':>10s}")
    for row in metrics_rows:
        print(f"{row[0]:>8.1f}  {row[1]:>10s}  {row[2]:>10s}  {row[3]:>10s}")


if __name__ == '__main__':
    main()
