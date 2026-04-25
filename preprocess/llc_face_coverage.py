"""Report which LLC4320 cube-sphere faces contain cells in a lat/lon box.

The LLC4320 grid is split across 13 faces (0..12). For any region you can
either query all faces or guess the right one — guessing wrong wastes a GCS
read. This script prints a table of cell counts per face so you can wire the
right CANDIDATE_FACES into llc_pangeo_yearly.py.

Run:
    python preprocess/llc_face_coverage.py YOUR_PROJECT_ID
    python preprocess/llc_face_coverage.py YOUR_PROJECT_ID --lat-min -10 --lat-max 10 \
                                                              --lon-min -180 --lon-max 180
"""
import argparse
import os
import sys
import gc
import numpy as np
import xarray as xr
import gcsfs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('project', help='GCP project ID for requester-pays billing')
    ap.add_argument('--lat-min', type=float, default=25.0)
    ap.add_argument('--lat-max', type=float, default=42.0)
    ap.add_argument('--lon-min', type=float, default=-80.0)
    ap.add_argument('--lon-max', type=float, default=-50.0)
    args = ap.parse_args()

    os.environ['GOOGLE_CLOUD_PROJECT'] = args.project
    fs = gcsfs.GCSFileSystem(token='google_default',
                             project=args.project, requester_pays=True)
    grid_url = 'gs://pangeo-ecco-llc4320/grid'
    print(f'Opening {grid_url} (project={args.project})...')
    grid = xr.open_zarr(fs.get_mapper(grid_url), consolidated=True)
    n_faces = grid.sizes['face']
    print(f'Grid: {n_faces} faces, j={grid.sizes["j"]}, i={grid.sizes["i"]}')
    print(f'Box:  lat {args.lat_min}..{args.lat_max}  lon {args.lon_min}..{args.lon_max}\n')

    print(f'{"face":>4}  {"cells":>10}  {"j_min":>6} {"j_max":>6}  {"i_min":>6} {"i_max":>6}'
          f'  {"lat range":>16}  {"lon range":>16}')
    hits = []
    for f in range(n_faces):
        XC = grid.XC.isel(face=f).values
        YC = grid.YC.isel(face=f).values
        in_box = ((YC >= args.lat_min) & (YC <= args.lat_max) &
                  (XC >= args.lon_min) & (XC <= args.lon_max))
        n = int(in_box.sum())
        if n == 0:
            print(f'{f:>4}  {n:>10}  ' + '-' * 64)
            continue
        j_inds, i_inds = np.where(in_box)
        j0, j1 = int(j_inds.min()), int(j_inds.max())
        i0, i1 = int(i_inds.min()), int(i_inds.max())
        la0 = float(YC[in_box].min()); la1 = float(YC[in_box].max())
        lo0 = float(XC[in_box].min()); lo1 = float(XC[in_box].max())
        print(f'{f:>4}  {n:>10,}  {j0:>6} {j1:>6}  {i0:>6} {i1:>6}'
              f'  {la0:>6.1f}..{la1:>6.1f}  {lo0:>6.1f}..{lo1:>6.1f}')
        hits.append(f)

    if not hits:
        print('\nNo face has cells in the requested box.')
        sys.exit(1)
    print(f'\nCANDIDATE_FACES = {tuple(hits)}')
    print('Wire this into llc_pangeo_yearly.py (CANDIDATE_FACES) for your region.')

    del fs
    gc.collect()


if __name__ == '__main__':
    main()
