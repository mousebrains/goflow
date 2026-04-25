"""Path C: emit fetch + train commands for a new lat/lon region.

Computes the grid size at the GOFLOW 2-km target resolution, suggests an
evenly-spaced 256x256 training-tile layout, and prints copy-paste commands
to fetch LLC4320 + GOES-16 over the region. The tile suggestion is geometric
(no physics) — for production regions you should still hand-place tiles on
the dominant flow features (jets, fronts, eddy hotspots) the way the GS box
in train_goflow.py is laid out.

Run:
    python preprocess/region_setup.py --lat-min -45 --lat-max -28 \
                                      --lon-min 30 --lon-max 60 \
                                      --start 2011-09-15 --end 2012-09-21 \
                                      --project YOUR_GCP_PROJECT
"""
import argparse
import math


DLAT = DLON = 2.0 / 111.0       # ~2 km nominal


def grid_size(lat_min, lat_max, lon_min, lon_max):
    """Match the lat/lon arange in llc_pangeo_yearly.py / goes_yearly.py."""
    ny = int(round((lat_max - lat_min) / DLAT)) + 1
    nx = int(round((lon_max - lon_min) / DLON)) + 1
    return ny, nx


def suggest_tiles(ny, nx, tile=256, n_train=5):
    """Place tiles on a coarse grid. Last tile reserved as test."""
    if ny < tile or nx < tile:
        raise SystemExit(
            f'Region too small for {tile}x{tile} tiles (got {ny}x{nx}).')
    rows = max(1, ny // tile)
    cols = max(1, nx // tile)
    cells = []
    for r in range(rows):
        j0 = r * (ny - tile) // max(1, rows - 1) if rows > 1 else 0
        for c in range(cols):
            i0 = c * (nx - tile) // max(1, cols - 1) if cols > 1 else 0
            cells.append((j0, j0 + tile, i0, i0 + tile))

    if len(cells) < n_train + 1:
        raise SystemExit(
            f'Region too small to place {n_train} train tiles + 1 test '
            f'(got {len(cells)} cells of {tile}x{tile} on a {rows}x{cols} grid).')

    # Spread train tiles across cells, hold one out as test (chosen as the
    # mid-domain cell so the held-out region differs visually from training).
    step = len(cells) // (n_train + 1)
    train = [cells[i * step] for i in range(1, n_train + 1)]
    test = cells[len(cells) // 2]
    if test in train:
        # Swap the colliding train slot for the first cell that is neither
        # already-train nor the test cell.
        idx = train.index(test)
        for cand in cells:
            if cand not in train and cand != test:
                train[idx] = cand
                break
    return train, test, rows, cols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lat-min', type=float, required=True)
    ap.add_argument('--lat-max', type=float, required=True)
    ap.add_argument('--lon-min', type=float, required=True)
    ap.add_argument('--lon-max', type=float, required=True)
    ap.add_argument('--start',   default='2011-09-15', help='LLC start (YYYY-MM-DD)')
    ap.add_argument('--end',     default='2012-09-21', help='LLC end (YYYY-MM-DD)')
    ap.add_argument('--goes-year', type=int, default=2023,
                    help='GOES year for inference data')
    ap.add_argument('--project', default='YOUR_GCP_PROJECT',
                    help='GCP project ID for requester-pays billing')
    ap.add_argument('--region-name', default='custom',
                    help='Short tag for output filenames')
    args = ap.parse_args()

    ny, nx = grid_size(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    train, test, rows, cols = suggest_tiles(ny, nx)

    box = f'lat {args.lat_min}..{args.lat_max}  lon {args.lon_min}..{args.lon_max}'
    print('-' * 72)
    print(f'Region: {box}')
    print(f'Grid:   {ny} x {nx}  (~{DLAT*111:.1f} km)')
    print(f'Tiles:  {rows}x{cols} grid of 256x256, {len(train)} train + 1 test')
    print('-' * 72)

    print('\n# 1) Confirm which LLC4320 face(s) cover this box:')
    print(f'python preprocess/llc_face_coverage.py {args.project} \\\n'
          f'    --lat-min {args.lat_min} --lat-max {args.lat_max} \\\n'
          f'    --lon-min {args.lon_min} --lon-max {args.lon_max}')

    llc_out = f'data/llc_pangeo_{args.region_name}_{args.start}_{args.end}.nc'
    print('\n# 2) Fetch LLC4320 ground truth (15-30 GB, hours):')
    print(f'python preprocess/llc_pangeo_yearly.py {args.project} \\\n'
          f'    --start {args.start}T00:00:00 --end {args.end}T23:00:00 \\\n'
          f'    --lat-min {args.lat_min} --lat-max {args.lat_max} \\\n'
          f'    --lon-min {args.lon_min} --lon-max {args.lon_max} \\\n'
          f'    --output {llc_out}')

    goes_out = f'data/goes_{args.goes_year}_{args.region_name}.nc'
    print(f'\n# 3) Fetch GOES-16 inference data ({args.goes_year}, ~80 GB, ~2 hrs):')
    print(f'python preprocess/goes_yearly.py \\\n'
          f'    --start {args.goes_year}-01-01 --end {args.goes_year}-12-31 \\\n'
          f'    --lat-min {args.lat_min} --lat-max {args.lat_max} \\\n'
          f'    --lon-min {args.lon_min} --lon-max {args.lon_max} \\\n'
          f'    --output {goes_out}')

    print('\n# 4) Suggested geometric tile layout (overwrite '
          'train_goflow.py with these,')
    print('#    or hand-place tiles on physical features for production):')
    print(f'#    Ny={ny}, Nx={nx}')
    print('train_inds = [')
    for j0, j1, i0, i1 in train:
        print(f'    ({j0:5d}, {j1:5d}, {i0:5d}, {i1:5d}),')
    print(']')
    j0, j1, i0, i1 = test
    print(f'test_inds = ({j0}, {j1}, {i0}, {i1})')

    print('\n# 5) Two-stage training (L1 then L1+spectral):')
    print('python train_twostage.py \\\n'
          f'    --llc_file {llc_out} \\\n'
          f'    --goes_file {goes_out} \\\n'
          f'    --output_dir data/ncfiles_{args.region_name}/')
    print('-' * 72)


if __name__ == '__main__':
    main()
