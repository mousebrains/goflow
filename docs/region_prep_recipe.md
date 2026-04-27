# Preparing GOFLOW Input Files for a New Region

This document records the exact NetCDF schemas required by the GOFLOW
training and inference scripts, derived from inspection of the paper's
distributed files (`llcGoes_gradT_trunc.nc`, `goes_nesma.nc`,
`goes_fig1.nc`) and the dataset classes in `dataSST.py`.

The goal is: given a new region, what file format do you need to produce so
that `train_goflow.py` and `inf_llc_stage1.py` work without code changes?

## Two file types are needed

| File | Used by | Purpose |
|---|---|---|
| **LLC training file** | `train_goflow.py` (`SSTDataset`) | Paired (log\|grad SST\|, U, V) on a regular lat/lon grid |
| **GOES inference file** | `inf_llc_stage1.py` (`SatelliteDataset`) | log\|grad BT\| only, no targets |

## LLC training file schema

Schema expected by `SSTDataset` (reading `var_names=['loggrad_T','U','V']`):

```
dimensions:
    time            unlimited
    lat             N_lat
    lon             N_lon
variables:
    time(time)      f8   units = "seconds since 2000-01-01 00:00:00"
    lat(lat)        f4   degrees_north
    lon(lon)        f4   degrees_east
    loggrad_T(time, lat, lon)  f4   range [-19, 0]   (log of |grad SST| in K/m)
    U(time, lat, lon)          f4   m/s   surface zonal velocity, 18-hr LP-filtered
    V(time, lat, lon)          f4   m/s   surface meridional velocity, 18-hr LP-filtered
```

Notes that matter:

- **Variable name is `loggrad_T`** (underscore between `loggrad` and `T`).
  `dataSST.py:208` and `dataSST.py:290` test this exact name to decide which
  normalization to apply. `log_gradT` (used in the GOES file) is a different
  name and triggers the SST-Celsius normalization branch instead.
- **Normalization happens inside the dataset class**, not in the file. Store
  raw `loggrad_T` clipped to `[-19, 0]`; the model receives it remapped to
  `[0, 1]` via `(x - lgtMin) / (lgtMax - lgtMin)` (`lgtMin=-19, lgtMax=0`).
- **U and V are 18-hr low-pass filtered** in the published training file.
  Paper Methods specify a Butterworth filter to remove internal-gravity-wave
  variability. Apply the same filter when building a new region or the
  predicted velocities will not be comparable to paper results. SST is *not*
  filtered (and shouldn't be — IGWs barely show up in BT).
- **Time stride** is set by `--step0` (default 1) in `train_goflow.py`.
  Paper file is hourly; with step0=1 and nframes=3 the model sees 3 frames
  spaced 1 hr apart. Match this cadence in any new file.
- **Land/coast handling**: U,V are NaN over land in the published file. The
  current `SSTDataset` returns a `valid_mask` alongside `(input, target)` so
  the train loop can mask the L1 loss. NaNs in `loggrad_T` over land are also
  fine (`np.nan_to_num` zeros them out).
- **Spatial extent**: `train_goflow.py` slices the array using
  `(y0, y1, x0, x1)` boxes. With `--layout auto/physics/paper/geometric` the
  layout is computed from `(N_lat, N_lon)`. With `--regions_file` the boxes
  are explicit and must fit within `(N_lat, N_lon)`.

### Confirmed layout of the paper's `llcGoes_gradT_trunc.nc` (121 GB)

```
dimensions: time=8230, lat=551, lon=1001
variables:
    lat(551)        f8   34.000 -> 45.000  (dlat=0.020 deg, ~1900 m)
    lon(1001)       f8   -80.000 -> -60.000 (dlon=0.020 deg, ~1700 m at 40 N)
    loggrad_T(time, lat, lon)  f4   range observed [-7.5, 0]   (well within [-19, 0] clip)
    U(time, lat, lon)          f4   m/s, 18-hr LP filtered, range ~[-3, +3]
    V(time, lat, lon)          f4   m/s, 18-hr LP filtered, range ~[-3, +3]
    vort(time, lat, lon)       f4   1/s, vorticity, ~1e-3 magnitude
    Uwave(time, lat, lon)      f4   m/s, IGW (high-frequency) component, ~[-4, +4]
    inverse_grid_size(lat, lon) f8  m^-2, 1/cell area
```

Notes that differ from what `SSTDataset` strictly needs:

- **No `time` coordinate variable** in the file — only the dimension exists.
  `SSTDataset` reads via integer indexing (`getData`) and never reads `time`,
  so this is fine. Don't bother adding one.
- **No NaN values** anywhere. Land must be zero-filled or interpolated; the
  mask-aware loss is a no-op on this file. (Still useful for files we build
  ourselves over coastal regions.)
- **`vort`, `Uwave`, `inverse_grid_size` are extras** the model never reads.
  `Uwave` is the high-frequency component complementary to the LP-filtered U.
- **8230 hourly frames = 343 days, ~11.3 months**. The "trunc" in the name
  reflects truncation at the LP-filter edges.
- **Same spatial grid as the paper GOES files** (551 x 1001 at 0.02 deg over
  34-45 N, 80-60 W). Inference on Fig1/NESMA against a model trained on this
  file is a direct drop-in.

With `(551, 1001)`, `train_goflow.py --layout auto` selects the **`paper`
layout** (south-half tiles), which is what the paper used.

Our existing `data/llc_pangeo_2011-09_to_2012-09.nc` (108 GB, 8952 hourly
frames over 944 x 1666, ~2 km, 25-42 N, 80-50 W, 18-hr LP filtered U,V)
already conforms to this schema. With `(944, 1666)`, `--layout auto` selects
the **`physics` layout** (5-tile Gulf Stream regime sweep) instead.

Our existing `data/llc_pangeo_2011-09_to_2012-09.nc` (108 GB, 8952 hourly
frames over 944 x 1666, ~2 km, 25-42 N, 80-50 W, 18-hr LP filtered U,V)
already conforms to this schema — it was built specifically to be drop-in.

## GOES inference file schema

Schema confirmed from inspection of `goes_nesma.nc` and `goes_fig1.nc`:

```
dimensions:
    time            N_time
    lat             N_lat
    lon             N_lon
variables:
    time(time)      f8   units = "seconds since 2000-01-01 00:00:00.0"  (UTC)
    lat(lat)        f8   degrees_north
    lon(lon)        f8   degrees_east
    BT(time,lat,lon)               f4   Celsius   (ABI Band 14 brightness temperature)
    gradT(time,lat,lon)            f4   K/m       (raw |grad BT|)
    log_gradT(time,lat,lon)        f4   log(K/m)  (clipped to ~[-19, 0])
    mask(time,lat,lon)             f4   {0, 1}    (1 = valid open-ocean)
    log_gradT_masked(time,lat,lon) f4
    BT_masked(time,lat,lon)        f4
```

The inference code (`SatelliteDataset` in `dataSST.py`) only reads:

- `var_names[0]` (default `log_gradT`)
- `time`, `lat`, `lon` (for output coordinate copying via `writeGridSat`)

Notes that matter:

- **Variable name is `log_gradT`** (underscore between `log_grad` and `T`)
  — different from the LLC file's `loggrad_T`. Don't typo this.
- **Cadence: 5 minutes.** Both paper files use 300 s spacing. This matters
  because `SatelliteDataset.__getitem__` hardcodes:
  ```
  sst_slices = [getData(..., idx),
                getData(..., idx + 12),
                getData(..., idx + 24)]
  ```
  At 5-min cadence the three frames are 0/60/120 minutes apart (1 hr stride).
  Our existing `data/goes_2023_full.nc` is at 1-hr cadence, which would give
  0/12/24-hour spacing instead and break the time-frame assumption. To use
  hourly data with the unmodified inference code, set
  `--var_names log_gradT --step 1` (no such flag exists today; would require
  adding a stride parameter to `SatelliteDataset`).
- **BT units = Celsius** in paper files, **Kelvin** in our build. The model
  inputs `log_gradT` only, and the gradient magnitude is unit-invariant under
  K/C (offset cancels in `d/dx`), so `log_gradT` from K-based BT and C-based
  BT are identical. Only matters if you want to re-derive `log_gradT` from
  `BT` after the fact.
- **Mask dtype is `float32`** in paper files (values in {0, 1}), but `uint8`
  in our build. `dataSST.SatelliteDataset` does not read mask, so this only
  matters if downstream code does.
- **Lat/lon are `f8`** in paper files, `f4` in ours. Both work.

### Building a new GOES region from scratch

`preprocess/goes_yearly.py` produces our hourly-cadence flavor. To match the
paper's 5-min cadence:

1. List **every** ABI L1b RadC Channel-14 file in the hour, not just the
   first (`list_first_per_hour` -> `list_all_per_hour`).
2. Project each frame to lat/lon as before.
3. Stream-write at 5-min cadence; expect ~12x the frame count of the hourly
   build (~80,000 frames per year, ~500 GB at full year of 5-min).
4. Add `BT_masked = BT * mask` and `log_gradT_masked = log_gradT * mask` if
   you want to match the paper file exactly. Inference does not require them.

For a *new region* where we only need a few days of inference (like the
paper's 48-hr Fig1 / 50-hr NESMA windows), `preprocess/goes_yearly.py` with
modified hour-frame logic is the right starting point. A focused 2-day
window at 5-min cadence is ~600 frames * 551 * 1001 * 4 bytes per variable,
~6-7 GB total — matches the paper file size.

## Two-stage training recipe (per published README)

Stage 0: pure L1 loss, 100 epochs.
Stage 1: L1 + spectral, 50 epochs, loaded from Stage 0 best.

```bash
python train_twostage.py \
    --llc_file data/paper/llcGoes_gradT_trunc.nc \
    --model unet --nbase 16 --epochs_stage1 100 --epochs_stage2 50 \
    --c_spec_stage2 0.5 \
    --metrics_file data/paper/run_metrics.json
```

The paper's Methods "Loss function design" section explicitly scanned `c_spec`
from 0.05 to 0.9 and chose **0.2** as the compromise solution. Use 0.2 as the
default. The README's Quick Start uses 0.5 as a placeholder; the README's
"Two-Stage Training Workflow" section (L189-207) and the paper both use 0.2.

## Hand-off checklist for a new region

1. Decide the lat/lon box and time window (hourly LLC + 5-min GOES).
2. Build the LLC file: hourly U, V, log\|grad SST\| with U,V 18-hr LP filtered,
   variable names exactly `loggrad_T`, `U`, `V`.
3. Build the GOES file: 5-min cadence, variable names exactly `log_gradT`
   (and `BT`, `mask` if you want to mask out clouds in plots).
4. Sanity-check both files with the inspection script in this doc (or
   `python3 -c "import netCDF4; ds=...; print(ds.variables)"`).
5. Run `train_twostage.py` with the new files.
6. Run `inf_llc_stage1.py --model_file ... --goes_files <new GOES>` for the
   inference figures.
