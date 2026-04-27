# 2023 GOFLOW Climatology Validation Notes

This doc records the validation we did for the c_spec=0.2 reproduction of the
GOFLOW paper recipe (Lenain et al. 2026, Nature Geoscience) applied to 2023
GOES Atlantic data, with comparison against LLC4320 truth (paper training file)
and CMEMS AVISO L4 altimetry.

## Setup

Model: `data/run_paper/lgt_unet16_1_3_0.2cs.pth` (paper recipe — Stage 0
L1-only 100 epochs, Stage 1 L1+spectral c_spec=0.2 50 epochs).

Inference: `inf_llc_stage1.py` on `data/goes_2023_full.nc` (8743 hourly frames,
25-42 N, 80-50 W). Two strips:
- South: `preds_..._goes_2023_full.nc`         (Y 0-512, lat 25-34.2 N)
- North: `preds_..._goes_2023_full_y512-944_x898-1666.nc` (Y 512-944, lat 34.2-42 N)

**Important:** Both strips were run with `--valid_inds ... 898 1666`, i.e.
limited to lon -63.82 to -50 W (x∈[898,1666] of the 1666-lon source). The
western half of the GOES domain (lon -80 to -64 W, including Cape Hatteras
separation) was *not* run — would require ~3-5 additional hours of inference.

After cropping to lon < -57.5 W (to remove the GOES RadC east-sector mask=0
stripe), the model's valid analysis region is **lat 25-42 N, lon -63.82 to
-57.51 W**. The Gulf Stream extension axis sits in this region around lat
38-40 N.

## Climatology pipeline

`scripts/climatology_v2.py` streams both strip files, accumulates per-pixel
mean/variance for U, V, and |U|, plus monthly bins for MKE/EKE/speed. Cache
is `data/run_paper/climatology_v2_0.2cs.npz` (~77 MB).

`scripts/climatology_replot.py` reuses the cache to regenerate plots without
re-streaming. Front detection (climatological GS axis) is restricted to the
lat search window [33, 42] N to suppress spurious argmax peaks at the southern
edge of the domain. The `np.convolve` smoothing uses edge-padding (otherwise
the kernel zero-pads at column 0 / nx-1 and pulls the smoothed front
unrealistically southward).

Output: `data/run_paper/climatology_v2_fixed_0.2cs.png`. Median front lat
38.9 N, range [38.25, 40.61] N — consistent with the GS extension at -65 W
in altimetry climatology.

## AVISO comparison

`scripts/aviso_compare.py` loads `data/aviso/aviso_l4_duacs_2023_25-42N_80-58W.nc`
(CMEMS L4 1/8 deg, 365 daily frames; pull recipe in
`memory/reference_cmems_aviso.md`).

Output: `data/run_paper/aviso_vs_model_0.2cs.png`. Caveat: AVISO geostrophic
velocity at 1/8 deg is paper-acknowledged as "diffused and smoothed... 10-day
averaging window" (paper Fig 3 caption), so this is a time-mean / seasonal-
cycle test, not point-wise.

## LLC truth comparison

`scripts/llc_truth_compare.py` reads `data/paper/llcGoes_gradT_trunc.nc`
(paper LLC training file, 8230 hourly frames, lat 34-45 N, lon -80 to -60 W,
2011-09 to 2012-09). Restricts to the LLC↔model overlap region: lat 38-41 N,
lon -63.82 to -60.0 W (a ~3.8° wide strip). Reads every 8th frame for speed
(~1000 samples).

## Results in GS overlap region (lat 38-41 N, lon -63.82 to -60.0 W)

Time-mean over respective time windows:

| Source | mean \|U\| | <U> | <V> | MKE | EKE | EKE/MKE |
|--------|----------|------|------|-----|-----|---------|
| LLC truth (2011-2012) | 0.295 m/s | +0.118 | +0.014 | 0.0071 | 0.0579 | 8.21 |
| MODEL c_spec=0.2 (2023) | 0.211 m/s | +0.138 | -0.018 | 0.0112 | 0.0312 | 2.80 |
| AVISO L4 1/8 deg (2023) | 0.623 m/s | +0.272 | +0.102 | 0.0422 | 0.2387 | 5.65 |

- **MODEL is ~28% lower than LLC truth** in mean speed.
- **LLC truth is ~53% lower than AVISO** in mean speed.
- MODEL has lower EKE/MKE ratio than LLC (less eddy dominance).

## Reconciliation with the paper

The GOFLOW paper documents three of the four discrepancies we observed:

1. **MODEL under-predicts vs LLC4320**: Methods, p9: *"GOFLOW under predicts
   high-velocity-gradient magnitudes in comparison to LLC4320"* (cites
   Extended Data Figs 2, 6).

2. **The under-prediction is worse in winter**: Methods, p9: *"The comparison
   with LLC4320 shows more accurate results during summer months relative to
   winter months."* Our MODEL MKE peak in January is the paper's exact
   description: *"shallower spectral slopes and larger skewness in winter,
   when SMCs are more energetic"* (Discussion, p5).

3. **LLC4320 separates the GS too far south vs AVISO**: Discussion, p5:
   *"the LLC4320 solution has been shown to separate the current too far
   south relative to AVISO-based estimates of the mean dynamic topography
   and associated jet path."* This explains the LLC↔AVISO factor-of-2 gap.

4. **AVISO is intrinsically smooth**: Fig 3 caption: *"the AVISO-derived
   vorticity field appears diffused and smoothed, a result of its 10-day
   averaging window..."*

The paper's own validation against in-situ ADCP/drifters (Extended Data Fig 1b)
gives RMSE 0.27 m/s for u and R² = 0.69 (u), 0.73 (v) over all-seasons 2023
data — that is the apples-to-apples test, and GOFLOW does well there.

## Seasonal cycle

| Metric | MODEL peak | AVISO peak | Kang+ 2016 | LLC truth peak |
|--------|------------|------------|------------|----------------|
| EKE primary | April | March | May | June |
| EKE secondary | September | September | September | (single broad peak) |
| MKE peak | January | May | summer | February+June (bimodal) |
| Mean \|U\| peak | March | October | -- | June |

EKE bimodal April + September matches Kang+ 2016 (May + September) within a
month. MKE January peak matches paper's "winter > summer" finding.

LLC truth (2011-2012) has *its own* seasonal cycle, distinct from AVISO 2023
and MODEL 2023 — partly inter-annual variability, partly LLC's path bias.

## Caveats and what this validation does *not* prove

- **Single year (2023)** of model output. Inter-annual GS variability is real;
  one year is not climatology.
- **Limited longitude range** (lon -64 to -57.5 W). The Cape Hatteras
  separation region (-75 to -65 W) was not run for inference and is a
  dynamically distinct regime.
- **No direct point-wise comparison**: time-mean and seasonal cycle only.
  The paper's ADCP/drifter scatter is the right test for point-wise skill;
  we did not reproduce that test for our 2023 inference.

## Reproduction commands

```bash
# Pull AVISO if not present
copernicusmarine subset \
  --dataset-id cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D \
  --variable ugos --variable vgos --variable adt \
  --start-datetime 2023-01-01T00:00:00 --end-datetime 2023-12-31T23:59:59 \
  --minimum-longitude -80 --maximum-longitude -58 \
  --minimum-latitude 25 --maximum-latitude 42 \
  --output-directory data/aviso/ \
  --output-filename aviso_l4_duacs_2023_25-42N_80-58W.nc

# Stream model climatology (or use existing cache)
python3 scripts/climatology_v2.py

# Replot with fixed front detection (uses cache)
python3 scripts/climatology_replot.py

# AVISO vs model comparison
python3 scripts/aviso_compare.py

# LLC truth vs model vs AVISO in overlap region
python3 scripts/llc_truth_compare.py
```
