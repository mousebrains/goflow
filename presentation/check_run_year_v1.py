"""Check on the year-long two-stage GOFLOW training (data/run_year_v1/).

Reports process state, current stage, epoch progress, latest metrics, and
artifact existence. Runs the stage-1 quicklook automatically once stage 1
is complete. Safe to invoke any time during or after the run.

Run:
    python presentation/check_run_year_v1.py
"""
import os
import re
import subprocess
import sys
from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN_DIR = os.path.join(ROOT, 'data', 'run_year_v1')
LOG = os.path.join(RUN_DIR, 'run_year_v1.log')
START_FILE = os.path.join(RUN_DIR, 'start.txt')
STAGE1_PTH = os.path.join(RUN_DIR, 'lgt_unet16_1_3_0.0cs.pth')
STAGE2_PTH = os.path.join(RUN_DIR, 'lgt_unet16_1_3_0.2cs.pth')
STAGE1_TEST = os.path.join(RUN_DIR, 'test_lgt_unet16_0.0cspec.nc')
STAGE2_TEST = os.path.join(RUN_DIR, 'test_lgt_unet16_0.2cspec.nc')
QUICKLOOK1 = os.path.join(RUN_DIR, 'stage1_quicklook.png')
QUICKLOOK_DIFF = os.path.join(RUN_DIR, 'stage1_vs_stage2.png')


RE_STAGE = re.compile(r'^=== Stage (\d+):')
RE_TRAIN = re.compile(r'^Epoch (\d+):\s+L1=([\d.eE+-]+),\s+(?:spec|grad)=([\d.eE+-]+)')
RE_EVAL  = re.compile(r'^Epoch (\d+)/\d+\s+\|\s+R..?:\s+([-\d.eE+]+).*Spec:\s+([\d.eE+-]+)')


def parse_log(path):
    """Parse epoch metrics. Stage detection: explicit '=== Stage N ===' banner
    if present (newer runs); otherwise infer that an Epoch 1 line that comes
    after any Epoch >1 marks the start of stage 2 (the train_twostage chain
    resets the epoch counter)."""
    if not os.path.exists(path):
        return None
    cur_stage = 1
    seen_higher_epoch = False
    by_stage = {1: {'train': [], 'eval': []}, 2: {'train': [], 'eval': []}}
    completion = False
    error_lines = []
    with open(path, errors='replace') as f:
        for line in f:
            m = RE_STAGE.match(line)
            if m:
                cur_stage = int(m.group(1))
                seen_higher_epoch = False
                continue
            if 'Two-stage training complete' in line:
                completion = True
                continue
            if 'Traceback' in line or 'Error' in line[:20]:
                error_lines.append(line.rstrip())
            m = RE_TRAIN.match(line)
            if m:
                ep = int(m.group(1))
                if ep == 1 and seen_higher_epoch and cur_stage == 1:
                    cur_stage = 2  # epoch counter reset = stage 2 started
                    seen_higher_epoch = False
                if ep > 1:
                    seen_higher_epoch = True
                by_stage[cur_stage]['train'].append(
                    (ep, float(m.group(2)), float(m.group(3))))
                continue
            m = RE_EVAL.match(line)
            if m:
                by_stage[cur_stage]['eval'].append(
                    (int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return dict(stage=cur_stage, by_stage=by_stage,
                completion=completion, error_lines=error_lines)


def process_state():
    """Try to find the train_twostage process. Returns (pid, etime, alive)."""
    try:
        p = subprocess.run(
            ['pgrep', '-f', 'train_twostage.py'],
            capture_output=True, text=True, timeout=5)
        pids = [x for x in p.stdout.strip().split('\n') if x]
        if not pids:
            return None, None, False
        pid = pids[0]
        ps = subprocess.run(
            ['ps', '-p', pid, '-o', 'pid=,etime=,stat='],
            capture_output=True, text=True, timeout=5)
        return pid, ps.stdout.strip(), True
    except Exception as e:
        return None, str(e), False


def correlation(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 10:
        return float('nan')
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def quicklook_stage1(test_nc, out_png):
    ds = xr.open_dataset(test_nc, decode_times=False)
    t = ds.sizes.get('time', 1) // 2
    panels = [
        ('U_inp',    'truth U',     'RdBu_r',  -1.0, 1.0),
        ('V_inp',    'truth V',     'RdBu_r',  -1.0, 1.0),
        ('vort_inp', 'truth vort',  'RdBu_r',  -2e-5, 2e-5),
        ('U_out',    'pred U',      'RdBu_r',  -1.0, 1.0),
        ('V_out',    'pred V',      'RdBu_r',  -1.0, 1.0),
        ('vort_out', 'pred vort',   'RdBu_r',  -2e-5, 2e-5),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (var, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        if var not in ds.data_vars:
            ax.set_title(f'{title} (missing)'); ax.set_axis_off(); continue
        a = ds[var].isel(time=t).values
        ax.imshow(a, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Stage 1 (L1-only) test predictions — t={t}', fontsize=12)
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)

    # Pearson over the whole test set
    r_u = correlation(ds.U_inp.values, ds.U_out.values)
    r_v = correlation(ds.V_inp.values, ds.V_out.values)
    r_w = correlation(ds.vort_inp.values, ds.vort_out.values)
    ds.close()
    return r_u, r_v, r_w


def main():
    print('=' * 72)
    print(f'GOFLOW year-long run check  ({datetime.now().isoformat(timespec="seconds")})')
    print('=' * 72)

    if os.path.exists(START_FILE):
        with open(START_FILE) as f:
            print(f'Started: {f.read().strip()}')

    pid, etime, alive = process_state()
    print(f'Process: pid={pid}  etime={etime}  alive={alive}')

    parsed = parse_log(LOG)
    if parsed is None:
        sys.exit(f'No log at {LOG}')
    if parsed['error_lines']:
        print('\nWARNING: error lines in log:')
        for ln in parsed['error_lines'][-5:]:
            print(f'  {ln}')

    print(f'\nLog: stage={parsed["stage"]}  completion={parsed["completion"]}')
    for s in (1, 2):
        tr = parsed['by_stage'][s]['train']
        ev = parsed['by_stage'][s]['eval']
        if tr or ev:
            last_tr = tr[-1] if tr else None
            last_ev = ev[-1] if ev else None
            print(f'  Stage {s}: train epochs={len(tr)}, eval epochs={len(ev)}')
            if last_tr:
                print(f'    last train: epoch {last_tr[0]}  L1={last_tr[1]:.4f}  '
                      f'aux={last_tr[2]:.4f}')
            if last_ev:
                print(f'    last eval:  epoch {last_ev[0]}  R^2={last_ev[1]:.4f}  '
                      f'spec={last_ev[2]:.4f}')

    print('\nArtifacts:')
    for label, path in [('stage1 .pth', STAGE1_PTH),
                        ('stage1 test', STAGE1_TEST),
                        ('stage2 .pth', STAGE2_PTH),
                        ('stage2 test', STAGE2_TEST)]:
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1e6
            mt = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%H:%M:%S')
            print(f'  {label:14s} {sz:>7.1f} MB  mtime {mt}  {path}')
        else:
            print(f'  {label:14s} (not yet)        {path}')

    if os.path.exists(STAGE1_TEST) and not os.path.exists(QUICKLOOK1):
        print(f'\nGenerating stage-1 quicklook -> {QUICKLOOK1}')
        try:
            r_u, r_v, r_w = quicklook_stage1(STAGE1_TEST, QUICKLOOK1)
            print(f'  Pearson on test set:  U={r_u:.3f}  V={r_v:.3f}  vort={r_w:.3f}')
        except Exception as e:
            print(f'  quicklook failed: {e}')

    print('\nNext:')
    if parsed['completion']:
        print('  Both stages complete. Suggest stage1-vs-stage2 comparison.')
    elif os.path.exists(STAGE1_PTH) and not os.path.exists(STAGE2_PTH):
        print('  Stage 2 in progress. Re-run this script in ~3 hr.')
    elif not os.path.exists(STAGE1_PTH):
        print('  Stage 1 still training. Re-run in ~1 hr to refresh.')


if __name__ == '__main__':
    main()
