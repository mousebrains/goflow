"""Compare two soak runs (e.g., physics vs paper layout).

Reads the train_goflow.py stdout logs from each run, parses per-epoch L1,
spec, and R^2, and plots them side-by-side. Optionally also plots the
satellite-prediction NetCDFs head-to-head at one timestep.

Run:
    python preprocess/soak_compare.py \
        --runA data/soak_physics --labelA physics \
        --runB data/soak_paper   --labelB paper \
        --out data/soak_compare.png
"""
import argparse
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt


# Two log line shapes train_goflow.py prints:
#   "Epoch 1: L1=0.3380, spec=41.7051"
#   "Epoch 1/1 | R^2: -0.7153 (best: -0.7153) | Spec: 44.4347 (best: 44.4347)"
RE_TRAIN = re.compile(r'Epoch\s+(\d+):\s+L1=([\d.eE+-]+),\s+(?:spec|grad)=([\d.eE+-]+)')
RE_EVAL  = re.compile(
    r'Epoch\s+(\d+)/\d+\s+\|\s+R..?:\s+([-\d.eE+]+)\s+\(best:.*?\|\s+Spec:\s+([\d.eE+-]+)')


def parse_run(run_dir):
    """Return dict of arrays keyed by 'epoch','l1','aux','r2','spec'."""
    log_files = glob.glob(os.path.join(run_dir, '*.log'))
    if not log_files:
        raise SystemExit(f'No *.log in {run_dir}')
    log = sorted(log_files, key=os.path.getmtime)[-1]
    print(f'Parsing {log}')

    train = {}      # epoch -> (l1, aux)
    eval_ = {}      # epoch -> (r2, spec)
    with open(log) as f:
        for line in f:
            m = RE_TRAIN.search(line)
            if m:
                e = int(m.group(1))
                train[e] = (float(m.group(2)), float(m.group(3)))
                continue
            m = RE_EVAL.search(line)
            if m:
                e = int(m.group(1))
                eval_[e] = (float(m.group(2)), float(m.group(3)))

    epochs = sorted(set(train) | set(eval_))
    out = dict(epoch=[], l1=[], aux=[], r2=[], spec=[])
    for e in epochs:
        out['epoch'].append(e)
        out['l1'].append(train.get(e, (np.nan, np.nan))[0])
        out['aux'].append(train.get(e, (np.nan, np.nan))[1])
        out['r2'].append(eval_.get(e, (np.nan, np.nan))[0])
        out['spec'].append(eval_.get(e, (np.nan, np.nan))[1])
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runA', required=True, help='Path to run A directory')
    ap.add_argument('--runB', required=True, help='Path to run B directory')
    ap.add_argument('--labelA', default='A')
    ap.add_argument('--labelB', default='B')
    ap.add_argument('--out', default='data/soak_compare.png')
    args = ap.parse_args()

    A = parse_run(args.runA)
    B = parse_run(args.runB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, key, ylabel, log in [
        (axes[0], 'l1',   'Train L1 loss', False),
        (axes[1], 'spec', 'Test spectral loss', True),
        (axes[2], 'r2',   r'Test gradient $R^2$', False),
    ]:
        ax.plot(A['epoch'], A[key], 'o-', label=args.labelA, markersize=4)
        ax.plot(B['epoch'], B[key], 's-', label=args.labelB, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f'GOFLOW soak: {args.labelA} vs {args.labelB}', fontsize=12)
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'Saved {args.out}  ({os.path.getsize(args.out)/1e3:.0f} KB)')

    print('\nFinal-epoch summary:')
    for label, run in [(args.labelA, A), (args.labelB, B)]:
        print(f'  {label:>10s}: epochs={len(run["epoch"])}  '
              f'final L1={run["l1"][-1]:.4f}  spec={run["spec"][-1]:.4f}  '
              f'R2={run["r2"][-1]:.4f}  '
              f'best R2={np.nanmax(run["r2"]):.4f}')


if __name__ == '__main__':
    main()
