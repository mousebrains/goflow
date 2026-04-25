"""Two-stage GOFLOW training driver: L1-only -> L1+spectral.

Per the paper Methods, the network is trained first with pointwise L1 only
(c_spec=0), then fine-tuned with the spectral loss term (c_spec=0.2) starting
from the stage-1 weights. The stage chaining is also supported by
train_goflow.py's auto-discovery, but this wrapper makes the workflow a single
explicit command and lets you override the stage-2 weight.

Run:
    python train_twostage.py --llc_file data/llc.nc --goes_file data/goes.nc
    python train_twostage.py --llc_file ... --c_spec_stage2 0.3 --epochs_stage2 75

All other args (--llc_file, --goes_file, --output_dir, --model, --nbase,
--epochs, etc.) are forwarded to both stages.
"""
import argparse
import subprocess
import sys
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--c_spec_stage2', type=float, default=0.2,
                    help='Spectral-loss weight for stage 2 (default: 0.2)')
    ap.add_argument('--epochs_stage1', type=int, default=None,
                    help='Override stage-1 epoch count (default: train_goflow.py decides)')
    ap.add_argument('--epochs_stage2', type=int, default=None,
                    help='Override stage-2 epoch count')
    ap.add_argument('--skip_stage1', action='store_true',
                    help='Skip stage 1 (assumes the *_0.0cs.pth checkpoint already exists)')
    args, passthrough = ap.parse_known_args()

    if any(a.startswith('--c_spec') and not a.startswith('--c_spec_stage')
           for a in passthrough):
        raise SystemExit('Do not pass --c_spec to train_twostage.py; use '
                         '--c_spec_stage2 instead.')
    if any(a.startswith('--init_from') for a in passthrough):
        raise SystemExit('Do not pass --init_from to train_twostage.py; '
                         'stage-2 picks up stage-1 weights automatically.')

    here = os.path.dirname(os.path.abspath(__file__))
    train_py = os.path.join(here, 'train_goflow.py')

    if not args.skip_stage1:
        cmd1 = [sys.executable, train_py, '--c_spec', '0.0'] + passthrough
        if args.epochs_stage1 is not None:
            cmd1 += ['--epochs', str(args.epochs_stage1)]
        print(f'\n=== Stage 1: L1-only ===\n  {" ".join(cmd1)}\n')
        subprocess.check_call(cmd1)
    else:
        print('Skipping stage 1 (--skip_stage1)')

    cmd2 = [sys.executable, train_py,
            '--c_spec', str(args.c_spec_stage2)] + passthrough
    if args.epochs_stage2 is not None:
        cmd2 += ['--epochs', str(args.epochs_stage2)]
    print(f'\n=== Stage 2: L1 + spectral (c_spec={args.c_spec_stage2}) ===\n'
          f'  {" ".join(cmd2)}\n')
    subprocess.check_call(cmd2)

    print('\nTwo-stage training complete.')


if __name__ == '__main__':
    main()
