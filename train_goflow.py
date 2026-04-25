"""
GOFLOW Training Script
======================
Training pipeline for GOFLOW (Geostationary Ocean Flow) models.

Usage:
    python train_goflow.py --cuda 0 --model unet --c_spec 0.5 --nbase 16

Author: Kaushik (UCLA Atmospheric and Oceanic Sciences)
"""

import os
import gc
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from netCDF4 import Dataset as NCDataset
from tqdm import tqdm
from copy import deepcopy

# Local imports
from goflow_core import (
    dx_kernel, dy_kernel,
    compute_velocity_gradients, compute_derived_fields,
    create_tukey_window, create_boundary_mask,
    gradient_loss, compute_gradient_r2, to_numpy,
    load_datasets, create_dataloaders,
    initialize_model, save_model, load_model,
    get_model_string, count_parameters
)
from spectral_loss import spectral_loss
from utils import cosineSGDR
from dataSST import SatelliteDataset, writeGridSat
from writenc import ncCreate, addVal


# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GOFLOW model')
    
    # Device and model selection
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'samudra0', 'samudraR', '2layer'],
                        help='Model architecture')
    parser.add_argument('--nbase', type=int, default=16, help='Base channels for UNet')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for 2layer CNN')
    
    # Loss configuration
    parser.add_argument('--c_spec', type=float, default=0.0,
                        help='Spectral/gradient loss weight (0-1)')
    parser.add_argument('--use_grad_loss', action='store_true',
                        help='Use gradient loss instead of spectral loss')
    parser.add_argument('--init_from', type=str, default=None,
                        help='Path to a .pth checkpoint to initialize weights from. '
                             'Overrides the c_spec>0 auto-discovery convention.')
    parser.add_argument('--layout', type=str, default='auto',
                        choices=['auto', 'physics', 'paper', 'geometric', 'quadrant'],
                        help='Training-tile layout. auto picks based on grid size.')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: 100 for c_spec=0, 50 otherwise)')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--tcycle', type=int, default=5, help='Cosine annealing cycle length')
    
    # Data paths
    parser.add_argument('--llc_file', type=str, default='llcGoes_gradT_trunc.nc',
                        help='LLC training data file')
    parser.add_argument('--goes_file', type=str, default='GS_BT_NESMA2023_HiRes_SUBSECTION_grad_mask.nc',
                        help='GOES satellite data file')
    parser.add_argument('--output_dir', type=str, default='./ncfiles/',
                        help='Output directory for NetCDF files')
    
    # Data parameters
    parser.add_argument('--nframes', type=int, default=3, help='Number of input frames')
    parser.add_argument('--step0', type=int, default=1, help='Time step stride')
    parser.add_argument('--pm', type=float, default=5.0, help='X grid metric')
    parser.add_argument('--pn', type=float, default=5.0, help='Y grid metric')
    
    return parser.parse_args()


def pick_layout(layout, Ny_llc, Nx_llc):
    """Select training-tile indices by name, with sensible auto fallback.

    Returns (train_inds, test_inds). All tiles are 256x256.
    """
    big = (Ny_llc >= 800 and Nx_llc >= 1500)
    medium = (Ny_llc >= 512 and Nx_llc >= 768)
    if layout == 'auto':
        layout = 'physics' if big else ('paper' if medium else 'quadrant')

    if layout == 'physics':
        if not big:
            raise SystemExit(
                f'--layout physics requires LLC >=800x1500 (got {Ny_llc}x{Nx_llc}).')
        # 17x30 deg Gulf Stream box (944x1666). Five 256x256 train tiles span
        # distinct submesoscale regimes; test tile is the south-wall mix.
        #   T1 Sargasso SW       (25-30 N, -75..-70 W)
        #   T2 Sargasso E        (25-30 N, -60..-55 W)
        #   T3 NW slope water    (36-40 N, -80..-75 W)
        #   T4 Gulf Stream jet   (36-40 N, -70..-65 W)
        #   T5 Stream extension  (36-40 N, -60..-55 W)
        #   TEST south-wall mix  (30-35 N, -70..-65 W)
        train_inds = [
            (0,   256, 256,  512),
            (0,   256, 1100, 1356),
            (600, 856, 0,    256),
            (600, 856, 550,  806),
            (600, 856, 1100, 1356),
        ]
        test_inds = (300, 556, 550, 806)
        print('Layout: physics (17x30 deg Gulf Stream, 5-tile)')
    elif layout == 'paper':
        if not medium:
            raise SystemExit(
                f'--layout paper requires LLC >=512x768 (got {Ny_llc}x{Nx_llc}).')
        # Original paper layout: all tiles in the south half of the box.
        train_inds = [
            (0,   256, 256, 512),
            (0,   256, 512, 768),
            (256, 512, 256, 512),
            (256, 512, 512, 768),
            (256, 512, Nx_llc - 256, Nx_llc),
        ]
        test_inds = (0, 256, Nx_llc - 256, Nx_llc)
        print('Layout: paper (south-half tiles)')
    elif layout == 'geometric':
        if Ny_llc < 256 or Nx_llc < 256:
            raise SystemExit(
                f'--layout geometric requires LLC >=256x256 (got {Ny_llc}x{Nx_llc}).')
        # Evenly-spaced 5 train + 1 test tiles, no physics knowledge.
        rows = max(1, Ny_llc // 256)
        cols = max(1, Nx_llc // 256)
        cells = []
        for r in range(rows):
            j0 = r * (Ny_llc - 256) // max(1, rows - 1) if rows > 1 else 0
            for c in range(cols):
                i0 = c * (Nx_llc - 256) // max(1, cols - 1) if cols > 1 else 0
                cells.append((j0, j0 + 256, i0, i0 + 256))
        if len(cells) < 6:
            raise SystemExit(f'Need at least 6 cells of 256x256 (got {len(cells)}).')
        step = len(cells) // 6
        train_inds = [cells[i * step] for i in range(1, 6)]
        test_inds = cells[len(cells) // 2]
        if test_inds in train_inds:
            idx = train_inds.index(test_inds)
            for cand in cells:
                if cand not in train_inds and cand != test_inds:
                    train_inds[idx] = cand
                    break
        print(f'Layout: geometric ({rows}x{cols} grid, 5 train + 1 test)')
    elif layout == 'quadrant':
        # Pilot layout for small tiles
        hy, hx = Ny_llc // 2, Nx_llc // 2
        if hy < 16 or hx < 16:
            raise SystemExit(f'Quadrant layout requires LLC >=32x32 (got {Ny_llc}x{Nx_llc}).')
        train_inds = [
            (0, hy, 0, hx),
            (0, hy, hx, Nx_llc),
            (hy, Ny_llc, 0, hx),
        ]
        test_inds = (hy, Ny_llc, hx, Nx_llc)
        print(f'Layout: quadrant (LLC {Ny_llc}x{Nx_llc})')
    else:
        raise SystemExit(f'Unknown layout: {layout!r}')
    return train_inds, test_inds


def setup_device(cuda_idx: int) -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_idx}')
        torch.cuda.set_device(cuda_idx)
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    gc.collect()
    print(f'Device: {device}')
    return device


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    mask: torch.Tensor,
    tukey_window: torch.Tensor,
    c_spec: float,
    use_grad_loss: bool = False
) -> tuple[float, float]:
    """
    Run one training epoch.
    
    Returns:
        Tuple of (l1_loss, auxiliary_loss) from first batch for logging.
    """
    model.train()
    first_batch_losses = None

    for ib, batch in enumerate(tqdm(train_loader, desc='Training')):
        # Dataset returns (x, y, valid) when its target is masked; older code
        # paths still return (x, y) — accept both.
        if len(batch) == 3:
            x, y, valid = batch
            valid = valid.to(kernel_x.device)
        else:
            x, y = batch
            valid = None
        x, y = x.to(kernel_x.device), y.to(kernel_x.device)

        y_pred = model(x)

        # Pointwise L1 with boundary mask + (when available) per-pixel validity
        # mask, so NaN-filled coastline targets don't bias the model toward zero.
        if valid is not None:
            weight = (mask[None, None, :, :] * valid.unsqueeze(1)).expand_as(y)
            loss_l1 = (y_pred - y).abs().mul(weight).sum() / weight.sum().clamp_min(1e-6)
        else:
            loss_l1 = criterion(
                y.squeeze() * mask[None, None, :, :],
                y_pred.squeeze() * mask[None, None, :, :]
            )

        # Auxiliary loss (gradient or spectral). Targets are zero-filled on
        # land; spectral loss tolerates this for tiles with small mask area.
        if use_grad_loss:
            loss_aux = gradient_loss(y_pred.squeeze(), y.squeeze(), criterion,
                                     mask, kernel_x, kernel_y)
        else:
            loss_aux = spectral_loss(y_pred, y, tukey_window)
        
        # Combined loss
        loss = (1 - c_spec) * loss_l1 + c_spec * loss_aux
        
        # Store first batch losses for logging
        if ib == 0:
            first_batch_losses = (loss_l1.item(), loss_aux.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return first_batch_losses


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    mask: torch.Tensor,
    tukey_window: torch.Tensor
) -> tuple[float, float]:
    """
    Evaluate model on test set.
    
    Returns:
        Tuple of (mean_r2, mean_spectral_loss)
    """
    model.eval()
    total_r2 = 0.0
    total_spec_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            if len(batch) == 3:
                x, y, valid = batch
                valid = valid.to(kernel_x.device)
            else:
                x, y = batch
                valid = None
            x, y = x.to(kernel_x.device), y.to(kernel_x.device)
            y_pred = model(x)

            # Spectral loss
            spec_loss = spectral_loss(y_pred, y, tukey_window)

            # R² on gradient fields (vorticity + strain). Boundary mask only;
            # validity-mask-aware R² is left as a TODO (sklearn r2_score path
            # would need broadcasting + careful zero-variance handling).
            r2 = compute_gradient_r2(y, y_pred, kernel_x, kernel_y, mask)

            total_r2 += r2
            total_spec_loss += spec_loss.item()
            count += 1
    
    return total_r2 / count, total_spec_loss / count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: argparse.Namespace,
    device: torch.device
) -> tuple[nn.Module, np.ndarray]:
    """
    Full training loop with checkpointing and evaluation.
    
    Returns:
        Tuple of (best_model, r2_history)
    """
    # Setup derivative kernels
    kernel_x = dx_kernel(config.pm).to(device)
    kernel_y = dy_kernel(config.pn).to(device)
    
    # Will be initialized on first batch
    mask = None
    tukey_window = None
    
    # Get model string for filenames
    model_str = get_model_string(config.model, config.nbase, config.kernel_size, config.use_grad_loss)
    
    # Tracking
    best_r2 = -1000
    best_spec = 1000
    r2_history = np.zeros(config.epochs)
    best_model = None
    
    for epoch in range(config.epochs):
        # Learning rate scheduling
        lr = cosineSGDR(optimizer, epoch, T0=config.tcycle, eta_min=0, 
                        eta_max=config.lr, scheme='constant')
        
        # Initialize mask/window on first epoch using data shape
        if mask is None:
            sample_batch = next(iter(train_loader))
            sample_y = sample_batch[1]
            shape = sample_y.shape[-2:]
            mask = create_boundary_mask(shape).to(device)
            tukey_window = create_tukey_window(shape).to(device)
        
        # Train one epoch
        l1_loss, aux_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            kernel_x, kernel_y, mask, tukey_window,
            config.c_spec, config.use_grad_loss
        )
        loss_type = 'grad' if config.use_grad_loss else 'spec'
        print(f'Epoch {epoch+1}: L1={l1_loss:.4f}, {loss_type}={aux_loss:.4f}')
        
        # Evaluate
        r2, spec_loss = evaluate_model(model, test_loader, kernel_x, kernel_y, mask, tukey_window)
        r2_history[epoch] = r2
        
        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = deepcopy(model)
            
            # Save checkpoint and run inference
            checkpoint_path = f'{model_str}_{config.step0}_{config.nframes}_{config.c_spec}cs.pth'
            save_model(best_model, checkpoint_path)
            
            # Write test results
            write_test_results(
                epoch, best_model, test_loader, kernel_x, kernel_y,
                config.c_spec, model_str, config.output_dir
            )
            
            # Process satellite data
            out_val, grad_val, sst_val = run_satellite_inference(
                best_model, config.goes_file, config.valid_inds,
                config.pm, config.pn
            )
            
            # Write satellite predictions
            output_file = f'preds_{model_str}_{config.step0}_{config.nframes}_{config.c_spec}cs_{os.path.splitext(os.path.basename(config.goes_file))[0]}.nc'
            write_satellite_netcdf(output_file, out_val, grad_val, sst_val,
                                   config.valid_inds, config.goes_file)
        
        if spec_loss < best_spec:
            best_spec = spec_loss
        
        print(f'Epoch {epoch+1}/{config.epochs} | R²: {r2:.4f} (best: {best_r2:.4f}) | '
              f'Spec: {spec_loss:.4f} (best: {best_spec:.4f})')
    
    return best_model, r2_history


# =============================================================================
# Satellite Data Processing
# =============================================================================

def run_satellite_inference(
    model: nn.Module,
    goes_file: str,
    valid_inds: tuple,
    pm: float,
    pn: float,
    batch_size: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on GOES satellite data.
    
    Returns:
        Tuple of (velocities, gradient_fields, sst_data)
    """
    device = next(model.parameters()).device
    kernel_x = dx_kernel(pm).to(device)
    kernel_y = dy_kernel(pn).to(device)
    
    goes_dataset = SatelliteDataset(goes_file, ['log_gradT'], valid_inds, train=False)
    nw = 0 if sys.platform == 'darwin' else 4
    goes_loader = DataLoader(goes_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    
    out_list = []
    grad_list = []
    sst_list = []
    
    model.eval()
    with torch.no_grad():
        for sst in tqdm(goes_loader, desc='Satellite inference'):
            # Store input SST
            sst_list.append(sst[:, 1, :, :].cpu().numpy()[:, None, :, :])
            
            sst = sst.to(device)
            out = model(sst)
            out_list.append(out.cpu().numpy())
            
            # Compute gradient fields
            ux, uy, vx, vy = compute_velocity_gradients(out, kernel_x, kernel_y)
            vort, div, strain = compute_derived_fields(ux, uy, vx, vy)
            grad_list.append(torch.stack((vort, div, strain), dim=1).cpu().numpy())
    
    out_val = np.concatenate(out_list, axis=0)
    grad_val = np.concatenate(grad_list, axis=0)
    sst_val = np.concatenate(sst_list, axis=0).squeeze()
    
    return out_val, grad_val, sst_val


# =============================================================================
# NetCDF Output
# =============================================================================

def write_satellite_netcdf(
    output_file: str,
    out_val: np.ndarray,
    grad_val: np.ndarray,
    sst_val: np.ndarray,
    valid_inds: tuple,
    goes_file: str
):
    """Write satellite prediction results to NetCDF."""
    nt, _, ny, nx = out_val.shape
    print(f'Writing {output_file}: shape=({nt}, {ny}, {nx})')
    
    with NCDataset(goes_file, 'r') as nch:
        varnames = ['U', 'V', 'Vorticity', 'Divergence', 'Strain', 'BT', 'loggrad_BT']
        nc = ncCreate(output_file, nx, ny, varnames, dt=2)
        
        for it in tqdm(range(nt), desc='Writing NetCDF'):
            BT = nch.variables['BT'][it + 12, 
                                     valid_inds[0]:valid_inds[1],
                                     valid_inds[2]:valid_inds[3]]
            addVal(nc, 'U', out_val[it, 0, :, :], it)
            addVal(nc, 'V', out_val[it, 1, :, :], it)
            addVal(nc, 'Vorticity', grad_val[it, 0, :, :], it)
            addVal(nc, 'Divergence', grad_val[it, 1, :, :], it)
            addVal(nc, 'Strain', grad_val[it, 2, :, :], it)
            addVal(nc, 'BT', BT, it)
            addVal(nc, 'loggrad_BT', sst_val[it, :, :], it)
        
        nc.close()
    
    writeGridSat(goes_file, output_file, valid_inds)


def write_test_results(
    epoch: int,
    model: nn.Module,
    test_loader: DataLoader,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    c_spec: float,
    model_str: str,
    output_dir: str = './ncfiles/'
):
    """Write test set predictions and gradient fields to NetCDF."""
    model.eval()
    device = next(model.parameters()).device
    
    inputs_list = []
    outputs_list = []
    targets_list = []
    true_grads_list = []
    pred_grads_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Processing test set'):
            if len(batch) == 3:
                x, y_true, _valid = batch
            else:
                x, y_true = batch
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            
            # Compute gradients
            ux_true, uy_true, vx_true, vy_true = compute_velocity_gradients(y_true, kernel_x, kernel_y)
            ux_pred, uy_pred, vx_pred, vy_pred = compute_velocity_gradients(y_pred, kernel_x, kernel_y)
            
            vort_true, div_true, strain_true = compute_derived_fields(ux_true, uy_true, vx_true, vy_true)
            vort_pred, div_pred, strain_pred = compute_derived_fields(ux_pred, uy_pred, vx_pred, vy_pred)
            
            inputs_list.append(x[:, 1, :, :].cpu().numpy())
            outputs_list.append(y_pred.cpu().numpy())
            targets_list.append(y_true.cpu().numpy())
            true_grads_list.append(torch.stack((vort_true, div_true, strain_true), dim=1).cpu().numpy())
            pred_grads_list.append(torch.stack((vort_pred, div_pred, strain_pred), dim=1).cpu().numpy())
    
    # Concatenate batches
    inputs = np.concatenate(inputs_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    true_grads = np.concatenate(true_grads_list, axis=0)
    pred_grads = np.concatenate(pred_grads_list, axis=0)
    
    # Write NetCDF
    os.makedirs(output_dir, exist_ok=True)
    nc_filename = os.path.join(output_dir, f'test_{model_str}_{c_spec}cspec.nc')
    
    Nt, Ny, Nx = inputs.shape
    varlist = ['gradT', 'U_inp', 'V_inp', 'vort_inp', 'div_inp', 'strain_inp',
               'U_out', 'V_out', 'vort_out', 'div_out', 'strain_out']
    
    with ncCreate(nc_filename, Nx, Ny, varlist) as nc:
        nc.variables['gradT'][:] = inputs
        nc.variables['U_inp'][:] = targets[:, 0, :, :]
        nc.variables['V_inp'][:] = targets[:, 1, :, :]
        nc.variables['U_out'][:] = outputs[:, 0, :, :]
        nc.variables['V_out'][:] = outputs[:, 1, :, :]
        nc.variables['vort_inp'][:] = true_grads[:, 0, :, :]
        nc.variables['div_inp'][:] = true_grads[:, 1, :, :]
        nc.variables['strain_inp'][:] = true_grads[:, 2, :, :]
        nc.variables['vort_out'][:] = pred_grads[:, 0, :, :]
        nc.variables['div_out'][:] = pred_grads[:, 1, :, :]
        nc.variables['strain_out'][:] = pred_grads[:, 2, :, :]
        
        nc.description = f'Test set results for epoch {epoch}'
        nc.input_field = 'SST gradient (middle time step)'
        nc.output_fields = 'Vorticity, Divergence, Strain (target and predicted)'
    
    print(f'Test results written to {nc_filename}')


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    # Parse arguments (also support legacy positional args)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        # Legacy mode: cuda_count c_spec model_opt [nbase]
        args = argparse.Namespace(
            cuda=int(sys.argv[1]) if len(sys.argv) > 1 else 0,
            c_spec=float(sys.argv[2]) if len(sys.argv) > 2 else 0.0,
            model=sys.argv[3] if len(sys.argv) > 3 else 'unet',
            nbase=int(sys.argv[4]) if len(sys.argv) > 4 else 16,
            kernel_size=5,
            use_grad_loss=False,
            epochs=None,
            lr=0.001,
            tcycle=5,
            llc_file='llcGoes_gradT_trunc.nc',
            goes_file='GS_BT_NESMA2023_HiRes_SUBSECTION_grad_mask.nc',
            output_dir='./ncfiles/',
            nframes=3,
            step0=1,
            pm=5.0,
            pn=5.0,
        )
        print('Using legacy argument mode')
    else:
        args = parse_args()
    
    # Setup device
    device = setup_device(args.cuda)
    
    # Determine epochs if not specified
    if args.epochs is None:
        args.epochs = 100 if args.c_spec == 0 else 50
    
    # Get grid dimensions from GOES and LLC files
    with NCDataset(args.goes_file, 'r') as nc:
        Nx = nc.dimensions['lon'].size
        Ny = nc.dimensions['lat'].size
    with NCDataset(args.llc_file, 'r') as nc:
        Nx_llc = nc.dimensions['lon'].size
        Ny_llc = nc.dimensions['lat'].size

    train_inds, test_inds = pick_layout(args.layout, Ny_llc, Nx_llc)

    valid_inds = (0, 512, Nx - 768, Nx) if (Ny >= 512 and Nx >= 768) else (0, Ny, 0, Nx)
    args.valid_inds = valid_inds
    
    # Batch sizes depend on model complexity
    if args.model == 'samudra0' or args.nbase == 32:
        batch_sizes = {'train': 32, 'test': 100, 'valid': 25}
    else:
        batch_sizes = {'train': 64, 'test': 200, 'valid': 50}
    
    # Load datasets
    varlist = ['loggrad_T', 'U', 'V']
    train_data, test_data = load_datasets(
        args.llc_file, varlist, train_inds, test_inds,
        step0=args.step0, nframes=args.nframes
    )
    # num_workers=0 on macOS (netCDF handles don't survive process spawn)
    nw = 0 if sys.platform == 'darwin' else 5
    train_loader, test_loader = create_dataloaders(train_data, test_data,
                                                   batch_sizes=batch_sizes,
                                                   num_workers=nw)
    
    # Initialize model
    sample_batch = next(iter(test_loader))
    sample_x, sample_y = sample_batch[0], sample_batch[1]
    n_input, n_output = sample_x.shape[1], sample_y.shape[1]
    
    model = initialize_model(
        n_input, n_output,
        model_name=args.model,
        nbase=args.nbase,
        kernel_size=args.kernel_size,
        device=device
    )
    
    # Load pretrained weights. Explicit --init_from wins; otherwise fall back to
    # the c_spec>0 auto-discovery convention (stage 1 -> stage 2 chain).
    model_str = get_model_string(args.model, args.nbase, args.kernel_size, args.use_grad_loss)
    if args.init_from:
        if not os.path.exists(args.init_from):
            raise SystemExit(f'--init_from {args.init_from!r} does not exist')
        print(f'Loading initial weights from {args.init_from}')
        model = load_model(model, args.init_from, device)
    elif args.c_spec > 0:
        stage0_file = f'{model_str}_{args.step0}_{args.nframes}_0.0cs.pth'
        if os.path.exists(stage0_file):
            print(f'Loading stage-1 checkpoint {stage0_file}')
            model = load_model(model, stage0_file, device)
        else:
            print(f'Warning: stage-1 checkpoint {stage0_file} not found, starting from scratch')
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5
    )
    
    # Training criterion
    criterion = nn.L1Loss()
    
    # Train
    best_model, r2_history = train_model(
        model, train_loader, test_loader,
        optimizer, criterion, args, device
    )
    
    # Save final results
    np.save(f'r2_{model_str}_ver_{args.c_spec}cs.npy', r2_history)
    save_model(best_model, f'{model_str}_{args.step0}_{args.nframes}_{args.c_spec}cs.pth')
    
    # Final satellite inference
    out_val, grad_val, sst_val = run_satellite_inference(
        best_model, args.goes_file, args.valid_inds,
        args.pm, args.pn
    )
    
    output_file = f'preds_{model_str}_{args.step0}_{args.nframes}_{args.c_spec}cs_{os.path.splitext(os.path.basename(args.goes_file))[0]}.nc'
    write_satellite_netcdf(output_file, out_val, grad_val, sst_val, args.valid_inds, args.goes_file)
    
    print('Training complete!')


if __name__ == '__main__':
    main()
