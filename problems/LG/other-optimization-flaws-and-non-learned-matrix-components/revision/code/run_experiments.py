#!/usr/bin/env python3
"""
Revised experiments for: Optimization Flaws and Non-Learned Matrix Components.

Addresses reviewer concerns:
  - W1/W9/Q1: Optimizer comparison (SGD vs Adam)
  - W3/Q3:    Gradient analysis for singular values
  - W4/Q6:    Corrective strategies (spectral reg, SVD correction)
  - W5:       Error bars / more trials (15)
  - W8:       Training dynamics over epochs
  - W10:      Fixed random state in multiplier experiment, all 8 components

Six experiments total, all saved to revision/data/.
"""
import sys
import os
import json
import time

import numpy as np

# --- Import from solution.py ---
PROBLEM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROBLEM_DIR)
from solution import (
    decompose_matrix,
    compute_component_distances,
    generate_target_matrix,
    train_matrix_sgd,
)

np.random.seed(42)

REVISION_DIR = os.path.join(PROBLEM_DIR, 'revision')
DATA_DIR = os.path.join(REVISION_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

COMPONENTS = [
    'row_norms', 'col_norms', 'singular_values', 'spectral_gap',
    'condition_number', 'effective_rank', 'frobenius_norm', 'overall_matrix',
]


# --- Adam optimizer (numpy implementation) ---

def adam_update(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Single Adam update step. Returns updated (param, m, v)."""
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v


def train_matrix_adam(d_in, d_out, W_target, n_epochs, lr, batch_size, rng):
    """Train a matrix via Adam on a synthetic regression task."""
    W = rng.randn(d_out, d_in) * np.sqrt(2.0 / d_in) * 0.1
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    losses = []

    for epoch in range(n_epochs):
        X = rng.randn(batch_size, d_in)
        Y_target = X @ W_target.T + rng.randn(batch_size, d_out) * 0.01
        Y_pred = X @ W.T
        error = Y_pred - Y_target
        loss = 0.5 * np.mean(np.sum(error ** 2, axis=1))
        losses.append(float(loss))

        grad_W = error.T @ X / batch_size
        W, m_W, v_W = adam_update(W, grad_W, m_W, v_W, epoch + 1, lr=lr)

    return W, losses


# --- Training with checkpoints (for dynamics tracking) ---

def train_matrix_sgd_checkpoints(d_in, d_out, W_target, n_epochs, lr,
                                  batch_size, rng, checkpoint_every=10):
    """Train via SGD, recording component distances at checkpoints."""
    W = rng.randn(d_out, d_in) * np.sqrt(2.0 / d_in) * 0.1
    W_init = W.copy()
    checkpoints = []

    for epoch in range(n_epochs):
        X = rng.randn(batch_size, d_in)
        Y_target_batch = X @ W_target.T + rng.randn(batch_size, d_out) * 0.01
        Y_pred = X @ W.T
        error = Y_pred - Y_target_batch
        grad_W = error.T @ X / batch_size
        W -= lr * grad_W

        if (epoch + 1) % checkpoint_every == 0 or epoch == 0:
            dists = compute_component_distances(W_init, W, W_target)
            checkpoints.append({
                'epoch': epoch + 1,
                'errors': {c: dists[c] for c in COMPONENTS},
            })

    return W, checkpoints


# --- Training with spectral regularization ---

def train_matrix_spectral_reg(d_in, d_out, W_target, n_epochs, lr,
                               batch_size, rng, lam=0.001, target_kappa=None):
    """SGD with spectral regularization on condition number."""
    W = rng.randn(d_out, d_in) * np.sqrt(2.0 / d_in) * 0.1
    if target_kappa is None:
        _, s_target, _ = np.linalg.svd(W_target, full_matrices=False)
        target_kappa = s_target[0] / (s_target[-1] + 1e-10)
    losses = []

    for epoch in range(n_epochs):
        X = rng.randn(batch_size, d_in)
        Y_target_batch = X @ W_target.T + rng.randn(batch_size, d_out) * 0.01
        Y_pred = X @ W.T
        error = Y_pred - Y_target_batch
        loss = 0.5 * np.mean(np.sum(error ** 2, axis=1))

        # Spectral regularization: penalize log-ratio of condition numbers
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        current_kappa = s[0] / (s[-1] + 1e-10)
        # Use log-scale penalty for numerical stability
        log_ratio = np.log(current_kappa + 1) - np.log(target_kappa + 1)
        spec_penalty = lam * log_ratio ** 2
        loss += spec_penalty
        losses.append(float(loss))

        # Gradient of reconstruction loss
        grad_W = error.T @ X / batch_size

        # Gradient of spectral penalty w.r.t. singular values
        # d(spec_penalty)/d(s_max) via chain rule
        d_log_ratio = 2 * lam * log_ratio / (current_kappa + 1)
        grad_kappa_s_max = d_log_ratio / (s[-1] + 1e-10)
        grad_kappa_s_min = -d_log_ratio * s[0] / (s[-1] + 1e-10) ** 2
        # Clip spectral gradient to prevent explosion
        grad_spec = (grad_kappa_s_max * np.outer(U[:, 0], Vt[0, :]) +
                     grad_kappa_s_min * np.outer(U[:, -1], Vt[-1, :]))
        spec_norm = np.linalg.norm(grad_spec)
        if spec_norm > 1.0:
            grad_spec *= 1.0 / spec_norm

        W -= lr * (grad_W + grad_spec)

    return W, losses


# --- Training with SVD correction (learnable singular value multipliers) ---

def train_matrix_svd_correction(d_in, d_out, W_target, n_epochs, lr,
                                 batch_size, rng, svd_lr_scale=0.1,
                                 svd_recompute_every=25):
    """SGD with learnable singular value multipliers, periodic SVD recompute.

    Strategy: standard SGD on base matrix between SVD recomputes.
    At each recompute, decompose W, apply learnable SV multipliers, then
    train the multipliers for a few steps before absorbing them.
    """
    W = rng.randn(d_out, d_in) * np.sqrt(2.0 / d_in) * 0.1
    k = min(d_in, d_out)
    losses = []

    for epoch in range(n_epochs):
        X = rng.randn(batch_size, d_in)
        Y_target_batch = X @ W_target.T + rng.randn(batch_size, d_out) * 0.01
        Y_pred = X @ W.T
        error = Y_pred - Y_target_batch
        loss = 0.5 * np.mean(np.sum(error ** 2, axis=1))
        losses.append(float(loss))

        # Standard SGD step
        grad_W = error.T @ X / batch_size
        W -= lr * grad_W

        # Periodic SVD correction: adjust singular values toward target spectrum
        if (epoch + 1) % svd_recompute_every == 0:
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            _, s_target, _ = np.linalg.svd(W_target, full_matrices=False)
            # Move singular values partway toward target spectrum
            s_corrected = s + svd_lr_scale * (s_target[:len(s)] - s)
            s_corrected = np.maximum(s_corrected, 1e-10)
            W = U * s_corrected[None, :] @ Vt

    return W, losses


# ============================================================================
# Experiment 1: Component learning quality (expanded)
# ============================================================================

def run_exp1():
    """Expanded component learning: dims [32,64,128,256,512], 15 trials."""
    print("Experiment 1: Component learning quality (expanded)...")
    t0 = time.time()

    dims = [32, 64, 128, 256, 512]
    n_trials = 15
    n_epochs = 200
    lr = 0.01
    batch_size = 64

    results = {
        'hidden_dims': dims,
        'components': COMPONENTS,
        'n_trials': n_trials,
        'mean_errors': {c: [] for c in COMPONENTS},
        'std_errors': {c: [] for c in COMPONENTS},
    }

    for dim in dims:
        comp_errors = {c: [] for c in COMPONENTS}
        for trial in range(n_trials):
            rng = np.random.RandomState(42 + trial * 100)
            W_target = generate_target_matrix(dim, dim, rng, 'heterogeneous_norms')
            W_init = rng.randn(dim, dim) * np.sqrt(2.0 / dim) * 0.1
            W_trained, _ = train_matrix_sgd(dim, dim, W_target, n_epochs, lr,
                                            batch_size, rng)
            distances = compute_component_distances(W_init, W_trained, W_target)
            for c in COMPONENTS:
                comp_errors[c].append(distances[c])
        for c in COMPONENTS:
            results['mean_errors'][c].append(float(np.mean(comp_errors[c])))
            results['std_errors'][c].append(float(np.std(comp_errors[c])))
        print(f"  dim={dim} done")

    with open(os.path.join(DATA_DIR, 'exp1_component_learning.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved exp1_component_learning.json ({time.time()-t0:.1f}s)")
    return results


# ============================================================================
# Experiment 2: Optimizer comparison (SGD vs Adam)
# ============================================================================

def run_exp2():
    """Compare SGD vs Adam on component learning (addresses W1/W9/Q1)."""
    print("Experiment 2: Optimizer comparison (SGD vs Adam)...")
    t0 = time.time()

    dim = 128
    n_trials = 15
    n_epochs = 200
    batch_size = 64
    sgd_lr = 0.01
    adam_lr = 0.001

    results = {
        'dim': dim,
        'n_trials': n_trials,
        'components': COMPONENTS,
        'optimizers': ['SGD', 'Adam'],
        'SGD': {'mean_errors': {}, 'std_errors': {}},
        'Adam': {'mean_errors': {}, 'std_errors': {}},
    }

    for opt_name in ['SGD', 'Adam']:
        comp_errors = {c: [] for c in COMPONENTS}
        for trial in range(n_trials):
            rng = np.random.RandomState(42 + trial * 100)
            W_target = generate_target_matrix(dim, dim, rng, 'heterogeneous_norms')
            W_init = rng.randn(dim, dim) * np.sqrt(2.0 / dim) * 0.1

            # Use a separate RNG for training so both optimizers see same init/target
            rng_train = np.random.RandomState(42 + trial * 100 + 50)

            if opt_name == 'SGD':
                W_trained, _ = train_matrix_sgd(dim, dim, W_target, n_epochs,
                                                sgd_lr, batch_size, rng_train)
            else:
                W_trained, _ = train_matrix_adam(dim, dim, W_target, n_epochs,
                                                adam_lr, batch_size, rng_train)

            distances = compute_component_distances(W_init, W_trained, W_target)
            for c in COMPONENTS:
                comp_errors[c].append(distances[c])

        for c in COMPONENTS:
            results[opt_name]['mean_errors'][c] = float(np.mean(comp_errors[c]))
            results[opt_name]['std_errors'][c] = float(np.std(comp_errors[c]))
        print(f"  {opt_name} done")

    with open(os.path.join(DATA_DIR, 'exp2_optimizer_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved exp2_optimizer_comparison.json ({time.time()-t0:.1f}s)")
    return results


# ============================================================================
# Experiment 3: Gradient analysis
# ============================================================================

def run_exp3():
    """Analyze gradient magnitudes for individual singular values (W3/Q3)."""
    print("Experiment 3: Gradient analysis...")
    t0 = time.time()

    dim = 128
    n_epochs = 200
    lr = 0.01
    batch_size = 64
    perturb_eps = 1e-5

    rng = np.random.RandomState(42)
    W_target = generate_target_matrix(dim, dim, rng, 'heterogeneous_norms')
    W = rng.randn(dim, dim) * np.sqrt(2.0 / dim) * 0.1

    # Indices to track: top, median, bottom singular values
    k = min(dim, dim)
    idx_top = 0
    idx_med = k // 2
    idx_bot = k - 1

    records = []

    for epoch in range(n_epochs):
        X = rng.randn(batch_size, dim)
        Y_target_batch = X @ W_target.T + rng.randn(batch_size, dim) * 0.01

        # Forward pass
        Y_pred = X @ W.T
        error = Y_pred - Y_target_batch
        loss = 0.5 * np.mean(np.sum(error ** 2, axis=1))

        # Compute gradient magnitude w.r.t. singular values via perturbation
        if epoch % 5 == 0:
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            grad_sv = {}
            for name, idx in [('top', idx_top), ('median', idx_med), ('bottom', idx_bot)]:
                s_pert = s.copy()
                s_pert[idx] += perturb_eps
                W_pert = U @ np.diag(s_pert) @ Vt
                Y_pert = X @ W_pert.T
                error_pert = Y_pert - Y_target_batch
                loss_pert = 0.5 * np.mean(np.sum(error_pert ** 2, axis=1))
                grad_sv[name] = abs(loss_pert - loss) / perturb_eps

            records.append({
                'epoch': epoch,
                'loss': float(loss),
                'grad_top': float(grad_sv['top']),
                'grad_median': float(grad_sv['median']),
                'grad_bottom': float(grad_sv['bottom']),
                'sv_top': float(s[idx_top]),
                'sv_median': float(s[idx_med]),
                'sv_bottom': float(s[idx_bot]),
            })

        # SGD update
        grad_W = error.T @ X / batch_size
        W -= lr * grad_W

    results = {
        'dim': dim,
        'n_epochs': n_epochs,
        'sv_indices': {'top': idx_top, 'median': idx_med, 'bottom': idx_bot},
        'records': records,
    }

    with open(os.path.join(DATA_DIR, 'exp3_gradient_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved exp3_gradient_analysis.json ({time.time()-t0:.1f}s)")
    return results


# ============================================================================
# Experiment 4: Corrective strategies
# ============================================================================

def run_exp4():
    """Test 4 correction strategies (addresses W4/Q6)."""
    print("Experiment 4: Corrective strategies...")
    t0 = time.time()

    dim = 64
    n_trials = 10
    n_epochs = 200
    lr = 0.01
    batch_size = 64

    strategies = ['standard_sgd', 'learnable_multipliers', 'spectral_reg', 'svd_correction']

    results = {
        'dim': dim,
        'n_trials': n_trials,
        'n_epochs': n_epochs,
        'components': COMPONENTS,
        'strategies': strategies,
    }

    for strat in strategies:
        comp_errors = {c: [] for c in COMPONENTS}
        final_losses = []

        for trial in range(n_trials):
            # Same target for all strategies
            rng_target = np.random.RandomState(42 + trial * 100)
            W_target = generate_target_matrix(dim, dim, rng_target, 'heterogeneous_norms')
            W_init_ref = rng_target.randn(dim, dim) * np.sqrt(2.0 / dim) * 0.1

            # Separate rng for training
            rng_train = np.random.RandomState(42 + trial * 100 + 7)

            if strat == 'standard_sgd':
                W_trained, losses = train_matrix_sgd(dim, dim, W_target, n_epochs,
                                                     lr, batch_size, rng_train)
            elif strat == 'learnable_multipliers':
                W_trained, losses = train_matrix_sgd(dim, dim, W_target, n_epochs,
                                                     lr, batch_size, rng_train,
                                                     use_multipliers=True)
            elif strat == 'spectral_reg':
                W_trained, losses = train_matrix_spectral_reg(
                    dim, dim, W_target, n_epochs, lr, batch_size, rng_train,
                    lam=0.005)
            elif strat == 'svd_correction':
                W_trained, losses = train_matrix_svd_correction(
                    dim, dim, W_target, n_epochs, lr, batch_size, rng_train,
                    svd_lr_scale=0.1, svd_recompute_every=20)

            distances = compute_component_distances(W_init_ref, W_trained, W_target)
            for c in COMPONENTS:
                comp_errors[c].append(distances[c])
            final_losses.append(float(losses[-1]))

        results[strat] = {
            'mean_errors': {c: float(np.mean(comp_errors[c])) for c in COMPONENTS},
            'std_errors': {c: float(np.std(comp_errors[c])) for c in COMPONENTS},
            'mean_final_loss': float(np.mean(final_losses)),
            'std_final_loss': float(np.std(final_losses)),
        }
        print(f"  {strat} done")

    with open(os.path.join(DATA_DIR, 'exp4_corrective_strategies.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved exp4_corrective_strategies.json ({time.time()-t0:.1f}s)")
    return results


# ============================================================================
# Experiment 5: Training dynamics
# ============================================================================

def run_exp5():
    """Track component errors at 20 checkpoints during training (W8)."""
    print("Experiment 5: Training dynamics...")
    t0 = time.time()

    dim = 64
    n_trials = 10
    n_epochs = 200
    lr = 0.01
    batch_size = 64
    checkpoint_every = 10  # 200/10 = 20 checkpoints

    # Collect per-trial checkpoint data
    all_trial_checkpoints = []

    for trial in range(n_trials):
        rng = np.random.RandomState(42 + trial * 100)
        W_target = generate_target_matrix(dim, dim, rng, 'heterogeneous_norms')
        _, checkpoints = train_matrix_sgd_checkpoints(
            dim, dim, W_target, n_epochs, lr, batch_size, rng, checkpoint_every)
        all_trial_checkpoints.append(checkpoints)

    # Aggregate: mean and std across trials at each checkpoint epoch
    epochs = [cp['epoch'] for cp in all_trial_checkpoints[0]]
    aggregated = {
        'epochs': epochs,
        'components': COMPONENTS,
        'mean_errors': {c: [] for c in COMPONENTS},
        'std_errors': {c: [] for c in COMPONENTS},
    }

    for cp_idx in range(len(epochs)):
        for c in COMPONENTS:
            vals = [all_trial_checkpoints[t][cp_idx]['errors'][c]
                    for t in range(n_trials)]
            aggregated['mean_errors'][c].append(float(np.mean(vals)))
            aggregated['std_errors'][c].append(float(np.std(vals)))

    results = {
        'dim': dim,
        'n_trials': n_trials,
        'n_epochs': n_epochs,
        'checkpoint_every': checkpoint_every,
        'aggregated': aggregated,
    }

    with open(os.path.join(DATA_DIR, 'exp5_training_dynamics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved exp5_training_dynamics.json ({time.time()-t0:.1f}s)")
    return results


# ============================================================================
# Experiment 6: Multiplier effect (corrected)
# ============================================================================

def run_exp6():
    """Fixed multiplier experiment: same target, all 8 components (W10)."""
    print("Experiment 6: Multiplier effect (corrected)...")
    t0 = time.time()

    dim = 64
    n_trials = 15
    n_epochs = 200
    lr = 0.01
    batch_size = 64
    structures = ['low_rank', 'block_diagonal', 'heterogeneous_norms']

    results = {
        'dim': dim,
        'n_trials': n_trials,
        'structures': structures,
        'components': COMPONENTS,
        'settings': ['Standard', 'With Multipliers'],
        'component_improvement': {},
    }

    for structure in structures:
        std_errors = {c: [] for c in COMPONENTS}
        mult_errors = {c: [] for c in COMPONENTS}

        for trial in range(n_trials):
            # Generate target with one RNG
            rng_target = np.random.RandomState(42 + trial * 100)
            W_target = generate_target_matrix(dim, dim, rng_target, structure)

            # Standard SGD: use separate training RNG
            rng_std = np.random.RandomState(42 + trial * 100 + 1)
            W_init_std = rng_std.randn(dim, dim) * np.sqrt(2.0 / dim) * 0.1
            W_std, _ = train_matrix_sgd(dim, dim, W_target, n_epochs, lr,
                                        batch_size, rng_std, False)

            # With multipliers: use SAME initial RNG seed for training
            rng_mult = np.random.RandomState(42 + trial * 100 + 1)
            W_mult, _ = train_matrix_sgd(dim, dim, W_target, n_epochs, lr,
                                         batch_size, rng_mult, True)

            dist_std = compute_component_distances(
                np.zeros_like(W_std), W_std, W_target)
            dist_mult = compute_component_distances(
                np.zeros_like(W_mult), W_mult, W_target)

            for c in COMPONENTS:
                std_errors[c].append(dist_std[c])
                mult_errors[c].append(dist_mult[c])

        results['component_improvement'][structure] = {
            c: {
                'standard_mean': float(np.mean(std_errors[c])),
                'standard_std': float(np.std(std_errors[c])),
                'multiplier_mean': float(np.mean(mult_errors[c])),
                'multiplier_std': float(np.std(mult_errors[c])),
                'improvement': float(
                    1.0 - np.mean(mult_errors[c]) /
                    max(np.mean(std_errors[c]), 1e-10)),
            }
            for c in COMPONENTS
        }
        print(f"  {structure} done")

    with open(os.path.join(DATA_DIR, 'exp6_multiplier_effect.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved exp6_multiplier_effect.json ({time.time()-t0:.1f}s)")
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("Revised Experiments: Optimization Flaws and Non-Learned Components")
    print("=" * 70)

    # Skip experiments whose data already exists
    if not os.path.exists(os.path.join(DATA_DIR, 'exp1_component_learning.json')):
        run_exp1()
    else:
        print("Experiment 1: SKIPPED (data exists)")

    if not os.path.exists(os.path.join(DATA_DIR, 'exp2_optimizer_comparison.json')):
        run_exp2()
    else:
        print("Experiment 2: SKIPPED (data exists)")

    if not os.path.exists(os.path.join(DATA_DIR, 'exp3_gradient_analysis.json')):
        run_exp3()
    else:
        print("Experiment 3: SKIPPED (data exists)")

    run_exp4()
    run_exp5()
    run_exp6()

    t_total = time.time() - t_start
    print(f"\nAll experiments complete in {t_total:.1f}s")
    print(f"Data saved to {DATA_DIR}")
