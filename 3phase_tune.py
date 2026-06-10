import os
import sys
import warnings
import optuna
import argparse

# Platform detection and configuration
import jax
jax.config.update('jax_platform_name', None)  # Auto-detect platform

# GPU-specific configurations (only applied if GPU is available)
if jax.default_backend() == 'gpu':
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
else:
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=true'

import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

import time
import jax.numpy as jnp
import jax.random as random
from pathlib import Path
from model import NGCTransformer
from data_preprocess.data_loader import DataLoader
from config import Config as base_config
from ngclearn.utils.metric_utils import measure_CatNLL
import gc
import numpy as np

# Fixed scaling factors (not tuned)
ALPHA_CE = 1.0  # Weight for CE (output layer loss)
BETA_EFE = 0.01  # Weight for EFE (hidden layers loss)

EFE_STABILITY_THRESHOLD = 2e2
COMBINED_LOSS_THRESHOLD = 1e1

def get_default_config():
    """Get default parameters from base config"""
    cfg = type('Config', (), {})()
    for key, value in base_config.__dict__.items():
        if not key.startswith('_'):
            setattr(cfg, key, value)
    return cfg

def define_search_space_phase0(trial):
    """PHASE 0: Warm-up - Tune architecture and basic parameters"""
    n_heads = trial.suggest_int("n_heads", 2, 8)
    embed_mult = trial.suggest_int("embed_mult", 16, 64, step=4)
    n_embed = n_heads * embed_mult
    trial.suggest_int("n_embed", n_embed, n_embed)
    return {
        "n_layers": trial.suggest_int("n_layers", 2, 8),
        "n_heads": n_heads,
        "n_embed": n_embed,
        "batch_size": trial.suggest_int("batch_size", 8, 64),
        "seq_len": trial.suggest_int("seq_len", 16, 256),
        "embed_mult": embed_mult,
        "pos_learnable": trial.suggest_categorical("pos_learnable", [True, False]),
        "n_iter": trial.suggest_int("n_iter", 10, 30),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.0),
        "optim_type": trial.suggest_categorical("optim_type", ["adam", "sgd"]),
        "eta": base_config.eta,
        "eta_o": base_config.eta_o,
        "tau_m": base_config.tau_m,
        "tau_o": base_config.tau_o,
        "act_fx": base_config.act_fx,
        "act_fx_o": base_config.act_fx_o,
        "wub": base_config.wub,
        "wlb": base_config.wlb,
        "wu": base_config.wu,
        "wl": base_config.wl,
    }

def define_search_space_phase1(trial, best_params_phase0, default_cfg):
    """PHASE 1: Tune output layer parameters for CE optimization only"""
    params = {**best_params_phase0}
    # Use new parameter names to avoid conflicts with Phase 0
    params.update({
        "eta_o_tuned": trial.suggest_float("eta_o_tuned", 1e-6, 0.01, log=True),
        "tau_o_tuned": trial.suggest_float("tau_o_tuned", 0.5, 10, log=True),
        "act_fx_o_tuned": trial.suggest_categorical("act_fx_o_tuned", ["relu", "gelu", "identity"]),
    })
    return params

def define_search_space_phase2(trial, best_params_phase0, best_params_phase1, default_cfg):
    """PHASE 2: Tune hidden layer parameters for EFE optimization only"""
    params = {**best_params_phase0, **best_params_phase1}
    # Use new parameter names to avoid conflicts with Phase 0 and Phase 1
    params.update({
        "eta_tuned": trial.suggest_float("eta_tuned", 1e-6, 0.01, log=True),
        "tau_m_tuned": trial.suggest_float("tau_m_tuned", 0.5, 10, log=True),
        "act_fx_tuned": trial.suggest_categorical("act_fx_tuned", ["relu", "gelu", "identity"]),
    })
    return params

def fill_missing_params(params, default_cfg):
    """Fill None values with defaults from config"""
    for key, value in default_cfg.__dict__.items():
        if key not in params or params[key] is None:
            params[key] = value
    if 'n_heads' in params and 'embed_mult' in params:
        expected_n_embed = params['n_heads'] * params['embed_mult']
        params['n_embed'] = expected_n_embed
    return params

def create_model_with_params(trial_number, params, cfg):
    """Create model with given parameters"""
    data_loader = DataLoader(seq_len=cfg.seq_len, batch_size=cfg.batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()
    dkey = random.PRNGKey(trial_number * 1000 + 42)

    # Use tuned values if available, otherwise use defaults
    eta_o = params.get('eta_o_tuned', params.get('eta_o', base_config.eta_o))
    tau_o = params.get('tau_o_tuned', params.get('tau_o', base_config.tau_o))
    act_fx_o = params.get('act_fx_o_tuned', params.get('act_fx_o', base_config.act_fx_o))
    eta = params.get('eta_tuned', params.get('eta', base_config.eta))
    tau_m = params.get('tau_m_tuned', params.get('tau_m', base_config.tau_m))
    act_fx = params.get('act_fx_tuned', params.get('act_fx', base_config.act_fx))

    model_args = {
        "dkey": dkey,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "n_embed": cfg.n_embed,
        "vocab_size": cfg.vocab_size,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "T": cfg.n_iter,
        "dt": 1.0,
        "tau_m": tau_m,
        "tau_o": tau_o,
        "act_fx": act_fx,
        "act_fx_o": act_fx_o,
        "eta": eta,
        "dropout_rate": cfg.dropout_rate,
        "exp_dir": None,
        "loadDir": None,
        "pos_learnable": cfg.pos_learnable,
        "optim_type": cfg.optim_type,
        "wub": cfg.wub,
        "wlb": cfg.wlb,
        "eta_o": eta_o,
        "wu": cfg.wu,
        "wl": cfg.wl,
        "model_name": f"trial_{trial_number}"
    }

    model = NGCTransformer(**model_args)
    return model, train_loader, valid_loader

def train_one_epoch(model, train_loader, cfg, objective="combined"):
    """
    Train for one FULL epoch and return metrics based on objective
    """
    total_ce = 0.0
    total_efe = 0.0
    batches_processed = 0
    
    # Use fixed scaling factors
    alpha_ce = ALPHA_CE
    beta_efe = BETA_EFE
    
    # Normalize EFE by number of hidden layers
    n_hidden_layers = max(cfg.n_layers, 1)
    efe_normalization = n_hidden_layers

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0][1]
        targets = batch[1][1]
        targets_flat = jax.nn.one_hot(targets.flatten(), cfg.vocab_size)

        try:
            _, y_mu, EFE = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            y_pred = y_mu.reshape(-1, cfg.vocab_size)
            batch_ce = measure_CatNLL(y_pred, targets_flat).mean()
            
            efe_val = abs(float(EFE)) if EFE is not None else 0.0
            efe_per_layer = efe_val / efe_normalization

            if jnp.isnan(batch_ce) or jnp.isinf(batch_ce):
                return 1e6, 1e6, 1e6

            total_ce += float(batch_ce)
            total_efe += efe_per_layer
            batches_processed += 1

            # Print every 50 batches to reduce output
            if batch_idx % 50 == 0:
                if objective == "combined":
                    combined = alpha_ce * batch_ce + beta_efe * efe_per_layer
                    print(f"  Batch {batch_idx:4d}: Combined={combined:.4f} | CE={batch_ce:.4f} | EFE_avg={efe_per_layer:.4f}")
                elif objective == "ce":
                    print(f"  Batch {batch_idx:4d}: CE={batch_ce:.4f} | EFE_avg={efe_per_layer:.4f}")
                elif objective == "efe":
                    print(f"  Batch {batch_idx:4d}: EFE_avg={efe_per_layer:.4f} | CE={batch_ce:.4f}")

        except Exception as e:
            print(f"  Error in batch {batch_idx}: {str(e)[:100]}")
            return 1e6, 1e6, 1e6

    if batches_processed == 0:
        return 1e6, 1e6, 1e6
        
    avg_ce = total_ce / batches_processed
    avg_efe_per_layer = total_efe / batches_processed
    
    if objective == "combined":
        main_loss = alpha_ce * avg_ce + beta_efe * avg_efe_per_layer
    elif objective == "ce":
        main_loss = avg_ce
    elif objective == "efe":
        main_loss = avg_efe_per_layer
    else:
        main_loss = alpha_ce * avg_ce + beta_efe * avg_efe_per_layer
    
    return main_loss, avg_ce, avg_efe_per_layer

def print_params_horizontal(params, phase_name, trial_number):
    """Print hyperparameters in a horizontal, compact format"""
    key_params = ['n_layers', 'n_heads', 'n_embed', 'batch_size', 'seq_len', 
                  'n_iter', 'optim_type', 'pos_learnable']
    
    param_strs = []
    for k in key_params:
        if k in params:
            if isinstance(params[k], float):
                param_strs.append(f"{k}={params[k]:.4f}")
            else:
                param_strs.append(f"{k}={params[k]}")
    
    print(f"[{phase_name} Trial {trial_number}] {' | '.join(param_strs)}")

# PHASE 0: Optimize combined loss
def run_phase0_trial(trial):
    """PHASE 0: Warm-up - Optimize combined loss (CE + α·EFE)"""
    try:
        default_cfg = get_default_config()
        params = define_search_space_phase0(trial)
        params = fill_missing_params(params, default_cfg)
        
        print_params_horizontal(params, "Phase0", trial.number)

        cfg = type('Config', (), {})()
        for key, value in default_cfg.__dict__.items():
            if not key.startswith('_'):
                setattr(cfg, key, value)
        for key, value in params.items():
            setattr(cfg, key, value)
        cfg.vocab_size = base_config.vocab_size

        model, train_loader, valid_loader = create_model_with_params(trial.number, params, cfg)
        
        start_time = time.time()
        print("  Training on full dataset:")
        combined_loss, avg_ce, avg_efe = train_one_epoch(model, train_loader, cfg, objective="combined")

        if combined_loss > COMBINED_LOSS_THRESHOLD or np.isnan(combined_loss):
            raise optuna.TrialPruned()

        total_time = time.time() - start_time

        trial.set_user_attr("phase0_ce", float(avg_ce))
        trial.set_user_attr("phase0_efe", float(avg_efe))
        trial.set_user_attr("phase0_time", total_time)

        print(f"  ✓ Complete: Combined={combined_loss:.4f} | CE={avg_ce:.4f} | EFE_avg={avg_efe:.4f} | Time={total_time:.1f}s\n")
        return float(combined_loss)

    except Exception as e:
        print(f"  ✗ Trial failed: {str(e)[:100]}")
        raise optuna.TrialPruned()

    finally:
        gc.collect()
        jax.clear_caches()

# PHASE 1: Optimize CE only
def run_phase1_trial(trial, best_params_phase0):
    """PHASE 1: Output layer tuning - Optimize CE only"""
    try:
        default_cfg = get_default_config()
        params = define_search_space_phase1(trial, best_params_phase0, default_cfg)
        params = fill_missing_params(params, default_cfg)
        
        print(f"[Phase1 Trial {trial.number}] eta_o={params.get('eta_o_tuned', 'N/A'):.6f} | tau_o={params.get('tau_o_tuned', 'N/A'):.4f} | act_fx_o={params.get('act_fx_o_tuned', 'N/A')}")

        cfg = type('Config', (), {})()
        for key, value in default_cfg.__dict__.items():
            if not key.startswith('_'):
                setattr(cfg, key, value)
        for key, value in params.items():
            setattr(cfg, key, value)
        cfg.vocab_size = base_config.vocab_size

        model, train_loader, valid_loader = create_model_with_params(trial.number, params, cfg)
        
        start_time = time.time()
        print("  Training on full dataset:")
        ce_loss, avg_ce, avg_efe = train_one_epoch(model, train_loader, cfg, objective="ce")

        if ce_loss > COMBINED_LOSS_THRESHOLD or np.isnan(ce_loss):
            raise optuna.TrialPruned()

        total_time = time.time() - start_time

        trial.set_user_attr("phase1_ce", float(avg_ce))
        trial.set_user_attr("phase1_efe", float(avg_efe))
        trial.set_user_attr("phase1_time", total_time)

        print(f"  ✓ Complete: CE={ce_loss:.4f} | EFE_avg={avg_efe:.4f} | Time={total_time:.1f}s\n")
        return float(ce_loss)

    except Exception as e:
        print(f"  ✗ Trial failed: {str(e)[:100]}")
        raise optuna.TrialPruned()

    finally:
        gc.collect()
        jax.clear_caches()

# PHASE 2: Optimize EFE only
def run_phase2_trial(trial, best_params_phase0, best_params_phase1):
    """PHASE 2: Hidden layer tuning - Optimize EFE only"""
    try:
        default_cfg = get_default_config()
        params = define_search_space_phase2(trial, best_params_phase0, best_params_phase1, default_cfg)
        params = fill_missing_params(params, default_cfg)
        
        print(f"[Phase2 Trial {trial.number}] eta={params.get('eta_tuned', 'N/A'):.6f} | tau_m={params.get('tau_m_tuned', 'N/A'):.4f} | act_fx={params.get('act_fx_tuned', 'N/A')}")

        cfg = type('Config', (), {})()
        for key, value in default_cfg.__dict__.items():
            if not key.startswith('_'):
                setattr(cfg, key, value)
        for key, value in params.items():
            setattr(cfg, key, value)
        cfg.vocab_size = base_config.vocab_size

        model, train_loader, valid_loader = create_model_with_params(trial.number, params, cfg)
        
        start_time = time.time()
        print("  Training on full dataset:")
        efe_loss, avg_ce, avg_efe = train_one_epoch(model, train_loader, cfg, objective="efe")

        if efe_loss > EFE_STABILITY_THRESHOLD or np.isnan(efe_loss):
            raise optuna.TrialPruned()

        total_time = time.time() - start_time

        trial.set_user_attr("phase2_ce", float(avg_ce))
        trial.set_user_attr("phase2_efe", float(avg_efe))
        trial.set_user_attr("phase2_time", total_time)

        print(f"  ✓ Complete: EFE_avg={efe_loss:.4f} | CE={avg_ce:.4f} | Time={total_time:.1f}s\n")
        return float(efe_loss)

    except Exception as e:
        print(f"  ✗ Trial failed: {str(e)[:100]}")
        raise optuna.TrialPruned()

    finally:
        gc.collect()
        jax.clear_caches()

def three_phase_tuning(study_suffix=""):
    """Three-phase tuning with different objectives per phase"""
    Path("tuning").mkdir(exist_ok=True)
    default_cfg = get_default_config()
    
    # Add suffix to study names to avoid conflicts in parallel runs
    suffix = f"_{study_suffix}" if study_suffix else ""
    
    print("="*80)
    print("THREE-PHASE SEQUENTIAL TUNING")
    print("="*80)
    print(f"Platform: {jax.default_backend()}")
    print(f"Available devices: {jax.device_count()}")
    print(f"Study suffix: {study_suffix if study_suffix else 'none'}")
    print(f"Fixed scaling factors: α_CE = {ALPHA_CE}, β_EFE = {BETA_EFE}")
    print("Phase 0: Architecture search (Objective: Combined Loss = α·CE + β·EFE)")
    print("Phase 1: Output layer fine-tuning (Objective: CE ONLY)")
    print("Phase 2: Hidden layer fine-tuning (Objective: EFE ONLY)")
    print("="*80)

    # ========== PHASE 0: Combined Loss ==========
    print("\n" + "="*80)
    print("PHASE 0: Architecture Search (Optimizing Combined Loss)")
    print("="*80)
    
    study_phase0 = optuna.create_study(
        study_name=f"phase0_combined{suffix}",
        storage=f"sqlite:///tuning/phase0_combined{suffix}.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=15)
    )
    
    study_phase0.optimize(run_phase0_trial, n_trials=1, n_jobs=1)
    
    best_phase0_params = fill_missing_params(
        dict(study_phase0.best_trial.params), default_cfg
    )
    print(f"\n[Phase 0] Best Combined Loss: {study_phase0.best_value:.4f}")

    # ========== PHASE 1: CE Only ==========
    print("\n" + "="*80)
    print("PHASE 1: Output Layer Optimization (Optimizing CE Only)")
    print("="*80)
    
    study_phase1 = optuna.create_study(
        study_name=f"phase1_ce{suffix}",
        storage=f"sqlite:///tuning/phase1_ce{suffix}.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    
    def phase1_wrapper(trial):
        return run_phase1_trial(trial, best_phase0_params)
    
    study_phase1.optimize(phase1_wrapper, n_trials=25, n_jobs=1)
    
    # Check if we have any successful trials
    if len(study_phase1.trials) == 0 or all(t.state != optuna.trial.TrialState.COMPLETE for t in study_phase1.trials):
        print("\n⚠ WARNING: No successful trials in Phase 1. Using Phase 0 parameters for Phase 1.")
        best_phase1_params = {**best_phase0_params}
    else:
        best_phase1_params = {**best_phase0_params, **dict(study_phase1.best_trial.params)}
        print(f"\n[Phase 1] Best CE Loss: {study_phase1.best_value:.4f}")

    # ========== PHASE 2: EFE Only ==========
    print("\n" + "="*80)
    print("PHASE 2: Hidden Layer Optimization (Optimizing EFE Only)")
    print("="*80)
    
    study_phase2 = optuna.create_study(
        study_name=f"phase2_efe{suffix}",
        storage=f"sqlite:///tuning/phase2_efe{suffix}.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    
    def phase2_wrapper(trial):
        return run_phase2_trial(trial, best_phase0_params, best_phase1_params)
    
    study_phase2.optimize(phase2_wrapper, n_trials=20, n_jobs=1)
    
    # Check if we have any successful trials
    if len(study_phase2.trials) == 0 or all(t.state != optuna.trial.TrialState.COMPLETE for t in study_phase2.trials):
        print("\n⚠ WARNING: No successful trials in Phase 2. Using Phase 1 parameters for final config.")
        best_phase2_params = {**best_phase1_params}
    else:
        best_phase2_params = {**best_phase1_params, **dict(study_phase2.best_trial.params)}
        print(f"\n[Phase 2] Best EFE Loss: {study_phase2.best_value:.4f}")

    # Save results
    final_params = fill_missing_params(best_phase2_params, default_cfg)
    
    with open(f"tuning/final_hyperparameters{suffix}.txt", "w") as f:
        f.write("FINAL HYPERPARAMETERS AFTER THREE-PHASE TUNING\n")
        f.write("="*50 + "\n")
        f.write(f"Platform: {jax.default_backend()}\n")
        f.write(f"Study suffix: {study_suffix if study_suffix else 'none'}\n")
        f.write(f"Fixed scaling factors: α_CE = {ALPHA_CE}, β_EFE = {BETA_EFE}\n")
        f.write("Phase 0 objective: Combined Loss (α·CE + β·EFE)\n")
        f.write("Phase 1 objective: CE Only\n")
        f.write("Phase 2 objective: EFE Only\n")
        f.write("="*50 + "\n\n")
        for key, value in final_params.items():
            f.write(f"{key} = {value}\n")

    print("\n" + "="*80)
    print("✓ THREE-PHASE TUNING COMPLETE!")
    print("="*80)
    print(f"Final params saved to: tuning/final_hyperparameters{suffix}.txt")
    
    print("\n" + "="*80)
    print("TUNING SUMMARY")
    print("="*80)
    print(f"Phase 0 Best Combined Loss: {study_phase0.best_value:.4f}")
    if len(study_phase1.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study_phase1.trials):
        print(f"Phase 1 Best CE Loss: {study_phase1.best_value:.4f}")
    else:
        print("Phase 1: No successful trials")
    if len(study_phase2.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study_phase2.trials):
        print(f"Phase 2 Best EFE Loss: {study_phase2.best_value:.4f}")
    else:
        print("Phase 2: No successful trials")
    
    return final_params

def main():
    parser = argparse.ArgumentParser(description='Three-phase tuning for NGC Transformer')
    parser.add_argument('--study-suffix', type=str, default='', 
                       help='Suffix for study names to avoid conflicts in parallel runs')
    args = parser.parse_args()
    
    print("="*80)
    print("NGC TRANSFORMER THREE-PHASE TUNING")
    print("="*80)
    
    # Verify GPU availability
    print(f"Available devices: {jax.device_count()}")
    print(f"Platform: {jax.default_backend()}")
    
    results = three_phase_tuning(study_suffix=args.study_suffix)
    
    if results:
        print("\nFINAL CONFIGURATION:")
        print("-"*60)
        param_strs = []
        for key, value in results.items():
            if isinstance(value, float):
                param_strs.append(f"{key}={value:.6f}")
            else:
                param_strs.append(f"{key}={value}")
        
        # Split into chunks for readability
        chunk_size = 5
        for i in range(0, len(param_strs), chunk_size):
            print("  " + " | ".join(param_strs[i:i+chunk_size]))

if __name__ == "__main__":
    main()