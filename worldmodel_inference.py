#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Magicconch with cc
Created: 2026-05-15
Last modified: 2026-05-15

Clinical World Model standalone inference script.

Purpose:
- Load a trained Clinical World Model (State Model + Outcome Model).
- Run a step-by-step state-prediction demo on one ICU sepsis patient trajectory:
    * Given a history window of obs / mask / static / action, call the State Model
      to predict the next-step dynamic features + ventilation probability.
    * At the last step, call the Outcome Model to predict 90-day mortality and
      output release / death.
- Print predicted vs. real values side by side at each step, so the World Model
  fit quality is easy to inspect.

Usage:
    python worldmodel_inference.py
    python worldmodel_inference.py --test
    python worldmodel_inference.py --case_path ./test_data/test_case.pkl --case_idx 0
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

# === Default paths ===
WORLDMODEL_DIR_DEFAULT = "./worldmodel"
TEST_CASE_PATH_DEFAULT = "./test_data/test_case.pkl"

# === Model hyperparameters (kept consistent with training) ===
HIDDEN_DIM = 128
GRU_LAYERS = 2
DROPOUT = 0.2
STATIC_EMBED_DIM = 32
ACTION_EMBED_DIM = 32

# === Window sizes (kept consistent with training) ===
STATE_WINDOW_SIZE = 6     # State Model input window
OUTCOME_WINDOW_SIZE = 12  # Outcome Model input window

# === Test mode: only show the first 5 steps ===
TEST_MODE_STEPS = 5

# ============================================================
# Feature definitions
# ============================================================

VITAL_COLS = [
    "heart_rate", "sysbp", "diabp", "meanbp",
    "resp_rate", "spo2", "temp_c", "fio2",
    "gcs_eye", "gcs_verbal", "gcs_motor"
]
LAB_COLS = [
    "albumin", "alt", "ast", "base_excess", "bicarbonate",
    "bilirubin_total", "bun", "calcium_free", "calcium_total",
    "chloride", "creatinine", "glucose", "hemoglobin", "hematocrit",
    "inr", "lactate", "magnesium", "paco2", "pao2", "ph",
    "platelet", "potassium", "pt", "ptt", "sodium", "total_co2", "wbc"
]
URINE_COLS = ["output_4hourly"]
URINE_IDX = 38

DYNAMIC_COLS = VITAL_COLS + LAB_COLS + URINE_COLS
MASK_COLS = VITAL_COLS + LAB_COLS
STATIC_COLS = ["age", "gender", "charlson_comorbidity_index"]
ACT_ONEHOT_COLS = [f"act_{i}" for i in range(10)]

DYN_DIM = len(DYNAMIC_COLS)
MASK_DIM = len(MASK_COLS)
STATIC_DIM = len(STATIC_COLS)
ACT_DIM = len(ACT_ONEHOT_COLS)

VASO_LEVELS = ["None", "Low", "Medium", "High", "Very High"]
FLUID_LEVELS = ["None", "Low", "Medium", "High", "Very High"]


# ============================================================
# WorldModel definitions
# ============================================================

class SepsisStateModel(nn.Module):
    """State Model - predicts the next state (Gaussian output) + ventilation probability."""
    def __init__(self, dyn_dim, mask_dim, static_dim, act_dim, hidden=128,
                 n_layers=2, dropout=0.2, static_embed=32, act_embed=32):
        super().__init__()
        self.static_embed = nn.Linear(static_dim, static_embed)
        self.act_embed = nn.Linear(act_dim, act_embed)
        input_dim = dyn_dim + mask_dim + static_embed + act_embed
        self.gru = nn.GRU(input_dim, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.vent_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.trans_pre = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.mu_head = nn.Linear(hidden, dyn_dim)
        self.log_sigma_head = nn.Linear(hidden, dyn_dim)
        self.hidden_dim = hidden
        self.dyn_dim = dyn_dim

    def forward(self, obs, mask, stat, act):
        B, T, D = obs.shape
        stat_e = self.static_embed(stat).unsqueeze(1).expand(-1, T, -1)
        act_e = self.act_embed(act)
        x = torch.cat([obs, mask, stat_e, act_e], dim=2)
        h, _ = self.gru(x)
        h_last = h[:, -1, :]
        vent_logits = self.vent_head(h_last)
        vent_pred = torch.sigmoid(vent_logits)
        h_aug = torch.cat([h_last, vent_pred], dim=1)
        h_pre = self.trans_pre(h_aug)
        mu = self.mu_head(h_pre)
        log_sigma = self.log_sigma_head(h_pre)
        return mu, log_sigma, vent_logits

    def sample(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma


class OutcomeModel(nn.Module):
    """Outcome Model - predicts 90-day mortality."""
    def __init__(self, dyn_dim, mask_dim, static_dim, act_dim, hidden=128,
                 n_layers=2, dropout=0.2, static_embed=32, act_embed=32):
        super().__init__()
        self.static_embed = nn.Linear(static_dim, static_embed)
        self.act_embed = nn.Linear(act_dim, act_embed)
        input_dim = dyn_dim + mask_dim + static_embed + act_embed
        self.gru = nn.GRU(input_dim, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.hidden_dim = hidden

    def forward(self, obs, mask, stat, act):
        B, T, D = obs.shape
        stat_e = self.static_embed(stat).unsqueeze(1).expand(-1, T, -1)
        act_e = self.act_embed(act)
        x = torch.cat([obs, mask, stat_e, act_e], dim=2)
        h, _ = self.gru(x)
        h_last = h[:, -1, :]
        logits = self.outcome_head(h_last).squeeze(-1)
        return logits


# ============================================================
# WorldModel wrapper class
# ============================================================

class WorldModel:
    """WorldModel wrapper - manages State Model and Outcome Model."""

    def __init__(self, model_dir: str, device: torch.device):
        self.device = device
        self.model_dir = model_dir

        scaler_path = os.path.join(model_dir, "scaler_params_log.json")
        with open(scaler_path, 'r') as f:
            self.scaler_params = json.load(f)

        self.urine_log_transform = self.scaler_params.get("urine_log_transform", False)
        self.urine_idx = self.scaler_params.get("urine_idx", URINE_IDX)

        feature_config_path = os.path.join(model_dir, "episode_feature_config.json")
        if os.path.exists(feature_config_path):
            with open(feature_config_path, 'r') as f:
                self.feature_config = json.load(f)
        else:
            self.feature_config = {
                "dynamic_cols": DYNAMIC_COLS,
                "mask_cols": [f"{col}_mask" for col in MASK_COLS],
                "static_cols": STATIC_COLS,
                "act_onehot_cols": ACT_ONEHOT_COLS,
            }

        self.model_kwargs = {
            "dyn_dim": DYN_DIM, "mask_dim": MASK_DIM, "static_dim": STATIC_DIM,
            "act_dim": ACT_DIM, "hidden": HIDDEN_DIM, "n_layers": GRU_LAYERS,
            "dropout": DROPOUT, "static_embed": STATIC_EMBED_DIM, "act_embed": ACTION_EMBED_DIM,
        }

        self.state_model = self._load_state_model()
        self.outcome_model = self._load_outcome_model()

    def _load_state_model(self):
        state_path = os.path.join(self.model_dir, 'state_model_log.pt')
        print(f"  Loading State Model from: {state_path}")
        model = SepsisStateModel(**self.model_kwargs).to(self.device)
        model.load_state_dict(torch.load(state_path, map_location=self.device))
        model.eval()
        return model

    def _load_outcome_model(self):
        outcome_path = os.path.join(self.model_dir, "outcome.pt")
        print(f"  Loading Outcome Model from: {outcome_path}")
        model = OutcomeModel(**self.model_kwargs).to(self.device)
        model.load_state_dict(torch.load(outcome_path, map_location=self.device))
        model.eval()
        return model

    def predict_next_state(self, obs_window, mask_window, static_features, action_window):
        """Predict the next-step dynamic features (random sample)."""
        with torch.no_grad():
            obs = torch.FloatTensor(obs_window).unsqueeze(0).to(self.device)
            mask = torch.FloatTensor(mask_window).unsqueeze(0).to(self.device)
            stat = torch.FloatTensor(static_features).unsqueeze(0).to(self.device)
            act = torch.FloatTensor(action_window).unsqueeze(0).to(self.device)
            mu, log_sigma, _ = self.state_model(obs, mask, stat, act)
            next_state = self.state_model.sample(mu, log_sigma)
            return next_state.cpu().numpy()[0]

    def predict_next_state_with_vent(self, obs_window, mask_window, static_features, action_window):
        """Predict the next-step dynamic features and also return the ventilation probability."""
        with torch.no_grad():
            obs = torch.FloatTensor(obs_window).unsqueeze(0).to(self.device)
            mask = torch.FloatTensor(mask_window).unsqueeze(0).to(self.device)
            stat = torch.FloatTensor(static_features).unsqueeze(0).to(self.device)
            act = torch.FloatTensor(action_window).unsqueeze(0).to(self.device)
            mu, log_sigma, vent_logits = self.state_model(obs, mask, stat, act)
            next_state = self.state_model.sample(mu, log_sigma)
            vent_prob = torch.sigmoid(vent_logits).item()
            return next_state.cpu().numpy()[0], vent_prob

    def predict_outcome(self, obs_window, mask_window, static_features, action_window):
        """Predict 90-day mortality from a trajectory window."""
        with torch.no_grad():
            obs = torch.FloatTensor(obs_window).unsqueeze(0).to(self.device)
            mask = torch.FloatTensor(mask_window).unsqueeze(0).to(self.device)
            stat = torch.FloatTensor(static_features).unsqueeze(0).to(self.device)
            act = torch.FloatTensor(action_window).unsqueeze(0).to(self.device)
            logits = self.outcome_model(obs, mask, stat, act)
            death_prob = torch.sigmoid(logits).item()
            outcome = 'death' if death_prob > 0.5 else 'release'
            return outcome, death_prob

    def denormalize_dynamic(self, normalized_value, feature_name):
        """Restore a standardized dynamic feature back to its raw clinical value."""
        idx = DYNAMIC_COLS.index(feature_name)
        mean = self.scaler_params["dyn_mean"][idx]
        std = self.scaler_params["dyn_std"][idx]
        unstd_value = normalized_value * std + mean
        if feature_name == "output_4hourly" and self.urine_log_transform:
            orig_value = np.expm1(unstd_value)
            orig_value = max(0.0, orig_value)
            return orig_value
        return unstd_value

    def denormalize_static(self, normalized_value, feature_name, static_scaler_mean, static_scaler_std):
        """Restore a standardized static feature back to its raw clinical value."""
        if feature_name == 'age':
            return normalized_value * static_scaler_std[0] + static_scaler_mean[0]
        elif feature_name == 'charlson_comorbidity_index':
            return normalized_value * static_scaler_std[1] + static_scaler_mean[1]
        return normalized_value


# ============================================================
# Helpers
# ============================================================

def pad_to_window(data_list, window_size):
    """Pad a list of arrays to the specified window size; pad with zeros at the front if too short."""
    if len(data_list) >= window_size:
        return np.array(data_list[-window_size:], dtype=np.float32)
    pad_size = window_size - len(data_list)
    feature_dim = data_list[0].shape[0]
    padding = [np.zeros(feature_dim, dtype=np.float32) for _ in range(pad_size)]
    return np.array(padding + list(data_list), dtype=np.float32)


def format_compare_row(name, pred_val, real_val, unit=""):
    """Format one row of predicted vs. real comparison."""
    diff = pred_val - real_val
    return f"  {name:<20} pred={pred_val:>8.2f}  real={real_val:>8.2f}  diff={diff:>+7.2f}  {unit}"


def print_step_comparison(world_model, t, action, next_obs_pred, next_obs_real, vent_prob,
                          highlight_cols=None):
    """Print predicted vs. real comparison for one step (highlighted columns)."""
    if highlight_cols is None:
        highlight_cols = ['heart_rate', 'meanbp', 'sysbp', 'spo2', 'lactate',
                          'creatinine', 'platelet', 'pao2', 'output_4hourly']

    vaso, fluid = action
    print(f"\n[Step t={t} -> t={t+1}]  Action: vaso={VASO_LEVELS[vaso]}({vaso}), "
          f"fluid={FLUID_LEVELS[fluid]}({fluid})")
    print(f"  Predicted ventilation prob: {vent_prob:.3f} "
          f"({'on vent' if vent_prob > 0.5 else 'off vent'})")
    print(f"  {'Feature':<20} {'Pred':>10}  {'Real':>10}  {'Diff':>10}")
    for col in highlight_cols:
        idx = DYNAMIC_COLS.index(col)
        pred_v = world_model.denormalize_dynamic(next_obs_pred[idx], col)
        real_v = world_model.denormalize_dynamic(next_obs_real[idx], col)
        if col == 'output_4hourly':
            pred_v = max(0.0, pred_v)
            real_v = max(0.0, real_v)
        print(format_compare_row(col, pred_v, real_v))


# ============================================================
# Inference main flow
# ============================================================

def run_worldmodel_inference(case_path, worldmodel_dir, case_idx=0, test_mode=False, device=None):
    """Run a step-by-step World Model prediction on a real episode."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading World Model from {worldmodel_dir}...")
    world_model = WorldModel(worldmodel_dir, device)

    print(f"\nLoading test case from {case_path}...")
    with open(case_path, 'rb') as f:
        episodes = pickle.load(f)
    print(f"  Loaded {len(episodes)} episode(s)")

    if case_idx >= len(episodes):
        raise IndexError(f"case_idx={case_idx} out of range (only {len(episodes)} episodes)")
    ep = episodes[case_idx]

    stay_id = ep['stay_id']
    obs_all = ep['obs']
    mask_all = ep['mask']
    static = ep['static']
    actions_all = ep['actions']
    vaso_all = ep['vaso_actions']
    fluid_all = ep['fluid_actions']
    real_outcome = ep.get('outcome', 'unknown')
    mortality_90d = ep.get('mortality_90d', None)
    total_steps = len(obs_all)

    print(f"\n{'=' * 70}")
    print(f"Episode info")
    print(f"{'=' * 70}")
    print(f"  stay_id:        {stay_id}")
    print(f"  trajectory:     {total_steps} steps ({total_steps * 4} hours)")
    print(f"  real outcome:   {real_outcome}")
    print(f"  mortality_90d:  {mortality_90d}")

    # De-standardize static features
    static_mean = world_model.feature_config.get("static_scaler_mean", [65.0, 5.0])
    static_std = world_model.feature_config.get("static_scaler_std", [15.0, 3.0])
    age = world_model.denormalize_static(static[0], 'age', static_mean, static_std)
    gender = "Male" if static[1] > 0.5 else "Female"
    charlson = world_model.denormalize_static(static[2], 'charlson_comorbidity_index',
                                              static_mean, static_std)
    print(f"  age:            {age:.0f}")
    print(f"  gender:         {gender}")
    print(f"  charlson:       {charlson:.1f}")

    # ========= State Model: step-by-step prediction =========
    print(f"\n{'=' * 70}")
    print(f"State Model: step-by-step prediction (teacher forcing)")
    print(f"{'=' * 70}")

    obs_history = [obs_all[0]]
    mask_history = [mask_all[0]]
    action_history = [np.zeros(ACT_DIM, dtype=np.float32)]

    n_steps_to_show = min(TEST_MODE_STEPS, total_steps - 1) if test_mode else total_steps - 1

    for t in range(n_steps_to_show):
        vaso = int(vaso_all[t])
        fluid = int(fluid_all[t])
        action_onehot = actions_all[t]
        action_history[-1] = action_onehot  # current-step action

        # Build the input windows
        obs_window = pad_to_window(obs_history, STATE_WINDOW_SIZE)
        mask_window = pad_to_window(mask_history, STATE_WINDOW_SIZE)
        action_window = pad_to_window(action_history, STATE_WINDOW_SIZE)

        # Predict the next state
        next_obs_pred, vent_prob = world_model.predict_next_state_with_vent(
            obs_window, mask_window, static, action_window
        )
        next_obs_real = obs_all[t + 1]

        print_step_comparison(world_model, t, (vaso, fluid),
                              next_obs_pred, next_obs_real, vent_prob)

        # Teacher forcing: advance the window with the real next observation
        obs_history.append(obs_all[t + 1])
        mask_history.append(mask_all[t + 1])
        action_history.append(np.zeros(ACT_DIM, dtype=np.float32))

    # ========= Outcome Model: terminal prediction =========
    print(f"\n{'=' * 70}")
    print(f"Outcome Model: 90-day mortality prediction")
    print(f"{'=' * 70}")

    obs_full = list(obs_all)
    mask_full = list(mask_all)
    action_full = [actions_all[i] for i in range(total_steps)]

    outcome_obs = pad_to_window(obs_full, OUTCOME_WINDOW_SIZE)
    outcome_mask = pad_to_window(mask_full, OUTCOME_WINDOW_SIZE)
    outcome_actions = pad_to_window(action_full, OUTCOME_WINDOW_SIZE)

    pred_outcome, death_prob = world_model.predict_outcome(
        outcome_obs, outcome_mask, static, outcome_actions
    )

    print(f"  Predicted death probability: {death_prob:.4f}")
    print(f"  Predicted outcome:           {pred_outcome}")
    print(f"  Real outcome:                {real_outcome}")
    print(f"  Match:                       {'YES' if pred_outcome == real_outcome else 'NO'}")

    print(f"\n{'=' * 70}")
    print(f"Done. State Model showed {n_steps_to_show} step(s); "
          f"Outcome Model evaluated full {total_steps}-step trajectory.")
    print(f"{'=' * 70}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Standalone Clinical World Model inference demo'
    )
    parser.add_argument('--worldmodel_dir', type=str, default=WORLDMODEL_DIR_DEFAULT,
                        help='Directory holding World Model weights and configs')
    parser.add_argument('--case_path', type=str, default=TEST_CASE_PATH_DEFAULT,
                        help='Path to the test-case pickle file')
    parser.add_argument('--case_idx', type=int, default=0,
                        help='Which episode in the pickle to use (default 0)')
    parser.add_argument('--test', action='store_true',
                        help=f'Test mode: only show the first {TEST_MODE_STEPS} prediction steps')
    parser.add_argument('--device', type=str, default=None,
                        help="Compute device (e.g. 'cuda', 'cuda:0', 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else None
    run_worldmodel_inference(
        case_path=args.case_path,
        worldmodel_dir=args.worldmodel_dir,
        case_idx=args.case_idx,
        test_mode=args.test,
        device=device,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())


# ============================================================
# Run commands:
#   python worldmodel_inference.py --test
#   python worldmodel_inference.py
#   python worldmodel_inference.py --case_path ./test_data/test_case.pkl --case_idx 0
# ============================================================
