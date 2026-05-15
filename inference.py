#!/usr/bin/env python3
"""
Sepsis Treatment Agent Workflow - automated evaluation script

Features:
- Automatically launch and manage vLLM services (one per GPU)
- Supports `simulation` and `prescription` function calls via the OpenAI tools API
- Uses `mortality_90d` as the ground-truth outcome
- Matches per-clinician reward by `stay_id`
- Multi-process parallelism (one worker per GPU)
- Resumable runs based on `stay_id`
- Logs failed episodes and supports retrying them
- Prompt format is fully aligned with the training data

Usage:
    python inference.py --model_path /path/to/model --model_name Test --num_gpus 8
    python inference.py --model_path /path/to/model --model_name Test --num_gpus 8 --test
    python inference.py --model_path /path/to/model --model_name Test --num_gpus 8 --retry_failed
"""

import os
import sys
import json
import argparse
import subprocess
import signal
import time
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import shutil
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import traceback
import requests
import atexit
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

# === Run parameters ===
TEST_MODE_NUM_EPISODES = 4        # In test mode, only run the first 4 episodes
API_TIMEOUT = 180                 # API timeout (seconds)
API_MAX_RETRIES = 3               # Maximum number of API retries
VLLM_STARTUP_TIMEOUT = 300        # vLLM startup timeout (seconds)
VLLM_HEALTH_CHECK_INTERVAL = 5    # vLLM health-check interval (seconds)

# === Global logging control ===
VERBOSE_MODE = True               # Whether to print detailed logs


def log_verbose(msg):
    """Only print when verbose mode is enabled."""
    if VERBOSE_MODE:
        tqdm.write(msg)

# === Multi-GPU parameters ===
DEFAULT_NUM_GPUS = 8              # Default number of GPUs
DEFAULT_BASE_PORT = 8000          # Base port; GPU i uses BASE_PORT + i

# === vLLM API config ===
VLLM_HOST = "localhost"

# === Model parameters ===
STATE_WINDOW_SIZE = 6             # State Model context window length
OUTCOME_WINDOW_SIZE = 12          # Outcome Model context window length (48h)
START_STEP = 0                    # Start decision-making from step 0

# === Paths ===
WORLDMODEL_DIR = "./worldmodel"
PREPROCESSED_DATA_PATH = "./test_data/test_case.pkl"
REAL_REWARDS_PATH = "./test_data/real_episode_rewards_test_case.json"
OUTPUT_DIR = "./output"
DEBUG_LOG_DIR = "./output/debug_logs"

# === Reward parameters ===
REWARD_RELEASE = 15.0             # Discharge reward
REWARD_DEATH = -15.0              # Death penalty
SOFA_PENALTY_SAME = -0.025        # Penalty for unchanged SOFA
SOFA_CHANGE_COEF = -0.125         # SOFA change coefficient
LACTATE_CHANGE_COEF = -2.0        # Lactate change coefficient

# === Simulation parameters ===
MAX_SIMULATION_ROUNDS = 3         # Max simulation rounds per step (used only for log warnings)
MAX_ACTIONS_PER_SIMULATION = 3    # Max actions per simulation call
MAX_SIMULATION_BEFORE_SKIP = 10   # Skip the episode if consecutive simulations exceed this

# === OpenAI tool definitions ===
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "prescription",
            "description": "Execute the final treatment decision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vasopressor": {
                        "type": "integer",
                        "description": "Vasopressor level (0-4)"
                    },
                    "iv_fluid": {
                        "type": "integer",
                        "description": "IV Fluid level (0-4)"
                    }
                },
                "required": ["vasopressor", "iv_fluid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulation",
            "description": "Simulate patient outcomes for different treatment actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "description": "List of treatment actions to simulate.",
                        "items": {"type": "string"}
                    }
                },
                "required": ["actions"]
            }
        }
    }
]

# === Model hyperparameters (kept consistent with training) ===
HIDDEN_DIM = 128
GRU_LAYERS = 2
DROPOUT = 0.2
STATIC_EMBED_DIM = 32
ACTION_EMBED_DIM = 32


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

FEATURE_UNITS = {
    'heart_rate': 'bpm', 'sysbp': 'mmHg', 'diabp': 'mmHg', 'meanbp': 'mmHg',
    'resp_rate': 'breaths/min', 'spo2': '%', 'temp_c': '°C',
    'fio2': '%', 'gcs_eye': 'score', 'gcs_verbal': 'score', 'gcs_motor': 'score',
    'albumin': 'g/dL', 'alt': 'U/L', 'ast': 'U/L', 'base_excess': 'mEq/L',
    'bicarbonate': 'mEq/L', 'bilirubin_total': 'mg/dL', 'bun': 'mg/dL',
    'calcium_free': 'mmol/L', 'calcium_total': 'mg/dL', 'chloride': 'mEq/L',
    'creatinine': 'mg/dL', 'glucose': 'mg/dL', 'hemoglobin': 'g/dL',
    'hematocrit': '%', 'inr': 'ratio', 'lactate': 'mmol/L', 'magnesium': 'mEq/L',
    'paco2': 'mmHg', 'pao2': 'mmHg', 'ph': '', 'platelet': 'K/μL',
    'potassium': 'mEq/L', 'pt': 'sec', 'ptt': 'sec', 'sodium': 'mEq/L',
    'total_co2': 'mEq/L', 'wbc': 'K/μL',
    'output_4hourly': 'mL/4h',
    'age': 'years', 'gender': '', 'charlson_comorbidity_index': 'score',
}

DISPLAY_VITALS = ['heart_rate', 'sysbp', 'diabp', 'meanbp', 'resp_rate', 'spo2', 'temp_c', 'fio2']
DISPLAY_LABS = [
    'albumin', 'alt', 'ast', 'base_excess', 'bicarbonate', 'bilirubin_total', 'bun',
    'calcium_free', 'calcium_total', 'chloride', 'creatinine', 'glucose',
    'hemoglobin', 'hematocrit', 'lactate', 'magnesium', 'paco2', 'pao2', 'ph',
    'platelet', 'potassium', 'pt', 'ptt', 'sodium', 'total_co2', 'wbc'
]


# ============================================================
# Global state: vLLM process management
# ============================================================
VLLM_PROCESSES = []


def cleanup_vllm_processes():
    """Tear down all vLLM processes."""
    global VLLM_PROCESSES
    tqdm.write("\n[Cleanup] Stopping vLLM services...")
    for proc in VLLM_PROCESSES:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except:
                proc.kill()
    VLLM_PROCESSES = []
    tqdm.write("[Cleanup] All vLLM services stopped")


atexit.register(cleanup_vllm_processes)


def signal_handler(signum, frame):
    """Signal handler."""
    tqdm.write(f"\n[Signal] Received signal {signum}, cleaning up...")
    cleanup_vllm_processes()
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================
# vLLM service management
# ============================================================

def start_vllm_service(gpu_id: int, model_path: str, model_name: str,
                       base_port: int, chat_template: str = None,
                       gpu_memory_utilization: float = 0.6,
                       max_model_len: int = None) -> subprocess.Popen:
    """Start a single vLLM service."""
    port = base_port + gpu_id

    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--served-model-name", model_name,
        "--reasoning-parser", "qwen3",
        "--tensor-parallel-size", "1",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]

    # Add max context length argument
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])

    if chat_template and os.path.exists(chat_template):
        cmd.extend(["--chat-template", chat_template])

    env = os.environ.copy()
    # Resolve the actual GPU ID (respecting CUDA_VISIBLE_DEVICES)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_devices:
        # If CUDA_VISIBLE_DEVICES is set, pick the gpu_id-th device from it
        device_list = visible_devices.split(",")
        if gpu_id < len(device_list):
            env["CUDA_VISIBLE_DEVICES"] = device_list[gpu_id].strip()
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_dir = os.path.join(OUTPUT_DIR, "vllm_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"vllm_gpu{gpu_id}.log")

    log_verbose(f"[vLLM] Starting service on GPU {gpu_id}, port {port}...")
    log_verbose(f"[vLLM] Log file: {log_file}")

    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )

    return proc


def check_vllm_health(port: int, timeout: int = 5) -> bool:
    """Check whether a vLLM service is healthy."""
    try:
        url = f"http://{VLLM_HOST}:{port}/health"
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False


def wait_for_vllm_services(base_port: int, num_gpus: int,
                           timeout: int = VLLM_STARTUP_TIMEOUT) -> bool:
    """Wait until all vLLM services are ready."""
    tqdm.write(f"\n[vLLM] Waiting for {num_gpus} services to start...")
    start_time = time.time()

    ready = [False] * num_gpus

    while time.time() - start_time < timeout:
        all_ready = True
        for i in range(num_gpus):
            if not ready[i]:
                port = base_port + i
                if check_vllm_health(port):
                    ready[i] = True
                    log_verbose(f"[vLLM] GPU {i} (port {port}) is ready")
                else:
                    all_ready = False

        if all_ready:
            tqdm.write(f"[vLLM] All {num_gpus} services are ready!")
            return True

        time.sleep(VLLM_HEALTH_CHECK_INTERVAL)
        elapsed = int(time.time() - start_time)
        ready_count = sum(ready)
        log_verbose(f"[vLLM] Waiting... {ready_count}/{num_gpus} ready, {elapsed}s elapsed")

    tqdm.write(f"[vLLM] Timeout! Only {sum(ready)}/{num_gpus} services started")
    return False


def start_all_vllm_services(model_path: str, model_name: str,
                            num_gpus: int, base_port: int,
                            chat_template: str = None,
                            gpu_memory_utilization: float = 0.6,
                            max_model_len: int = None) -> bool:
    """Start all vLLM services."""
    global VLLM_PROCESSES

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Starting {num_gpus} vLLM services...")
    tqdm.write(f"Model: {model_path}")
    tqdm.write(f"Model Name: {model_name}")
    tqdm.write(f"Base Port: {base_port}")
    tqdm.write(f"GPU Memory Utilization: {gpu_memory_utilization}")
    tqdm.write(f"Max Model Length: {max_model_len if max_model_len else 'default'}")
    tqdm.write(f"{'='*60}")

    # Check model path
    if not os.path.exists(model_path):
        tqdm.write(f"[ERROR] Model path does not exist: {model_path}")
        return False

    # Auto-detect chat_template
    if chat_template is None:
        default_template = os.path.join(model_path, "chat_template.jinja")
        if os.path.exists(default_template):
            chat_template = default_template
            log_verbose(f"[vLLM] Found chat template: {chat_template}")

    # Launch all services
    for gpu_id in range(num_gpus):
        proc = start_vllm_service(gpu_id, model_path, model_name,
                                  base_port, chat_template, gpu_memory_utilization,
                                  max_model_len)
        VLLM_PROCESSES.append(proc)
        time.sleep(2)  # Stagger startup

    # Wait for services to come up
    if not wait_for_vllm_services(base_port, num_gpus):
        cleanup_vllm_processes()
        return False

    return True


# ============================================================
# WorldModel definitions
# ============================================================

class SepsisStateModel(nn.Module):
    """State Model - predicts the next state (Gaussian output)."""
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
        log_verbose(f"  Loading State Model from: {state_path}")
        model = SepsisStateModel(**self.model_kwargs).to(self.device)
        model.load_state_dict(torch.load(state_path, map_location=self.device))
        model.eval()
        return model

    def _load_outcome_model(self):
        outcome_path = os.path.join(self.model_dir, "outcome.pt")
        log_verbose(f"  Loading Outcome Model from: {outcome_path}")
        model = OutcomeModel(**self.model_kwargs).to(self.device)
        model.load_state_dict(torch.load(outcome_path, map_location=self.device))
        model.eval()
        return model

    def predict_next_state(self, obs_window, mask_window, static_features, action_window):
        with torch.no_grad():
            obs = torch.FloatTensor(obs_window).unsqueeze(0).to(self.device)
            mask = torch.FloatTensor(mask_window).unsqueeze(0).to(self.device)
            stat = torch.FloatTensor(static_features).unsqueeze(0).to(self.device)
            act = torch.FloatTensor(action_window).unsqueeze(0).to(self.device)
            mu, log_sigma, _ = self.state_model(obs, mask, stat, act)
            next_state = self.state_model.sample(mu, log_sigma)
            return next_state.cpu().numpy()[0]

    def predict_next_state_with_vent(self, obs_window, mask_window, static_features, action_window):
        """Predict the next state and also return the ventilation probability."""
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
        if feature_name == 'age':
            return normalized_value * static_scaler_std[0] + static_scaler_mean[0]
        elif feature_name == 'charlson_comorbidity_index':
            return normalized_value * static_scaler_std[1] + static_scaler_mean[1]
        return normalized_value


# ============================================================
# SOFA computation (hard-threshold version)
# ============================================================

def compute_sofa_from_obs(next_obs, vent_prob, vaso_level, past_24h_urine, world_model):
    """
    Compute total SOFA score from predicted dynamics (hard-threshold version).

    Args:
        next_obs: 39-d dynamic features in standardized space
        vent_prob: ventilation probability (>0.5 considered ventilated)
        vaso_level: current vasopressor level (0-4)
        past_24h_urine: cumulative urine output over the past 24 hours (mL)
        world_model: used for de-standardization

    Returns:
        sofa_total: total SOFA score (0-24)
    """
    # Helper to get a value in raw clinical space
    def get_val(col):
        idx = DYNAMIC_COLS.index(col)
        return world_model.denormalize_dynamic(next_obs[idx], col)

    # Pull out key variables
    pao2 = get_val('pao2')
    fio2 = max(21.0, get_val('fio2'))  # FiO2 must be at least 21%
    platelet = get_val('platelet')
    bilirubin = get_val('bilirubin_total')
    meanbp = get_val('meanbp')
    creatinine = get_val('creatinine')
    gcs_eye = get_val('gcs_eye')
    gcs_verbal = get_val('gcs_verbal')
    gcs_motor = get_val('gcs_motor')

    is_vent = vent_prob > 0.5

    # 1. SOFA respiratory (0-4): based on PF ratio and ventilation
    pf_ratio = pao2 / (fio2 / 100.0)
    if pf_ratio >= 400:
        sofa_resp = 0
    elif pf_ratio >= 300:
        sofa_resp = 1
    elif pf_ratio >= 200:
        sofa_resp = 2 if not is_vent else 2
    elif pf_ratio >= 100:
        sofa_resp = 3 if is_vent else 2
    else:
        sofa_resp = 4 if is_vent else 2

    # 2. SOFA coagulation (0-4): based on platelets
    if platelet >= 150:
        sofa_coag = 0
    elif platelet >= 100:
        sofa_coag = 1
    elif platelet >= 50:
        sofa_coag = 2
    elif platelet >= 20:
        sofa_coag = 3
    else:
        sofa_coag = 4

    # 3. SOFA liver (0-4): based on bilirubin
    if bilirubin < 1.2:
        sofa_liver = 0
    elif bilirubin < 2.0:
        sofa_liver = 1
    elif bilirubin < 6.0:
        sofa_liver = 2
    elif bilirubin < 12.0:
        sofa_liver = 3
    else:
        sofa_liver = 4

    # 4. SOFA cardiovascular (0-4): based on MAP and vasopressors
    # 1 point if MAP < 70, plus the vaso level
    sofa_cardio = (1 if meanbp < 70 else 0) + vaso_level
    sofa_cardio = min(4, sofa_cardio)

    # 5. SOFA CNS (0-4): based on GCS
    gcs_total = gcs_eye + gcs_verbal + gcs_motor
    if gcs_total >= 15:
        sofa_cns = 0
    elif gcs_total >= 13:
        sofa_cns = 1
    elif gcs_total >= 10:
        sofa_cns = 2
    elif gcs_total >= 6:
        sofa_cns = 3
    else:
        sofa_cns = 4

    # 6. SOFA renal (0-4): based on creatinine and 24h urine output, take the worse one
    # Creatinine sub-score
    if creatinine < 1.2:
        cr_score = 0
    elif creatinine < 2.0:
        cr_score = 1
    elif creatinine < 3.5:
        cr_score = 2
    elif creatinine < 5.0:
        cr_score = 3
    else:
        cr_score = 4

    # Urine output sub-score
    if past_24h_urine < 200:
        uo_score = 4
    elif past_24h_urine < 500:
        uo_score = 3
    else:
        uo_score = 0

    sofa_renal = max(cr_score, uo_score)

    # Total
    sofa_total = sofa_resp + sofa_coag + sofa_liver + sofa_cardio + sofa_cns + sofa_renal
    return sofa_total


def check_guideline_compliance(current_obs, vaso, fluid, prev_vaso, world_model):
    """
    Check medical guideline compliance (called after the model issues a prescription).

    Guideline 1 (Hypoperfusion Rule):
        When Lactate > 2 OR MAP < 65, the action must not be (0, 0)
    Guideline 2 (Vasopressor Weaning Rule):
        When MAP >= 65, the vasopressor dose must not be increased (vaso must not be > prev_vaso)

    Args:
        current_obs: current standardized observation (state seen at decision time)
        vaso: vasopressor level chosen by the model (0-4)
        fluid: IV fluid level chosen by the model (0-4)
        prev_vaso: previous-step vasopressor level (0-4); None at t=0
        world_model: used for de-standardization

    Returns:
        dict: {
            'hypoperfusion_violated': bool,  # whether guideline 1 is violated
            'vaso_weaning_violated': bool,   # whether guideline 2 is violated
            'lactate': float,                # current lactate value
            'map': float,                    # current MAP value
            'is_hypoperfusion': bool         # whether the patient is in hypoperfusion
        }
    """
    # Get values in raw clinical space
    lactate_idx = DYNAMIC_COLS.index('lactate')
    map_idx = DYNAMIC_COLS.index('meanbp')

    lactate = world_model.denormalize_dynamic(current_obs[lactate_idx], 'lactate')
    meanbp = world_model.denormalize_dynamic(current_obs[map_idx], 'meanbp')

    # Determine hypoperfusion state
    is_hypoperfusion = (lactate > 2.0) or (meanbp < 65.0)

    # Guideline 1: cannot use (0, 0) when in hypoperfusion
    hypoperfusion_violated = is_hypoperfusion and (vaso == 0 and fluid == 0)

    # Guideline 2: vaso must not increase when MAP >= 65
    # Only check when prev_vaso is not None (i.e. not the first step)
    vaso_weaning_violated = False
    if prev_vaso is not None and meanbp >= 65.0:
        vaso_weaning_violated = (vaso > prev_vaso)

    return {
        'hypoperfusion_violated': hypoperfusion_violated,
        'vaso_weaning_violated': vaso_weaning_violated,
        'lactate': lactate,
        'map': meanbp,
        'is_hypoperfusion': is_hypoperfusion
    }


# ============================================================
# LLM Agent helper functions
# ============================================================

def build_system_prompt():
    """Build the system prompt."""
    return """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "simulation", "description": "Simulate patient outcomes for different treatment actions before making a final decision. Use this when you want to compare multiple treatment options.", "parameters": {"type": "object", "properties": {"actions": {"type": "array", "description": "List of treatment actions to simulate. Each action is '[vasopressor_level, iv_fluid_level]' where levels are 0-4. Maximum 3 actions per call.", "items": {"type": "string"}}}, "required": ["actions"]}}}
{"type": "function", "function": {"name": "prescription", "description": "Execute the final treatment decision. Use this when you are confident about the best treatment after analysis or simulation.", "parameters": {"type": "object", "properties": {"vasopressor": {"type": "integer", "description": "Vasopressor level (0-4): None(0), Low(1), Medium(2), High(3), Very High(4)"}, "iv_fluid": {"type": "integer", "description": "IV Fluid level (0-4): None(0), Low(1), Medium(2), High(3), Very High(4)"}}, "required": ["vasopressor", "iv_fluid"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def build_initial_user_prompt(static_features, state_history, world_model,
                               static_scaler_mean, static_scaler_std, current_step):
    """Build the initial user prompt."""
    age = world_model.denormalize_static(static_features[0], 'age', static_scaler_mean, static_scaler_std)
    gender = "Male" if static_features[1] > 0.5 else "Female"
    charlson = world_model.denormalize_static(static_features[2], 'charlson_comorbidity_index',
                                               static_scaler_mean, static_scaler_std)

    def format_value_history(col, states):
        idx = DYNAMIC_COLS.index(col)
        values = []
        for s in states:
            raw_val = s[idx]
            denorm_val = world_model.denormalize_dynamic(raw_val, col)
            if col == 'output_4hourly':
                denorm_val = max(0.0, denorm_val)
            values.append(denorm_val)
        value_str = ', '.join([f"{v:.1f}" for v in values])
        return f"[{value_str}]"

    hours_since_admission = current_step * 4

    lines = [
        "You are an expert ICU physician AI assistant specializing in sepsis treatment decisions.",
        "",
        "## Patient Information",
        f"- Age: {age:.0f} years",
        f"- Gender: {gender}",
        f"- Charlson Comorbidity Index: {charlson:.1f}",
        "",
        "## Important Notes",
        "- Patient vital signs and lab values are monitored every 4 hours",
        "- You will receive updates on patient status and make treatment decisions accordingly",
        "- Treatment actions take effect over the next 4-hour period",
        "- You are starting from ICU admission time (t=0)",
        "",
        "## Treatment Levels",
        "- IV Fluid: None (0), Low (1), Medium (2), High (3), Very High (4)",
        "- Vasopressor: None (0), Low (1), Medium (2), High (3), Very High (4)",
        "",
        "## Available Tools",
        "You have access to two tools:",
        "",
        "1. **simulation**: Simulate patient outcomes for different treatment actions before making a final decision.",
        '   - Parameter: actions (list of "[vasopressor_level, iv_fluid_level]" strings, max 3 actions)',
        "",
        "2. **prescription**: Execute the final treatment decision.",
        "   - Parameters: vasopressor (int 0-4), iv_fluid (int 0-4)",
        "",
        "## Clinical Protocols (Strict Adherence Required)",
        "1. **Emergency Priority**: Sepsis is a medical emergency. Hypoperfusion requires immediate resuscitation.",
        "2. **Early Resuscitation Rule (0-3h)**: Within the first 3 hours of admission (Hour 0), if there are signs of hypoperfusion, at least LOW-level IV fluid is MANDATORY.",
        "3. **Vasopressor Threshold**: If MAP remains < 65 mmHg after adequate fluid resuscitation, vasopressor support should be considered.",
        "4. **MAP Target**: For patients with septic shock on vasopressors, the initial target MAP is 65 mmHg rather than higher targets.",
        "5. **Definition of Septic Shock**: A patient is in 'Septic Shock' ONLY if ALL three conditions are met:",
        "   (a) Vasopressor level > 0 (Currently on vasopressors)",
        "   (b) MAP (meanbp) < 65 mmHg",
        "   (c) Lactate > 2 mmol/L",
        "",
        f"# Hour {hours_since_admission} Since ICU Admission (timestep t={current_step})",
        "",
        "## Vital Signs History",
    ]

    for col in DISPLAY_VITALS:
        unit = FEATURE_UNITS.get(col, '')
        lines.append(f"- {col}({unit}): {format_value_history(col, state_history)}")

    lines.append("")
    lines.append("## Laboratory Values History")

    for col in DISPLAY_LABS:
        unit = FEATURE_UNITS.get(col, '')
        lines.append(f"- {col}({unit}): {format_value_history(col, state_history)}")

    lines.append("")
    lines.append("## Urine Output History")
    unit = FEATURE_UNITS.get('output_4hourly', '')
    lines.append(f"- output_4hourly({unit}): {format_value_history('output_4hourly', state_history)}")

    lines.append("")
    lines.append("---")
    lines.append("You may call `simulation` to predict outcomes, or call `prescription` to make your final decision.")

    return "\n".join(lines)


def format_state_history_for_tool_response(state_history, action_history, world_model, current_step):
    """Format state history for the prescription tool_response."""

    def format_value_history(col, states):
        idx = DYNAMIC_COLS.index(col)
        values = []
        for s in states:
            raw_val = s[idx]
            denorm_val = world_model.denormalize_dynamic(raw_val, col)
            if col == 'output_4hourly':
                denorm_val = max(0.0, denorm_val)
            values.append(denorm_val)
        value_str = ', '.join([f"{v:.1f}" for v in values])
        return f"[{value_str}]"

    hours_since_admission = current_step * 4
    lines = [f"# Hour {hours_since_admission} Since ICU Admission"]

    lines.append("")
    lines.append("## Vital Signs History")
    for col in DISPLAY_VITALS:
        unit = FEATURE_UNITS.get(col, '')
        lines.append(f"- {col}({unit}): {format_value_history(col, state_history)}")

    lines.append("")
    lines.append("## Laboratory Values History")
    for col in DISPLAY_LABS:
        unit = FEATURE_UNITS.get(col, '')
        lines.append(f"- {col}({unit}): {format_value_history(col, state_history)}")

    lines.append("")
    lines.append("## Urine Output History")
    unit = FEATURE_UNITS.get('output_4hourly', '')
    lines.append(f"- output_4hourly({unit}): {format_value_history('output_4hourly', state_history)}")

    if action_history:
        lines.append("")
        lines.append("## Treatment History")
        for i, (vaso, fluid) in enumerate(action_history):
            hour = i * 4
            lines.append(f"- Hour {hour}: Vasopressor={VASO_LEVELS[vaso]}, IV Fluid={FLUID_LEVELS[fluid]}")

    lines.append("")
    lines.append("---")
    lines.append("You may call `simulation` again, or call `prescription` to make your final decision.")

    return "\n".join(lines)


def parse_tool_call_from_api_response(response):
    """Parse the tool call out of an OpenAI API response."""
    result = {
        "function_name": None,
        "arguments": {},
        "content": "",
        "tool_call_id": None
    }

    message = response.choices[0].message
    result["content"] = (message.content or "").rstrip()

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        result["function_name"] = tool_call.function.name
        result["tool_call_id"] = tool_call.id
        try:
            result["arguments"] = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            result["arguments"] = {}

    return result


def validate_simulation_actions(actions):
    """Validate and normalize simulation actions."""
    validated = []
    for act in actions[:3]:
        try:
            if isinstance(act, str):
                act = act.strip().strip('[]')
                parts = act.split(',')
                if len(parts) == 2:
                    vaso = int(parts[0].strip())
                    fluid = int(parts[1].strip())
                else:
                    continue
            elif isinstance(act, dict):
                vaso = int(act.get("vasopressor", 0))
                fluid = int(act.get("iv_fluid", 0))
            else:
                continue
            vaso = max(0, min(4, vaso))
            fluid = max(0, min(4, fluid))
            validated.append({"vasopressor": vaso, "iv_fluid": fluid})
        except Exception:
            continue
    return validated if validated else [{"vasopressor": 0, "iv_fluid": 0}]


def validate_prescription_args(args):
    """Validate and normalize prescription arguments."""
    try:
        vaso = int(args.get("vasopressor", 0))
        fluid = int(args.get("iv_fluid", 0))
        vaso = max(0, min(4, vaso))
        fluid = max(0, min(4, fluid))
        return vaso, fluid
    except Exception:
        return 0, 0


def action_to_onehot(vaso, fluid):
    """Convert an action to a 10-d one-hot vector."""
    onehot = np.zeros(10, dtype=np.float32)
    onehot[vaso] = 1.0
    onehot[5 + fluid] = 1.0
    return onehot


def pad_to_window(data_list, window_size):
    """Pad a list of arrays to the specified window size."""
    if len(data_list) >= window_size:
        return np.array(data_list[-window_size:], dtype=np.float32)
    else:
        pad_size = window_size - len(data_list)
        feature_dim = data_list[0].shape[0]
        padding = [np.zeros(feature_dim, dtype=np.float32) for _ in range(pad_size)]
        return np.array(padding + list(data_list), dtype=np.float32)


def simulate_single_action(vaso, fluid, obs_history, mask_history, action_history_onehot,
                           static, world_model, use_real_data, real_next_obs, real_vaso, real_fluid):
    """Simulate the next state for a single action."""
    if use_real_data and vaso == real_vaso and fluid == real_fluid and real_next_obs is not None:
        return real_next_obs

    action_onehot = action_to_onehot(vaso, fluid)
    temp_action_history = list(action_history_onehot)
    if temp_action_history:
        temp_action_history[-1] = action_onehot

    state_obs = pad_to_window(obs_history, STATE_WINDOW_SIZE)
    state_mask = pad_to_window(mask_history, STATE_WINDOW_SIZE)
    state_actions = pad_to_window(temp_action_history, STATE_WINDOW_SIZE)

    return world_model.predict_next_state(state_obs, state_mask, static, state_actions)


def format_predicted_state(next_obs, world_model):
    """Format the predicted next state."""
    lines = []

    vital_metrics = [
        ('heart_rate', 'Heart Rate', 'bpm'),
        ('sysbp', 'Sysbp', 'mmHg'),
        ('diabp', 'Diabp', 'mmHg'),
        ('meanbp', 'Meanbp', 'mmHg'),
        ('resp_rate', 'Resp Rate', 'breaths/min'),
        ('spo2', 'Spo2', '%'),
        ('temp_c', 'Temp C', '°C'),
    ]

    lines.append("**Vital Signs:**")
    for col, display_name, unit in vital_metrics:
        idx = DYNAMIC_COLS.index(col)
        val = world_model.denormalize_dynamic(next_obs[idx], col)
        lines.append(f"- {display_name}: {val:.1f} {unit}")

    lab_metrics = [
        ('lactate', 'Lactate', 'mmol/L'),
        ('creatinine', 'Creatinine', 'mg/dL'),
        ('bilirubin_total', 'Bilirubin Total', 'mg/dL'),
        ('platelet', 'Platelet', 'K/μL'),
        ('ph', 'Ph', ''),
        ('pao2', 'Pao2', 'mmHg'),
    ]

    lines.append("")
    lines.append("**Key Lab Values:**")
    for col, display_name, unit in lab_metrics:
        idx = DYNAMIC_COLS.index(col)
        val = world_model.denormalize_dynamic(next_obs[idx], col)
        lines.append(f"- {display_name}: {val:.1f} {unit}")

    urine_idx = DYNAMIC_COLS.index('output_4hourly')
    urine_val = world_model.denormalize_dynamic(next_obs[urine_idx], 'output_4hourly')
    urine_val = max(0.0, urine_val)
    lines.append("")
    lines.append(f"**Urine Output:** {urine_val:.1f} mL/4h")

    return "\n".join(lines)


def format_simulation_response(actions, predicted_states, world_model):
    """Format the simulation tool_response."""
    lines = ["## Simulation Results"]
    lines.append("")
    lines.append("")
    lines.append("*Note**: The simulation results come from an approximate predictive model and may contain specific biases. You must apply the following **Clinical Skepticism**:\\n1. **Inaction Skepticism ([0,0] Bias)**: Sepsis is a progressive, life-threatening condition. If the patient has hypoperfusion or sepsis, be **extremely skeptical** if the simulation predicts stability or improvement with **NO ACTION ([0,0])**. Spontaneous recovery without intervention is medically improbable in this context and likely indicates a \\\"Status Quo Bias\\\" in the model.\\n2. **High-Dose Skepticism**: Be cautious if the simulation predicts unrealistically perfect outcomes for **excessive or maximum** dosages of Vasopressors or Fluids. Real-world aggressive treatment carries risks (e.g., fluid overload, arrhythmia) and diminishing returns. If the model shows linear improvement with extreme doses, suspect a \\\"Linearity Bias\\\".\\n3. **Override Rule**: If the simulation results violate these clinical realities or the **Clinical Protocols**, explicitly identify the discrepancy in your response, **DISCARD** the misleading simulation data, and stick to the Guidelines.")
    lines.append("")
    lines.append("")

    for i, (act, next_obs) in enumerate(zip(actions, predicted_states)):
        vaso = act["vasopressor"]
        fluid = act["iv_fluid"]
        vaso_text = VASO_LEVELS[vaso]
        fluid_text = FLUID_LEVELS[fluid]

        lines.append(f"### Option {i+1}: IV Fluid={fluid_text}, Vasopressor={vaso_text}")
        lines.append("Predicted patient state after 4 hours:")
        lines.append("")
        lines.append(format_predicted_state(next_obs, world_model))
        lines.append("")

    lines.append("---")
    lines.append("You may call `simulation` again, or call `prescription` to make your final decision.")

    return "\n".join(lines)


def calculate_reward(current_obs, next_obs, current_sofa, next_sofa, is_terminated, outcome, world_model):
    """Compute the reward."""
    if is_terminated:
        total_reward = REWARD_RELEASE if outcome == 'release' else REWARD_DEATH
        return {'sofa_penalty': None, 'sofa_change': None, 'lactate_change': None, 'total_reward': total_reward}

    lactate_idx = DYNAMIC_COLS.index('lactate')
    lactate_curr = world_model.denormalize_dynamic(current_obs[lactate_idx], 'lactate')
    lactate_next = world_model.denormalize_dynamic(next_obs[lactate_idx], 'lactate')

    sofa_penalty = SOFA_PENALTY_SAME if (next_sofa == current_sofa and next_sofa > 0) else 0.0
    sofa_change_reward = SOFA_CHANGE_COEF * (next_sofa - current_sofa)
    lactate_change_reward = LACTATE_CHANGE_COEF * np.tanh(lactate_next - lactate_curr)

    total_reward = sofa_penalty + sofa_change_reward + lactate_change_reward

    return {
        'sofa_penalty': float(sofa_penalty),
        'sofa_change': float(sofa_change_reward),
        'lactate_change': float(lactate_change_reward),
        'total_reward': float(total_reward)
    }


def format_prescription_response(vaso, fluid, is_terminated, outcome,
                                   state_history, action_history, world_model, current_step):
    """Format the prescription tool_response."""
    vaso_text = VASO_LEVELS[vaso]
    fluid_text = FLUID_LEVELS[fluid]
    hours_since_admission = current_step * 4
    next_hour = hours_since_admission + 4

    lines = []

    lines.append(f"Based on your decision, the patient received {vaso_text} vasopressor and {fluid_text} IV fluid over the past 4 hours.")
    lines.append("")

    if is_terminated:
        if outcome == "death":
            lines.append(f"## Patient Status Update (Hour {next_hour})")
            lines.append("Unfortunately, despite all medical efforts, the patient has passed away.")
        else:
            lines.append(f"## Patient Status Update (Hour {next_hour})")
            lines.append("Good news! The patient's condition has stabilized sufficiently for ICU discharge.")
    else:
        lines.append(format_state_history_for_tool_response(
            state_history, action_history, world_model, current_step + 1
        ))

    return "\n".join(lines)


# ============================================================
# API call (vLLM OpenAI tools API)
# ============================================================

def save_formatted_conversation(stay_id, messages, debug_log_dir, tokenizer):
    """Save the full formatted conversation to a file using tokenizer.apply_chat_template."""
    os.makedirs(debug_log_dir, exist_ok=True)

    # Save as .txt for easy reading
    formatted_file = os.path.join(debug_log_dir, f"stay_{stay_id}_conversation.txt")

    try:
        # Use the tokenizer's apply_chat_template method
        # tokenize=False returns a string instead of token ids
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception as e:
        # Fall back to a simple format if apply_chat_template fails
        log_verbose(f"  [stay_id={stay_id}] Warning: apply_chat_template failed: {e}")
        formatted_text = json.dumps(messages, indent=2, ensure_ascii=False)

    with open(formatted_file, 'w', encoding='utf-8') as f:
        f.write(f"# Stay ID: {stay_id}\n")
        f.write(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total messages: {len(messages)}\n")
        f.write("=" * 80 + "\n\n")
        f.write(formatted_text)

    log_verbose(f"  [stay_id={stay_id}] Formatted conversation saved to: {formatted_file}")


def save_debug_log(stay_id, step, round_num, messages, response_content, tool_name, tool_args, debug_log_dir):
    """Save a debug log entry."""
    os.makedirs(debug_log_dir, exist_ok=True)
    log_file = os.path.join(debug_log_dir, f"stay_{stay_id}.jsonl")

    log_entry = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "stay_id": stay_id,
        "step": step,
        "round": round_num,
        "messages_count": len(messages),
        "last_message_role": messages[-1]["role"] if messages else None,
        "response_content": response_content,
        "tool_name": tool_name,
        "tool_args": tool_args
    }

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


def call_llm_api_sync(messages, tools, stay_id, step, gpu_id, base_port, model_name, debug_log_dir, round_num=0):
    """Synchronous call to the local vLLM API (using the OpenAI tools API)."""
    from openai import OpenAI

    port = base_port + gpu_id
    base_url = f"http://{VLLM_HOST}:{port}/v1"

    client = OpenAI(
        base_url=base_url,
        api_key="not-needed"
    )

    for attempt in range(API_MAX_RETRIES):
        try:
            attempt_msg = f" (attempt {attempt + 1}/{API_MAX_RETRIES})" if attempt > 0 else ""
            log_verbose(f"  [stay_id={stay_id} Step {step}] Calling vLLM API (GPU {gpu_id}, port {port}){attempt_msg}...")

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=8192,
                top_p=0.95,
                extra_body={
                    "top_k": -1
                }
            )

            result = parse_tool_call_from_api_response(response)
            content_len = len(result["content"]) if result["content"] else 0
            func_name = result["function_name"] or "none"
            log_verbose(f"  [stay_id={stay_id} Step {step}] API returned: content={content_len} chars, tool={func_name}")

            save_debug_log(stay_id, step, round_num, messages,
                          result["content"], result["function_name"], result["arguments"], debug_log_dir)

            return result

        except Exception as e:
            log_verbose(f"  [stay_id={stay_id} Step {step}] Error: {e}")
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue

            return {
                "function_name": "prescription",
                "arguments": {"vasopressor": 0, "iv_fluid": 0},
                "content": "",
                "tool_call_id": None
            }

    return {
        "function_name": "prescription",
        "arguments": {"vasopressor": 0, "iv_fluid": 0},
        "content": "",
        "tool_call_id": None
    }


# ============================================================
# Episode Runner
# ============================================================

@dataclass
class EpisodeResult:
    """Result of running one episode."""
    episode_id: int
    stay_id: int
    steps: int
    outcome: str
    model_total_reward: float = 0.0
    messages: List[Dict] = field(default_factory=list)
    model_actions: List[Tuple[int, int]] = field(default_factory=list)
    real_actions: List[Tuple[int, int]] = field(default_factory=list)
    error_type: str = None
    # Medical guideline compliance counters
    total_prescriptions: int = 0           # Total number of prescriptions
    hypoperfusion_violations: int = 0      # Number of guideline-1 violations
    vaso_weaning_violations: int = 0       # Number of guideline-2 violations

    def format_actions_str(self, actions):
        return ",".join([f"[{v},{f}]" for v, f in actions])


def run_episode(episode_id, episode_data, world_model, static_scaler_mean, static_scaler_std,
                gpu_id, base_port, model_name, debug_log_dir, tokenizer):
    """Run a single episode (synchronous version)."""

    stay_id = episode_data['stay_id']
    obs_all = episode_data['obs']
    mask_all = episode_data['mask']
    static = episode_data['static']
    actions_all = episode_data['actions']
    sofa_all = episode_data['sofa']
    real_vaso_all = episode_data['vaso_actions']
    real_fluid_all = episode_data['fluid_actions']

    if 'mortality_90d' in episode_data:
        real_outcome = 'death' if episode_data['mortality_90d'] == 1 else 'release'
    else:
        real_outcome = episode_data['outcome']

    total_steps = len(obs_all)

    result = EpisodeResult(episode_id=episode_id, stay_id=stay_id, steps=0, outcome='unknown')

    current_step = START_STEP
    obs_history = []
    mask_history = []
    action_history_onehot = []
    sofa_history = []
    action_history_tuple = []
    urine_history_raw = []  # Raw urine output history (mL), used to compute past_24h_urine
    use_worldmodel = False

    obs_history.append(obs_all[0])
    mask_history.append(mask_all[0])
    sofa_history.append(sofa_all[0])
    action_history_onehot.append(np.zeros(ACT_DIM, dtype=np.float32))
    # Initialize raw urine output for the first step
    urine_val = world_model.denormalize_dynamic(obs_all[0][URINE_IDX], 'output_4hourly')
    urine_history_raw.append(max(0.0, urine_val))

    for t in range(total_steps):
        result.real_actions.append((int(real_vaso_all[t]), int(real_fluid_all[t])))

    system_prompt = build_system_prompt()
    initial_user_prompt = build_initial_user_prompt(
        static, obs_history, world_model, static_scaler_mean, static_scaler_std, current_step
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_prompt}
    ]

    saved_messages = [
        {"role": "user", "content": initial_user_prompt}
    ]

    while current_step < total_steps:
        result.steps = current_step + 1

        current_obs = obs_history[-1]
        current_mask = mask_history[-1]
        current_sofa = sofa_history[-1]

        real_vaso_cur = int(real_vaso_all[current_step])
        real_fluid_cur = int(real_fluid_all[current_step])
        real_next_obs = obs_all[current_step + 1] if current_step + 1 < total_steps else None

        simulation_round = 0
        api_call_round = 0

        while True:
            api_call_round += 1

            api_result = call_llm_api_sync(messages, TOOLS, stay_id, current_step, gpu_id,
                                           base_port, model_name, debug_log_dir, api_call_round)

            func_name = api_result["function_name"]
            func_args = api_result["arguments"]
            content = api_result["content"]
            tool_call_id = api_result["tool_call_id"]

            if content:
                saved_messages.append({"role": "assistant", "content": content})

            assistant_msg = {"role": "assistant", "content": content}
            if func_name:
                assistant_msg["tool_calls"] = [{
                    "id": tool_call_id or f"call_{current_step}_{simulation_round}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(func_args)
                    }
                }]
            messages.append(assistant_msg)

            if func_name == "simulation":
                simulation_round += 1
                actions_to_simulate = validate_simulation_actions(func_args.get("actions", []))

                log_verbose(f"  [stay_id={stay_id} Step {current_step}] Simulation round {simulation_round}, actions: {actions_to_simulate}")

                tool_call_content = json.dumps({
                    "name": "simulation",
                    "arguments": {"actions": [f"[{a['vasopressor']},{a['iv_fluid']}]" for a in actions_to_simulate]}
                })
                saved_messages.append({"role": "tool_call", "content": tool_call_content})

                predicted_states = []
                for act in actions_to_simulate:
                    vaso = act["vasopressor"]
                    fluid = act["iv_fluid"]
                    pred_next = simulate_single_action(
                        vaso, fluid, obs_history, mask_history, action_history_onehot,
                        static, world_model, use_real_data=(not use_worldmodel),
                        real_next_obs=real_next_obs, real_vaso=real_vaso_cur, real_fluid=real_fluid_cur
                    )
                    predicted_states.append(pred_next)

                sim_response_content = format_simulation_response(actions_to_simulate, predicted_states, world_model)

                saved_messages.append({
                    "role": "tool_response",
                    "content": json.dumps({"result": sim_response_content})
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id or f"call_{current_step}_{simulation_round}",
                    "content": sim_response_content
                })

                if simulation_round >= MAX_SIMULATION_ROUNDS:
                    log_verbose(f"  [stay_id={stay_id} Step {current_step}] Max simulation rounds reached")

                # Safety valve: skip the episode if too many consecutive simulations
                if simulation_round >= MAX_SIMULATION_BEFORE_SKIP:
                    result.error_type = "excessive_simulation"
                    raise RuntimeError(f"Episode stuck: {simulation_round} consecutive simulations without prescription at step {current_step}")

            elif func_name == "prescription":
                vaso, fluid = validate_prescription_args(func_args)
                action_onehot = action_to_onehot(vaso, fluid)
                action_history_tuple.append((vaso, fluid))
                result.model_actions.append((vaso, fluid))

                # Medical guideline compliance check
                result.total_prescriptions += 1
                prev_vaso = action_history_tuple[-2][0] if len(action_history_tuple) >= 2 else None
                compliance = check_guideline_compliance(current_obs, vaso, fluid, prev_vaso, world_model)
                if compliance['hypoperfusion_violated']:
                    result.hypoperfusion_violations += 1
                    log_verbose(f"  [stay_id={stay_id} Step {current_step}] Guideline 1 VIOLATED: "
                               f"Hypoperfusion (Lac={compliance['lactate']:.1f}, MAP={compliance['map']:.1f}) but action=(0,0)")
                if compliance['vaso_weaning_violated']:
                    result.vaso_weaning_violations += 1
                    log_verbose(f"  [stay_id={stay_id} Step {current_step}] Guideline 2 VIOLATED: "
                               f"MAP={compliance['map']:.1f}>=65 but vaso increased {prev_vaso}->{vaso}")

                log_verbose(f"  [stay_id={stay_id} Step {current_step}] Prescription: vaso={vaso}, fluid={fluid}")

                tool_call_content = json.dumps({
                    "name": "prescription",
                    "arguments": {"vasopressor": vaso, "iv_fluid": fluid}
                })
                saved_messages.append({"role": "tool_call", "content": tool_call_content})

                if not use_worldmodel:
                    if vaso != real_vaso_cur or fluid != real_fluid_cur:
                        use_worldmodel = True
                        log_verbose(f"  [stay_id={stay_id}] Switching to WorldModel at step {current_step}")

                is_last_step = (current_step == total_steps - 1)

                if is_last_step:
                    action_history_onehot[-1] = action_onehot

                    outcome_obs = pad_to_window(obs_history, OUTCOME_WINDOW_SIZE)
                    outcome_mask = pad_to_window(mask_history, OUTCOME_WINDOW_SIZE)
                    outcome_actions = pad_to_window(action_history_onehot, OUTCOME_WINDOW_SIZE)

                    if use_worldmodel:
                        outcome, death_prob = world_model.predict_outcome(outcome_obs, outcome_mask, static, outcome_actions)
                        log_verbose(f"  [stay_id={stay_id}] WorldModel Outcome: {outcome} (death_prob={death_prob:.3f})")
                    else:
                        outcome = real_outcome
                        log_verbose(f"  [stay_id={stay_id}] Real Outcome: {outcome}")

                    result.outcome = outcome

                    tool_content = format_prescription_response(
                        vaso, fluid, True, outcome,
                        obs_history, action_history_tuple, world_model, current_step
                    )

                    reward_info = calculate_reward(current_obs, None, current_sofa, None, True, outcome, world_model)
                    result.model_total_reward += reward_info['total_reward']

                    saved_messages.append({
                        "role": "tool_response",
                        "content": json.dumps({"result": tool_content})
                    })

                    log_verbose(f"  [stay_id={stay_id}] Terminated: {result.outcome}, Steps: {result.steps}, Reward: {result.model_total_reward:.2f}")

                else:
                    if use_worldmodel:
                        action_history_onehot[-1] = action_onehot

                        state_obs = pad_to_window(obs_history, STATE_WINDOW_SIZE)
                        state_mask = pad_to_window(mask_history, STATE_WINDOW_SIZE)
                        state_actions = pad_to_window(action_history_onehot, STATE_WINDOW_SIZE)

                        # Use the new method to also get vent_prob
                        next_obs, vent_prob = world_model.predict_next_state_with_vent(state_obs, state_mask, static, state_actions)
                        # Compute past_24h_urine (last 6 four-hour timesteps)
                        past_24h_urine = sum(urine_history_raw[-6:])
                        # Compute SOFA from the new dynamics
                        next_sofa = compute_sofa_from_obs(next_obs, vent_prob, vaso, past_24h_urine, world_model)
                        next_mask = current_mask

                        obs_history.append(next_obs)
                        mask_history.append(next_mask)
                        sofa_history.append(next_sofa)
                        action_history_onehot.append(action_onehot)
                        # Append the new urine output
                        urine_val = world_model.denormalize_dynamic(next_obs[URINE_IDX], 'output_4hourly')
                        urine_history_raw.append(max(0.0, urine_val))
                    else:
                        next_obs = obs_all[current_step + 1]
                        next_sofa = sofa_all[current_step + 1]
                        next_mask = mask_all[current_step + 1]

                        obs_history.append(next_obs)
                        mask_history.append(next_mask)
                        sofa_history.append(next_sofa)
                        action_history_onehot.append(action_onehot)
                        # Append the real urine output
                        urine_val = world_model.denormalize_dynamic(next_obs[URINE_IDX], 'output_4hourly')
                        urine_history_raw.append(max(0.0, urine_val))

                    tool_content = format_prescription_response(
                        vaso, fluid, False, None,
                        obs_history, action_history_tuple, world_model, current_step
                    )

                    reward_info = calculate_reward(current_obs, next_obs, current_sofa, next_sofa, False, None, world_model)
                    result.model_total_reward += reward_info['total_reward']

                    saved_messages.append({
                        "role": "tool_response",
                        "content": json.dumps({"result": tool_content})
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id or f"call_{current_step}_prescription",
                        "content": tool_content
                    })

                break

            else:
                log_verbose(f"  [stay_id={stay_id} Step {current_step}] Unknown or no function call (round {api_call_round})")

                # Safety valve: skip the episode if too many consecutive invalid calls
                if api_call_round >= MAX_SIMULATION_BEFORE_SKIP:
                    result.error_type = "no_valid_tool_call"
                    raise RuntimeError(f"Episode stuck: {api_call_round} API calls without valid tool call at step {current_step}")

        if current_step == total_steps - 1:
            break

        current_step += 1

    result.messages = saved_messages

    # Save the formatted full conversation (using tokenizer.apply_chat_template)
    save_formatted_conversation(stay_id, messages, debug_log_dir, tokenizer)

    return result


# ============================================================
# Worker process
# ============================================================

def worker_process(worker_id, gpu_id, task_queue, result_queue, base_port, model_name, debug_log_dir, model_path, quiet_mode=False):
    """Worker process: serially handle assigned tasks."""
    from transformers import AutoTokenizer

    # Set the global verbose flag inside the child process
    global VERBOSE_MODE
    VERBOSE_MODE = not quiet_mode

    device = torch.device(f'cuda:{gpu_id}')
    log_verbose(f"[Worker {worker_id}] Starting on GPU {gpu_id}, device={device}...")

    try:
        world_model = WorldModel(WORLDMODEL_DIR, device)
        log_verbose(f"[Worker {worker_id}] WorldModel loaded")

        # Load the tokenizer for conversation formatting
        log_verbose(f"[Worker {worker_id}] Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        log_verbose(f"[Worker {worker_id}] Tokenizer loaded")

        static_scaler_mean = world_model.feature_config.get("static_scaler_mean", [65.0, 5.0])
        static_scaler_std = world_model.feature_config.get("static_scaler_std", [15.0, 3.0])

        while True:
            task = task_queue.get()
            if task is None:
                log_verbose(f"[Worker {worker_id}] Received termination signal")
                break

            episode_id, episode_data = task
            stay_id = episode_data['stay_id']

            log_verbose(f"[Worker {worker_id}] Processing episode {episode_id} (stay_id={stay_id})")

            try:
                result = run_episode(episode_id, episode_data, world_model,
                                    static_scaler_mean, static_scaler_std, gpu_id, base_port,
                                    model_name, debug_log_dir, tokenizer)

                result_item = {
                    "metadata": {
                        "episode_id": result.episode_id,
                        "stay_id": result.stay_id,
                        "model": {
                            "total_reward": result.model_total_reward,
                            "trajectory_length": result.steps,
                            "outcome": result.outcome,
                            "actions": result.format_actions_str(result.model_actions)
                        },
                        "real": {
                            "actions": result.format_actions_str(result.real_actions)
                        },
                        "guideline_compliance": {
                            "total_prescriptions": result.total_prescriptions,
                            "hypoperfusion_violations": result.hypoperfusion_violations,
                            "vaso_weaning_violations": result.vaso_weaning_violations
                        }
                    },
                    "messages": result.messages
                }

                result_queue.put((stay_id, result_item, None, None))

            except Exception as e:
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                error_type = getattr(e, 'error_type', 'unknown') if hasattr(e, 'error_type') else 'unknown'
                # Check whether this is a known error type
                if "excessive_simulation" in str(e):
                    error_type = "excessive_simulation"
                elif "no_valid_tool_call" in str(e) or "without valid tool call" in str(e):
                    error_type = "no_valid_tool_call"
                log_verbose(f"[Worker {worker_id}] {error_msg}")
                result_queue.put((stay_id, None, error_msg, error_type))

    except Exception as e:
        tqdm.write(f"[Worker {worker_id}] Fatal error: {e}\n{traceback.format_exc()}")

    log_verbose(f"[Worker {worker_id}] Exited")


# ============================================================
# Main program helpers
# ============================================================

def load_real_rewards_by_stay_id(path):
    """Load per-clinician rewards and build a stay_id -> reward_info mapping."""
    if not os.path.exists(path):
        tqdm.write(f"Warning: Real rewards file not found: {path}")
        return {}

    with open(path, 'r') as f:
        raw_data = json.load(f)

    stay_id_map = {}
    for idx, info in raw_data.items():
        stay_id = info['stay_id']
        stay_id_map[stay_id] = {
            'total_reward': info['total_reward'],
            'trajectory_length': info['trajectory_length'],
            'outcome': info['outcome']
        }

    return stay_id_map


def load_completed_stay_ids(output_file):
    """Load the set of completed stay_ids."""
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data.get('results', []):
            stay_id = item['metadata']['stay_id']
            completed.add(stay_id)
    return completed


def load_failed_stay_ids(failed_file):
    """Load the set of failed stay_ids."""
    failed = {}
    if os.path.exists(failed_file):
        with open(failed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data.get('failed_episodes', []):
            stay_id = item['stay_id']
            failed[stay_id] = item
    return failed


def save_result_to_file(output_file, result_item, real_rewards_map, lock):
    """Save one result to file."""
    with lock:
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"results": [], "summary": {}}

        stay_id = result_item['metadata']['stay_id']
        if stay_id in real_rewards_map:
            real_info = real_rewards_map[stay_id]
            result_item['metadata']['real']['total_reward'] = real_info['total_reward']
            result_item['metadata']['real']['trajectory_length'] = real_info['trajectory_length']
            result_item['metadata']['real']['outcome'] = real_info['outcome']

        data['results'].append(result_item)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def save_failed_episode(failed_file, stay_id, error_msg, error_type, lock):
    """Persist information about a failed episode."""
    with lock:
        if os.path.exists(failed_file):
            with open(failed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"failed_episodes": [], "summary": {}}

        # Check whether already present
        existing_ids = {item['stay_id'] for item in data['failed_episodes']}
        if stay_id not in existing_ids:
            data['failed_episodes'].append({
                "stay_id": stay_id,
                "error_type": error_type,
                "error_message": error_msg[:500],  # Truncate error message
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


def remove_from_failed(failed_file, stay_id, lock):
    """Remove a successful episode from the failed list."""
    with lock:
        if os.path.exists(failed_file):
            with open(failed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data['failed_episodes'] = [
                item for item in data['failed_episodes']
                if item['stay_id'] != stay_id
            ]

            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


def compute_and_print_summary(output_file, failed_file, real_rewards_map):
    """Compute and print the summary statistics."""
    if not os.path.exists(output_file):
        tqdm.write("No results to summarize")
        return

    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    if not results:
        tqdm.write("No results to summarize")
        return

    # Model stats
    model_rewards = [r['metadata']['model']['total_reward'] for r in results]
    model_outcomes = [r['metadata']['model']['outcome'] for r in results]
    model_death_count = sum(1 for o in model_outcomes if o == 'death')

    # Real-doctor stats
    real_rewards = []
    real_outcomes = []
    for r in results:
        stay_id = r['metadata']['stay_id']
        if stay_id in real_rewards_map:
            real_rewards.append(real_rewards_map[stay_id]['total_reward'])
            real_outcomes.append(real_rewards_map[stay_id]['outcome'])

    real_death_count = sum(1 for o in real_outcomes if o == 'death')

    # Failed-episode stats
    failed_count = 0
    failed_by_type = {}
    if os.path.exists(failed_file):
        with open(failed_file, 'r', encoding='utf-8') as f:
            failed_data = json.load(f)
        failed_episodes = failed_data.get('failed_episodes', [])
        failed_count = len(failed_episodes)
        for item in failed_episodes:
            error_type = item.get('error_type', 'unknown')
            failed_by_type[error_type] = failed_by_type.get(error_type, 0) + 1

    # Medical guideline compliance stats
    total_prescriptions = 0
    total_hypoperfusion_violations = 0
    total_vaso_weaning_violations = 0

    for r in results:
        gc = r['metadata'].get('guideline_compliance', {})
        total_prescriptions += gc.get('total_prescriptions', 0)
        total_hypoperfusion_violations += gc.get('hypoperfusion_violations', 0)
        total_vaso_weaning_violations += gc.get('vaso_weaning_violations', 0)

    # Compute compliance rates
    if total_prescriptions > 0:
        hypoperfusion_violation_rate = total_hypoperfusion_violations / total_prescriptions * 100
        vaso_weaning_violation_rate = total_vaso_weaning_violations / total_prescriptions * 100
        # Full compliance = actions that violate no guideline
        total_violations = total_hypoperfusion_violations + total_vaso_weaning_violations
        full_compliance_rate = max(0, (total_prescriptions - total_violations) / total_prescriptions * 100)
    else:
        hypoperfusion_violation_rate = 0
        vaso_weaning_violation_rate = 0
        full_compliance_rate = 100

    # Print the summary table
    tqdm.write("\n" + "=" * 70)
    tqdm.write("EVALUATION SUMMARY")
    tqdm.write("=" * 70)
    tqdm.write(f"{'Metric':<25} {'Model':<20} {'Real Doctor':<20}")
    tqdm.write("-" * 70)
    tqdm.write(f"{'Episodes (Success)':<25} {len(results):<20} {len(real_rewards):<20}")
    tqdm.write(f"{'Episodes (Failed)':<25} {failed_count:<20} {'N/A':<20}")
    tqdm.write(f"{'Avg Reward':<25} {np.nanmean(model_rewards):<20.4f} {np.mean(real_rewards) if real_rewards else 'N/A':<20}")
    tqdm.write(f"{'Std Reward':<25} {np.nanstd(model_rewards):<20.4f} {np.std(real_rewards) if real_rewards else 'N/A':<20}")
    tqdm.write(f"{'Death Count':<25} {model_death_count:<20} {real_death_count:<20}")
    tqdm.write(f"{'Death Rate':<25} {model_death_count/len(results)*100:<19.2f}% {real_death_count/len(real_outcomes)*100 if real_outcomes else 0:<19.2f}%")
    tqdm.write("=" * 70)

    # Print the medical-guideline compliance table
    tqdm.write("")
    tqdm.write("=" * 70)
    tqdm.write("MEDICAL GUIDELINE COMPLIANCE")
    tqdm.write("=" * 70)
    tqdm.write(f"{'Metric':<45} {'Count':<12} {'Rate':<12}")
    tqdm.write("-" * 70)
    tqdm.write(f"{'Total Prescriptions':<45} {total_prescriptions:<12} {'':<12}")
    tqdm.write(f"{'Guideline 1 Violations (Hypoperfusion+No Tx)':<45} {total_hypoperfusion_violations:<12} {hypoperfusion_violation_rate:<11.2f}%")
    tqdm.write(f"{'Guideline 2 Violations (Vaso Increase @MAP>=65)':<45} {total_vaso_weaning_violations:<12} {vaso_weaning_violation_rate:<11.2f}%")
    tqdm.write(f"{'Full Compliance Rate':<45} {'':<12} {full_compliance_rate:<11.2f}%")
    tqdm.write("=" * 70)

    if failed_count > 0:
        tqdm.write("\nFailed Episodes by Error Type:")
        for error_type, count in failed_by_type.items():
            tqdm.write(f"  - {error_type}: {count}")
        tqdm.write(f"\nTo retry failed episodes, run with --retry_failed flag")

    # Update summary
    data['summary'] = {
        "model": {
            "num_episodes": len(results),
            "avg_reward": float(np.nanmean(model_rewards)),
            "std_reward": float(np.nanstd(model_rewards)),
            "death_count": model_death_count,
            "death_rate": model_death_count / len(results)
        },
        "real_doctor": {
            "num_episodes": len(real_rewards),
            "avg_reward": float(np.mean(real_rewards)) if real_rewards else None,
            "std_reward": float(np.std(real_rewards)) if real_rewards else None,
            "death_count": real_death_count,
            "death_rate": real_death_count / len(real_outcomes) if real_outcomes else None
        },
        "failed": {
            "count": failed_count,
            "by_type": failed_by_type
        },
        "guideline_compliance": {
            "total_prescriptions": total_prescriptions,
            "hypoperfusion_violations": total_hypoperfusion_violations,
            "hypoperfusion_violation_rate": hypoperfusion_violation_rate,
            "vaso_weaning_violations": total_vaso_weaning_violations,
            "vaso_weaning_violation_rate": vaso_weaning_violation_rate,
            "full_compliance_rate": full_compliance_rate
        },
        "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# Main entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Sepsis Treatment Agent Workflow - Automated Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for vLLM serving')
    parser.add_argument('--num_gpus', type=int, default=DEFAULT_NUM_GPUS, help=f'Number of GPUs (default: {DEFAULT_NUM_GPUS})')
    parser.add_argument('--base_port', type=int, default=DEFAULT_BASE_PORT, help=f'Base port for vLLM (default: {DEFAULT_BASE_PORT})')
    parser.add_argument('--test', action='store_true', help=f'Test mode: evaluate first {TEST_MODE_NUM_EPISODES} episodes')
    parser.add_argument('--retry_failed', action='store_true', help='Only retry previously failed episodes')
    parser.add_argument('--chat_template', type=str, default=None, help='Path to chat template (auto-detected if not specified)')
    parser.add_argument('--skip_vllm_start', action='store_true', help='Skip vLLM service startup (assume already running)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='GPU memory utilization for vLLM (default: 0.95)')
    parser.add_argument('--max_model_len', type=int, default=None, help='Maximum context length for vLLM (default: use model config)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode: suppress detailed logs, only show progress bar')
    parser.add_argument('--max_retry_rounds', type=int, default=5, help='Max retry rounds for failed episodes (default: 5)')
    args = parser.parse_args()

    # Set the global verbose flag
    global VERBOSE_MODE
    VERBOSE_MODE = not args.quiet

    tqdm.write("=" * 70)
    tqdm.write("Sepsis Treatment Agent Workflow - Automated Evaluation")
    tqdm.write("=" * 70)
    tqdm.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tqdm.write(f"Model path: {args.model_path}")
    tqdm.write(f"Model name: {args.model_name}")
    tqdm.write(f"Number of GPUs: {args.num_gpus}")
    tqdm.write(f"Base port: {args.base_port}")
    tqdm.write(f"Test mode: {args.test}")
    tqdm.write(f"Retry failed: {args.retry_failed}")
    tqdm.write(f"Quiet mode: {args.quiet}")

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    debug_log_dir = os.path.join(OUTPUT_DIR, "debug_logs", args.model_name)
    os.makedirs(debug_log_dir, exist_ok=True)

    # Launch vLLM services
    if not args.skip_vllm_start:
        if not start_all_vllm_services(args.model_path, args.model_name,
                                       args.num_gpus, args.base_port, args.chat_template,
                                       args.gpu_memory_utilization):
            tqdm.write("[ERROR] Failed to start vLLM services")
            return 1
    else:
        tqdm.write("[INFO] Skipping vLLM service startup (--skip_vllm_start)")

    # Load data
    tqdm.write(f"\nLoading episodes from {PREPROCESSED_DATA_PATH}...")
    with open(PREPROCESSED_DATA_PATH, 'rb') as f:
        all_episodes = pickle.load(f)
    tqdm.write(f"  Loaded {len(all_episodes)} episodes")

    # Load per-clinician rewards
    tqdm.write(f"Loading real rewards from {REAL_REWARDS_PATH}...")
    real_rewards_map = load_real_rewards_by_stay_id(REAL_REWARDS_PATH)
    tqdm.write(f"  Loaded rewards for {len(real_rewards_map)} stay_ids")

    # Select samples
    if args.test:
        num_samples = min(TEST_MODE_NUM_EPISODES, len(all_episodes))
        tqdm.write(f"\nTEST MODE: Using first {num_samples} episodes")
    else:
        num_samples = len(all_episodes)

    selected_episodes = all_episodes[:num_samples]

    # Build output file paths
    test_or_full = f"test_{num_samples}" if args.test else f"full_{num_samples}"
    output_file = os.path.join(OUTPUT_DIR, f"eval_{args.model_name}_{test_or_full}.json")
    failed_file = os.path.join(OUTPUT_DIR, f"failed_{args.model_name}_{test_or_full}.json")

    # File-write lock
    manager = Manager()
    file_lock = manager.Lock()

    start_time = datetime.now()

    # ============ Auto-retry loop ============
    for retry_round in range(1, args.max_retry_rounds + 1):
        # Reload completed/failed state at the start of each round
        completed_stay_ids = load_completed_stay_ids(output_file)
        failed_stay_ids = load_failed_stay_ids(failed_file)

        # Build the list of pending tasks
        tasks_to_run = []

        if retry_round == 1 and args.retry_failed:
            # Round-1 retry_failed mode: only run previously failed episodes
            tqdm.write("\n[MODE] Retrying failed episodes only")
            for i, episode in enumerate(selected_episodes):
                stay_id = episode['stay_id']
                if stay_id in failed_stay_ids and stay_id not in completed_stay_ids:
                    tasks_to_run.append((i, episode))
        else:
            # Normal mode / subsequent retry rounds: run everything not yet completed
            for i, episode in enumerate(selected_episodes):
                stay_id = episode['stay_id']
                if stay_id not in completed_stay_ids:
                    tasks_to_run.append((i, episode))

        if not tasks_to_run:
            tqdm.write(f"\n[Round {retry_round}] All episodes completed!")
            break

        tqdm.write(f"\n[Round {retry_round}/{args.max_retry_rounds}] {len(tasks_to_run)} tasks to run...")

        # Create multi-process queues
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        # Push tasks into the queue
        for task in tasks_to_run:
            task_queue.put(task)

        # Push termination sentinels
        for _ in range(args.num_gpus):
            task_queue.put(None)

        # Launch worker processes
        tqdm.write(f"  Starting {args.num_gpus} worker processes...")
        workers = []
        for gpu_id in range(args.num_gpus):
            p = Process(target=worker_process, args=(
                gpu_id, gpu_id, task_queue, result_queue,
                args.base_port, args.model_name, debug_log_dir, args.model_path, args.quiet
            ))
            p.start()
            workers.append(p)

        # Collect results
        success_count = 0
        error_count = 0

        tqdm.write(f"  Processing {len(tasks_to_run)} tasks...")

        # Use a tqdm progress bar
        pbar = tqdm(total=len(tasks_to_run), desc=f"Round {retry_round}", unit="episode",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Success:{postfix[0]} Failed:{postfix[1]}',
                    postfix=[0, 0])

        tasks_processed = 0
        while tasks_processed < len(tasks_to_run):
            try:
                stay_id, result_item, error, error_type = result_queue.get(timeout=600)

                if error:
                    error_count += 1
                    save_failed_episode(failed_file, stay_id, error, error_type, file_lock)
                    pbar.write(f"  FAILED stay_id={stay_id} ({error_type}): {error[:100]}...")
                else:
                    success_count += 1
                    save_result_to_file(output_file, result_item, real_rewards_map, file_lock)
                    # If previously failed, remove it from the failed list
                    if stay_id in failed_stay_ids:
                        remove_from_failed(failed_file, stay_id, file_lock)
                    reward = result_item['metadata']['model']['total_reward']
                    outcome = result_item['metadata']['model']['outcome']
                    pbar.write(f"  DONE stay_id={stay_id}: reward={reward:.2f}, outcome={outcome}")

                tasks_processed += 1

                # Update the progress bar
                pbar.postfix[0] = success_count
                pbar.postfix[1] = error_count
                pbar.update(1)

            except Exception as e:
                alive_workers = sum(1 for p in workers if p.is_alive())
                pbar.write(f"  Timeout or error waiting for result, {alive_workers} workers alive")
                if alive_workers == 0:
                    break

        pbar.close()

        # Wait for workers to finish
        tqdm.write("  Waiting for workers to finish...")
        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                tqdm.write(f"  Worker {p.pid} still alive, terminating...")
                p.terminate()

        # Check whether any failures remain
        if error_count == 0:
            tqdm.write(f"\n[Round {retry_round}] All tasks succeeded!")
            break
        else:
            tqdm.write(f"\n[Round {retry_round}] {success_count} succeeded, {error_count} failed.")
            if retry_round < args.max_retry_rounds:
                tqdm.write(f"  Will retry failed episodes in next round...")

    # ============ End of loop ============

    # Tear down vLLM services
    cleanup_vllm_processes()

    # Print final statistics
    compute_and_print_summary(output_file, failed_file, real_rewards_map)

    end_time = datetime.now()
    tqdm.write(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    tqdm.write(f"Total time: {end_time - start_time}")
    tqdm.write(f"\nResults saved to: {output_file}")
    tqdm.write(f"Failed episodes saved to: {failed_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
