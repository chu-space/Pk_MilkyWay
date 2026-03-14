#!/usr/bin/env python3

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import subprocess
import concurrent.futures
import glob

H = 0.7
OMEGA_B = 0.047
OMEGA_DMEFF = 0.239
OMEGA_M = OMEGA_B + OMEGA_DMEFF

# --- UPDATED: WDM Target Parameters (COZMIC-1 / ETHOS Parameterisation) ---
# Transitioning from 5.9 keV to 6.5 keV target
M_WDM_KEV = 6.5 
WDM_A = 0.0437   # Thermal relic alpha normalization
WDM_B = -1.11    # COZMIC-1 specific mass scaling (v3.1.0 update)
WDM_NU = 1.12    # COZMIC-1 exponent factor
WDM_THETA = 2.0  # Hubble scaling
WDM_ETA = 0.25   # Matter density scaling

# Path Configurations 
OUTPUT_DIR = "6.5KEV_PK_AND_COZMIC_VISUALISATION"
CDM_TEMPLATE_INI = "VG_short_cdm_Pk.ini"
IDM_TEMPLATE_INI = "VG_short_idm_Pk.ini"
COZMIC_REF_DIR = "COZMIC_IDM_Pk"

# Explicitly set CLASS executable path
CLASS_EXECUTABLE = "../class_public-master-new-dmeff/class"

# Dataset: (n, mass_GeV, sigma_halfmode, sigma_6.5keV, sigma_envelope)
# Note: Cross-sections may need slight adjustment for 6.5keV versus 5.9keV
DATASETS = [
    (2, 1e-4, 4.2e-28, 1.0304e-27, 2.8e-27),
    (2, 1e-2, 1.3e-25, 1.3345e-24, 7.1e-24),
    (2, 1, 1.6e-23, 3.0532e-23, 8e-22),
    (4, 1e-4, 2.2e-27, 9.9132e-27, 3.4e-26),
    (4, 1e-2, 1.7e-22, 9.8346e-21, 1.7e-19),
    (4, 1, 8.6e-19, 2.0882e-17, 2.8e-16),
]

def check_files():
    if not os.path.exists(CLASS_EXECUTABLE):
        print(f"WARNING: CLASS executable not found at '{CLASS_EXECUTABLE}'.", flush=True)
    for template in [CDM_TEMPLATE_INI, IDM_TEMPLATE_INI]:
        if not os.path.exists(template):
            sys.exit(f"FATAL ERROR: Template '{template}' not found.")
    if not os.path.exists(COZMIC_REF_DIR):
        print(f"WARNING: COZMIC reference directory '{COZMIC_REF_DIR}' not found.", flush=True)

def run_class_and_get_pk(ini_path: str, root_path: str) -> tuple:
    search_pattern = f"{root_path}*pk.dat"
    found_files = glob.glob(search_pattern)

    if found_files:
        target_file = found_files[0]
        data = np.loadtxt(target_file)
        return data[:, 0], data[:, 1]

    try:
        custom_env = os.environ.copy()
        custom_env["OMP_NUM_THREADS"] = "1" 
        result = subprocess.run([CLASS_EXECUTABLE, ini_path], check=True, capture_output=True, text=True, env=custom_env)
        found_files = glob.glob(search_pattern)

        if found_files:
            target_file = found_files[0]
            data = np.loadtxt(target_file)
            return data[:, 0], data[:, 1]
        else:
            raise FileNotFoundError(f"CLASS P(k) output not found.")

    except subprocess.CalledProcessError as e:
        sys.exit(f"FATAL ERROR: CLASS failed:\n{e.stderr}")

def find_halfmode_k(k_vals: np.ndarray, T2_vals: np.ndarray, threshold: float = 0.25) -> float:
    below_idx = np.where(T2_vals < threshold)[0]
    if len(below_idx) == 0: return np.nan
    idx2 = below_idx[0]
    if idx2 == 0: return k_vals[0]
    idx1 = idx2 - 1
    k1, k2 = k_vals[idx1], k_vals[idx2]
    T1_sq, T2_sq = T2_vals[idx1], T2_vals[idx2]
    log_k1, log_k2 = np.log(k1), np.log(k2)
    log_k_half = log_k1 + (log_k2 - log_k1) * (threshold - T1_sq) / (T2_sq - T1_sq)
    return np.exp(log_k_half)

def calc_wdm_transfer(k_mpc: np.ndarray, h: float, omega_m: float) -> np.ndarray:
    """Calculates the WDM T^2(k) using COZMIC-1 fitting for 6.5 keV."""
    omega_mh2 = omega_m * h**2
    k_h = k_mpc / h  
    # Scaling relation for alpha (thermal relic transfer function scale)
    alpha = WDM_A * (M_WDM_KEV**WDM_B) * ((omega_mh2 / 0.12)**WDM_ETA) * ((h / 0.7)**WDM_THETA)
    T_wdm = (1.0 + (alpha * k_h)**(2 * WDM_NU))**(-5.0 / WDM_NU)
    return T_wdm**2

def run_single_model(task_params: tuple) -> tuple:
    sig, idm_mass, npow, output_dir_worker = task_params
    sig_str = f"{sig:g}".replace("-", "m").replace("+", "p")
    idm_root_path = os.path.join(output_dir_worker, f"idm_n{npow}_m{idm_mass}_s{sig_str}_")
    ini_path = os.path.join(output_dir_worker, f"idm_n{npow}_m{idm_mass}_s{sig_str}.ini")

    with open(IDM_TEMPLATE_INI) as f_in, open(ini_path, "w") as f_out:
        for line in f_in:
            if line.strip().startswith("sigma_dmeff"): f_out.write(f"sigma_dmeff = {sig}\n")
            elif line.strip().startswith("m_dmeff"): f_out.write(f"m_dmeff = {idm_mass}\n")
            elif line.strip().startswith("npow_dmeff"): f_out.write(f"npow_dmeff = {npow}\n")
            elif line.strip().startswith("root"): f_out.write(f"root = {idm_root_path}\n")
            else: f_out.write(line)

    k_raw_h, pk_raw = run_class_and_get_pk(ini_path, idm_root_path)
    return (npow, idm_mass, sig), k_raw_h * H, pk_raw

def main():
    check_files()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n--- Running Baseline CDM ---", flush=True)
    cdm_root_path = os.path.join(OUTPUT_DIR, "cdm_run_")
    cdm_run_ini = os.path.join(OUTPUT_DIR, "cdm_run.ini")
    with open(CDM_TEMPLATE_INI) as f_in, open(cdm_run_ini, "w") as f_out:
        for line in f_in:
            if line.strip().startswith("root"): f_out.write(f"root = {cdm_root_path}\n")
            else: f_out.write(line)
    k_cdm_mpc, pk_cdm = run_class_and_get_pk(cdm_run_ini, cdm_root_path)
    k_cdm_mpc *= H

    k_common = np.logspace(-4, 3.0, 3000)
    pk_cdm_interp = np.interp(k_common, k_cdm_mpc, pk_cdm)

    print(f"\n--- Calculating {M_WDM_KEV} keV WDM Target (COZMIC-1) ---", flush=True)
    T2_wdm = calc_wdm_transfer(k_common, H, OMEGA_M)
    k_half_wdm = find_halfmode_k(k_common, T2_wdm)
    print(f"Target WDM {M_WDM_KEV} keV $k_{{1/2}}$: {k_half_wdm:.3f} 1/Mpc", flush=True)

    tasks = []
    for n, mass, sig_hm, sig_65, sig_env in DATASETS:
        tasks.extend([(sig_hm, mass, n, OUTPUT_DIR), (sig_65, mass, n, OUTPUT_DIR), (sig_env, mass, n, OUTPUT_DIR)])
    tasks = list(set(tasks))

    results_dict = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(run_single_model, tasks):
            key, k_idm, pk_idm = result
            results_dict[key] = (k_idm, pk_idm)

    for n, mass, sig_hm, sig_65, sig_env in DATASETS:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(k_common, T2_wdm, color='black', lw=2.5, label=f'WDM {M_WDM_KEV} keV Target')
        if not np.isnan(k_half_wdm):
            ax.axvline(k_half_wdm, color='black', ls=':', alpha=0.8, label=f'WDM $k_{{1/2}} = {k_half_wdm:.2f}$')

        mass_str = "1GeV" if mass == 1 else f"{mass:g}"
        plot_configs = [
            ("Half-mode Match", sig_hm, '#007BFF', ':'), 
            (f"{M_WDM_KEV} keV Match", sig_65, '#2ca02c', '-'), 
            ("Envelope Match", sig_env, '#d62728', ':')
        ]

        for label_prefix, sig, color, linestyle in plot_configs:
            k_idm, pk_idm = results_dict[(n, mass, sig)]
            t2_idm = np.interp(k_common, k_idm, pk_idm) / pk_cdm_interp
            k_half_idm = find_halfmode_k(k_common, t2_idm)
            ax.plot(k_common, t2_idm, color=color, lw=2.5, ls=linestyle,
                    label=fr'{label_prefix} ($\sigma = {sig:g}$) [$k_{{1/2}}={k_half_idm:.1f}$]')

        ax.set_xscale('log')
        ax.set_xlim([1e-3, 5e2])
        ax.set_ylim([0, 1.1])
        ax.set_xlabel(r'$k \ [1/\mathrm{Mpc}]$', fontsize=14)
        ax.set_ylabel(r'$T^2(k) = P_{\mathrm{IDM}} / P_{\mathrm{CDM}}$', fontsize=14)
        ax.axhline(0.25, color='gray', ls='--', alpha=0.6)
        ax.legend(loc='lower left', fontsize='small')
        ax.grid(True, which="both", ls=':', alpha=0.5)
        plt.title(f"IDM vs {M_WDM_KEV} keV WDM Matching (n={n}, m={mass} GeV)")
        plt.savefig(os.path.join(OUTPUT_DIR, f"wdm_match_n{n}_{mass_str}GeV.pdf"))
        plt.close(fig)

if __name__ == '__main__':
    main()