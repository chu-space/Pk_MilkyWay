#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import sys
import concurrent.futures
import glob

# Change based on comsological parameters
H = 0.7
OMEGA_B = 0.047
OMEGA_DMEFF = 0.239
OMEGA_M = OMEGA_B + OMEGA_DMEFF

# Change based on IDM model
NPOW_DMEFF = 4
N_STR = 'n4'
IDM_MASS = 0.001
SIG_VALUES = [7.22414e-22, 2.37415e-24]
MASS_STR_PREFIX = "0.001"
COLORS = ['#007BFF', '#990000']

# WDM Target Parameters (Vogel & Abazajian Parameterisation)
M_WDM_KEV = 5.9
WDM_A = 0.0437
WDM_B = -1.188
WDM_NU = 1.049
WDM_THETA = 2.012
WDM_ETA = 0.2463

CLASS_EXECUTABLE = "./class"
CDM_TEMPLATE_INI = "VG_short_cdm_Pk.ini"
IDM_TEMPLATE_INI = "VG_short_idm_Pk.ini"

def check_files():
    if not os.path.exists(CLASS_EXECUTABLE):
        sys.exit(f"FATAL ERROR: CLASS executable not found at '{CLASS_EXECUTABLE}'.")
    for template in [CDM_TEMPLATE_INI, IDM_TEMPLATE_INI]:
        if not os.path.exists(template):
            sys.exit(f"FATAL ERROR: Template '{template}' not found.")

def run_class_and_get_pk(ini_path: str, root_path: str) -> tuple:
    """Checks for existing P(k) output; runs CLASS only if necessary."""
    search_pattern = f"{root_path}*pk.dat"
    found_files = glob.glob(search_pattern)

    if found_files:
        print(f"  -> Skipping CLASS run: Found existing data for {root_path}")
        target_file = found_files[0]
        data = np.loadtxt(target_file)
        return data[:, 0], data[:, 1]

    try:
        print(f"  -> Running CLASS for {root_path}...")
        result = subprocess.run([CLASS_EXECUTABLE, ini_path], check=True, capture_output=True, text=True)
        found_files = glob.glob(search_pattern)

        if found_files:
            target_file = found_files[0]
            data = np.loadtxt(target_file)
            return data[:, 0], data[:, 1]

    except subprocess.CalledProcessError as e:
        sys.exit(f"FATAL ERROR: CLASS failed on {ini_path}:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")

def find_halfmode_k(k_vals: np.ndarray, T2_vals: np.ndarray, threshold: float = 0.25) -> float:
    """Finds the scale k where T^2(k) crosses the threshold using log-linear interpolation."""
    below_idx = np.where(T2_vals < threshold)[0]
    if len(below_idx) == 0:
        return np.nan
    idx2 = below_idx[0]
    if idx2 == 0:
        return k_vals[0]

    idx1 = idx2 - 1
    k1, k2 = k_vals[idx1], k_vals[idx2]
    T1_sq, T2_sq = T2_vals[idx1], T2_vals[idx2]

    if T1_sq == T2_sq:
        return k1

    log_k1, log_k2 = np.log(k1), np.log(k2)
    log_k_half = log_k1 + (log_k2 - log_k1) * (threshold - T1_sq) / (T2_sq - T1_sq)
    return np.exp(log_k_half)

def calc_wdm_transfer(k_mpc: np.ndarray, h: float, omega_m: float) -> np.ndarray:
    """Calculates the WDM T^2(k) using the Vogel & Abazajian fitting formula."""
    omega_mh2 = omega_m * h**2
    k_h = k_mpc / h  # V&A formula expects k in h/Mpc
    alpha = WDM_A * (M_WDM_KEV**WDM_B) * ((omega_mh2 / 0.12)**WDM_ETA) * ((h / 0.6736)**WDM_THETA)
    T_wdm = (1.0 + (alpha * k_h)**(2 * WDM_NU))**(-5.0 / WDM_NU)
    return T_wdm**2

def run_single_simulation(task_params: tuple) -> tuple:
    sig, idm_mass, npow, output_dir_worker = task_params
    # CHANGED: Using :g to preserve exact significant figures in cache names
    sig_str = f"{sig:g}".replace("-", "m").replace("+", "p")
    idm_root_path = os.path.join(output_dir_worker, f"idm_s{sig_str}_m{idm_mass}_")
    ini_path = os.path.join(output_dir_worker, f"idm_sig_{sig_str}.ini")

    with open(IDM_TEMPLATE_INI) as f_in, open(ini_path, "w") as f_out:
        for line in f_in:
            if line.strip().startswith("sigma_dmeff"):
                f_out.write(f"sigma_dmeff = {sig}\n")
            elif line.strip().startswith("m_dmeff"):
                f_out.write(f"m_dmeff = {idm_mass}\n")
            elif line.strip().startswith("npow_dmeff"):
                f_out.write(f"npow_dmeff = {npow}\n")
            elif line.strip().startswith("root"):
                f_out.write(f"root = {idm_root_path}\n")
            else:
                f_out.write(line)

    k_raw_h, pk_raw = run_class_and_get_pk(ini_path, idm_root_path)
    return sig, k_raw_h * H, pk_raw

def run_cdm_baseline(output_dir: str) -> tuple:
    cdm_root_path = os.path.join(output_dir, "cdm_run_")
    cdm_run_ini = os.path.join(output_dir, "cdm_run.ini")
    with open(CDM_TEMPLATE_INI) as f_in, open(cdm_run_ini, "w") as f_out:
        for line in f_in:
            if line.strip().startswith("root"):
                f_out.write(f"root = {cdm_root_path}\n")
            else:
                f_out.write(line)
    k_raw_h, pk_raw = run_class_and_get_pk(cdm_run_ini, cdm_root_path)
    return k_raw_h * H, pk_raw

def main():
    check_files()
    output_dir = f"new_run_{N_STR}_{MASS_STR_PREFIX}"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Running Baseline CDM ---")
    k_cdm_mpc, pk_cdm = run_cdm_baseline(output_dir)

    # CHANGED: Expanded k-grid down to 1e-4 so the 1.664e-22 envelope doesn't fall off the edge
    k_common = np.logspace(-4, 2.3, 2000)
    pk_cdm_interp = np.interp(k_common, k_cdm_mpc, pk_cdm)

    print("\n--- Calculating 5.9 keV WDM Target ---")
    T2_wdm = calc_wdm_transfer(k_common, H, OMEGA_M)
    k_half_wdm = find_halfmode_k(k_common, T2_wdm)
    print(f"🎯 Target WDM 5.9 keV Half-mode k: {k_half_wdm:.3f} 1/Mpc")

    print(f"\n--- Running IDM Parallel Tasks (n={NPOW_DMEFF}) ---")
    tasks = [(sig, IDM_MASS, NPOW_DMEFF, output_dir) for sig in SIG_VALUES]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_simulation, tasks))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(k_common, T2_wdm, color='black', lw=2.5, ls='-', label=f'WDM 5.9 keV Target')
    if not np.isnan(k_half_wdm):
        ax.axvline(k_half_wdm, color='black', ls=':', alpha=0.8, label=f'WDM $k_{{1/2}} = {k_half_wdm:.2f}$')

    results.sort(key=lambda x: x[0])
    for idx, (sig, k_idm, pk_idm) in enumerate(results):
        pk_idm_interp = np.interp(k_common, k_idm, pk_idm)
        t2_idm = pk_idm_interp / pk_cdm_interp

        k_half_idm = find_halfmode_k(k_common, t2_idm)

        # CHANGED: Using :g in the console print to show all sig figs
        print(f"IDM \u03c3={sig:g}: Half-mode k = {k_half_idm:.3f} 1/Mpc")

        # CHANGED: Using :g in the plot label
        ax.plot(k_common, t2_idm, color=COLORS[idx], lw=2.5,
                label=fr'IDM $\sigma = {sig:g}$ ($k_{{1/2}}={k_half_idm:.1f}$)')

    ax.set_xscale('log')
    ax.set_xlim([1e-3, 2e2])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel(r'$k \ [1/\mathrm{Mpc}]$', fontsize=14)
    ax.set_ylabel(r'$T^2(k) = P_{\mathrm{IDM}} / P_{\mathrm{CDM}}$', fontsize=14)
    ax.axhline(0.25, color='gray', ls='--', alpha=0.6)
    ax.legend(loc='lower left')
    ax.grid(True, which="both", ls=':', alpha=0.5)

    plt.title(f"IDM vs 5.9 keV WDM Matching ($z=0, n={NPOW_DMEFF}$)")
    out_name = f"wdm_match_n{NPOW_DMEFF}_m{MASS_STR_PREFIX}.pdf"
    plt.savefig(out_name)
    print(f"\nPlot saved as {out_name}")

if __name__ == '__main__':
    main()
