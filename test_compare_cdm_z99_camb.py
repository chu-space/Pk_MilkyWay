#!/usr/bin/env python3
import os
import subprocess

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

CLASS_EXE = "./class"
OUTPUT_DIR = "/resnick/groups/carnegie_poc/achu/cdm_calibration_final_match"
OUTPUT_ROOT_NAME = "gold_final_flat_"
INI_FILENAME = "calibration_run.ini"

# ==============================================================================
# 2. THE TEMPLATE (VG_correct_params_idm_synchronous.ini)
# ==============================================================================
TEMPLATE_CONTENT = """
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
# * CLASS input parameter file  *
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
m_dmeff = 1e-4
sigma_dmeff = 4.2e-28
npow_dmeff = 2
root = output/correct_idm_n2_1e-4GeV_halfmode_synchronous
output = dTk
modes = s
ic = ad
gauge = synchronous
h = 0.7
YHe = BBN
recombination = HyRec
reio_parametrization = reio_camb
z_reio = 7.6711
reionization_exponent = 1.5
reionization_width = 0.5
helium_fullreio_redshift = 3.5
helium_fullreio_width = 0.5
varying_fundamental_constants = none
bbn_alpha_sensitivity = 1.
T_cmb = 2.7255
Omega_b = 0.049
N_ur = 3.044
omega_cdm = 1e-15 
Omega_dmeff = 0.237
Omega_idm_dr = 0.
Omega_fld = 0
Omega_scf = 0
use_ppf = yes
c_gamma_over_c_fld = 0.4
fluid_equation_of_state = CLP
scf_parameters = 10.0, 0.0, 0.0, 0.0, 100.0, 0.0
attractor_ic_scf = yes
scf_tuning_index = 0
dmeff_target = hydrogen
Vrel_dmeff = 0.0
DM_annihilation_efficiency = 0.
DM_decay_fraction = 0.
Pk_ini_type = analytic_Pk
k_pivot = 0.05
sigma8 = 0.82
n_s = 0.96
alpha_s = 0.
r = 1.
l_max_scalars = 2500
l_max_tensors = 500
P_k_max_1/Mpc = 200.
k_per_decade_for_pk = 50
tol_ncdm_bg = 1.e-10
tol_ncdm_synchronous = 1.e-10
tol_ncdm_newtonian = 1.e-10
tol_perturbations_integration = 1.e-8
tol_background_integration = 1.e-8
lensing = no
sd_branching_approx = exact
sd_PCA_size = 2
sd_detector_name = PIXIE
sd_only_exotic = no
sd_include_g_distortion = no
sd_add_y = 0.
sd_add_mu = 0.
include_SZ_effect = no
overwrite_root = no
headers = yes
format = camb
write_background = no
write_thermodynamics = no
write_primordial = no
write_exotic_injection = no
write_distortions = no
write_parameters = yes
write_warnings = no
"""

# ==============================================================================
# 3. CALIBRATION OVERRIDES
# ==============================================================================
# FIX v4: 
# 1. Removed 'omega_dmeff' entirely. We only use 'Omega_dmeff' (Big O).
# 2. Kept 'Omega_b' and 'omega_cdm' to match the specific keys in the template.

CALIBRATION_PARAMS = {
    # --- I/O ---
    'root': os.path.join(OUTPUT_DIR, OUTPUT_ROOT_NAME),
    'z_pk': 99,
    'output': 'mPk, mTk',
    
    # --- PRECISION ---
    'k_per_decade_for_pk': 200,
    'P_k_max_1/Mpc': 1000,
    'tol_perturbations_integration': 1.e-5,

    # --- COSMOLOGY ---
    'h': 0.7,
    # Use 'Omega_b' to match template (Big O)
    'Omega_b': 0.047,         
    
    # Use 'omega_cdm' to match template (little o)
    # Value calculated from: (Omega_m=0.286 - Omega_b=0.047) * h^2(0.49) = 0.11711
    'omega_cdm': 0.11711,    
    
    'n_s': 0.96,
    'sigma8': 0.820,
    
    # --- PHYSICS: DISABLE IDM/dmeff ---
    # We ONLY define Omega_dmeff. 
    # CLASS will error if we define both Omega_dmeff and omega_dmeff.
    'Omega_dmeff': 0.0,      
    
    # Clean up other IDM params
    'm_dmeff': 0.0,          
    'sigma_dmeff': 0.0,       
    'npow_dmeff': 0,
    'Omega_fld': 0,
    'Omega_scf': 0,
}

# ==============================================================================
# 4. EXECUTION LOGIC
# ==============================================================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created directory: {OUTPUT_DIR}")

    ini_path = os.path.join(OUTPUT_DIR, INI_FILENAME)

    print(f"[INFO] Generating INI file from template...")
    
    lines = TEMPLATE_CONTENT.strip().split('\n')
    keys_to_override = set(CALIBRATION_PARAMS.keys())
    
    final_lines = []
    
    # Pass 1: Filter out overridden lines
    for line in lines:
        if not line.strip() or line.strip().startswith('#'):
            final_lines.append(line)
            continue
            
        if '=' in line:
            key = line.split('=')[0].strip()
            if key in keys_to_override:
                final_lines.append(f"# [OVERRIDDEN] {line}")
                continue
            final_lines.append(line)
        else:
            final_lines.append(line)

    # Pass 2: Append Correct Values
    final_lines.append("\n# === CDM CALIBRATION OVERRIDES (Gold v4) ===")
    for key, val in CALIBRATION_PARAMS.items():
        final_lines.append(f"{key} = {val}")
    
    with open(ini_path, 'w') as f:
        f.write('\n'.join(final_lines))
    
    print(f"[INFO] INI saved to: {ini_path}")

    print(f"[INFO] Running CLASS...")
    try:
        # Check output for errors
        res = subprocess.run([CLASS_EXE, ini_path], capture_output=True, text=True)
        
        if res.returncode == 0:
            print("[SUCCESS] CLASS simulation complete.")
            print(f"          Output expected at: {os.path.join(OUTPUT_DIR, OUTPUT_ROOT_NAME)}*_tk.dat")
        else:
            print("[ERROR] CLASS failed.")
            print("--- STDERR ---")
            print(res.stderr)
            print("--- STDOUT (tail) ---")
            print(res.stdout[-1000:])
            
    except FileNotFoundError:
        print(f"[CRITICAL] Could not find CLASS executable at: {CLASS_EXE}")

if __name__ == "__main__":
    main()
