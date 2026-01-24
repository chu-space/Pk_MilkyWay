#!/usr/bin/env python3
"""
CLASS IDM - WMAP9 FINAL PRODUCTION RUN
Applies the optimized 'Golden' parameters:
- WMAP9 Best Fit cosmology (h=0.6932, Ob=0.0463)
- Fixed Helium (YHe=0.24) and RECFAST for perfect reference matching
- Safe high-precision settings (l_max=3000, tol=1e-7)
"""

import numpy as np
import os
import subprocess
import glob

CLASS_EXECUTABLE = "./class"
OUTPUT_DIR = "output_improved_wmap9_micro"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Your Templates (Conflicting settings here will be overridden)
TEMPLATES = {
    'newtonian': 'VG_correct_params_idm_newtonian.ini',
    'synchronous': 'VG_correct_params_idm_synchronous.ini'
}

# ==============================================================================
# VARIATIONS: OPTIMIZED PARAMETERS
# ==============================================================================
VARIATIONS = {
    'Run_Fixed_Helium': {
        'desc': 'WMAP9 BestFit + Fixed YHe=0.24 (Golden Set)',
        'params': {
            # --- PHYSICS (WMAP-9 Exact Match) ---
            'h': 0.6932,
            'Omega_b': 0.0463,
            'Omega_dmeff': 0.2402,
            'n_s': 0.9608,
            'N_ur': 3.046,
            'T_cmb': 2.725,
            'YHe': 0.24,           # Fixes high-k tilt
            'recombination': 'RECFAST', # Stable with manual YHe
            
            # --- OUTPUT SETTINGS ---
            'write_background': 'yes',
            'overwrite_root': 'yes',
            
            # --- SAFE PRECISION (Matches your successful debug run) ---
            'k_per_decade_for_pk': 100,
            'l_max_scalars': 3000,
            'tol_perturbations_integration': 1.e-7,
            'tol_background_integration': 1.e-7,
        }
    }
}

# You can add all your models here (n=2, n=4, halfmode, envelope)
MODELS = [(4, "1e-4", "2.2e-27", "halfmode")]

# ==============================================================================
# HELPER: Robust INI Creator (Injecting Safety & Physics Overrides)
# ==============================================================================
def create_ini_production(template_path, params_to_override, output_path):
    with open(template_path, 'r') as f:
        lines = f.readlines()
    
    # These keys will be STRIPPED from the template to prevent duplicates
    keys_to_override = set(params_to_override.keys())
    
    new_lines = []
    for line in lines:
        line_strip = line.strip()
        if not line_strip or line_strip.startswith('#'):
            new_lines.append(line)
            continue
        
        # If the template line defines a key we are overriding, skip it
        if '=' in line:
            key = line.split('=')[0].strip()
            if key in keys_to_override:
                continue 
            new_lines.append(line)
        else:
            new_lines.append(line)
            
    new_lines.append("\n# --- FINAL OPTIMIZED PRODUCTION OVERRIDES ---\n")
    
    # Inject the Golden Parameter set
    for key, val in params_to_override.items():
        new_lines.append(f"{key} = {val}\n")
        
    with open(output_path, 'w') as f:
        f.writelines(new_lines)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("Starting Optimized WMAP9 Production Run...")

    for n, m_str, s_str, rtype in MODELS:
        print(f"\nModel: n={n}, m={m_str}GeV, type={rtype}")
        
        for run_name, var_data in VARIATIONS.items():
            run_id = f"idm_{run_name}_n{n}_m{m_str}_s{s_str}_{rtype}"
            
            run_params = var_data['params'].copy()
            run_params.update({
                'npow_dmeff': n,
                'm_dmeff': float(m_str),
                'sigma_dmeff': float(s_str)
            })
            
            for gauge in ['newtonian', 'synchronous']:
                gauge_id = f"{run_id}_{gauge}"
                ini_path = os.path.join(OUTPUT_DIR, f"{gauge_id}.ini")
                
                # root must include the trailing underscore for CLASS suffixing
                run_params['root'] = os.path.join(OUTPUT_DIR, gauge_id) + '_'
                
                create_ini_production(TEMPLATES[gauge], run_params, ini_path)
                
                print(f" > Running {gauge}...", end='', flush=True)
                
                # Capture output for diagnostics
                res = subprocess.run([CLASS_EXECUTABLE, ini_path], capture_output=True, text=True)
                
                if res.returncode == 0:
                    # Verify .dat files exist
                    dat_files = glob.glob(run_params['root'] + "*tk.dat")
                    if dat_files:
                        print(f" DONE")
                    else:
                        print(" FAILED (No .dat files)")
                else:
                    print(" FAILED (CLASS Error)")
                    # The error message helps identify if memory limits were hit
                    print(f"   [Error Snippet]: {res.stderr[-300:].strip()}")

    print("\nProduction Complete. All models now use optimized cosmology and Helium settings.")

if __name__ == "__main__":
    main()
