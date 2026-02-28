#!/usr/bin/env python3
import os
import subprocess
import glob
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_EXECUTABLE = os.path.join(SCRIPT_DIR, "class")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_reference_5.9kev_cross_sections")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PROGRESS_LOG_FILE = os.path.join(LOG_DIR, "run_progress.log")

TEMPLATES = {
    'newtonian': os.path.join(SCRIPT_DIR, 'VG_correct_params_idm_newtonian.ini'),
    'synchronous': os.path.join(SCRIPT_DIR, 'VG_correct_params_idm_synchronous.ini')
}

# (n, mass_GeV, sigma_halfmode, sigma_5.9keV, sigma_envelope)
INPUT_MODELS = [
    (2, 1e-4, 4.2e-28, 2.9064e-27, 2.8e-27),
    (2, 1e-2, 1.3e-25, 2.6145e-24, 7.1e-24),
    (2, 1, 1.6e-23, 6.4057e-23, 8e-22),
    (4, 1e-4, 2.2e-27, 1.0496e-25, 3.4e-26),
    (4, 1e-2, 1.7e-22, 2.7134e-20, 1.7e-19),
    (4, 1, 8.6e-19, 5.7344e-17, 2.8e-16),
]

K_RES_2 = 80 # Identified best fitting k_per_decade resolution for n=2
K_RES_4 = 150 # Identified best fitting k_per_decade resolution for n=4

def create_ini_production(template_path, params_to_override, output_path, k_res):
    if not os.path.exists(template_path):
        print(f"\nCRITICAL ERROR: Template not found at {template_path}")
        return False

    with open(template_path, 'r') as f:
        lines = f.readlines()
    
    standardization_keys = {
        'root', 'k_per_decade_for_pk',
        'P_k_max_1/Mpc', 'write_background', 'overwrite_root'
    }
    
    new_lines = []
    for line in lines:
        if '=' in line and not line.strip().startswith('#'):
            key = line.split('=')[0].strip()
            if key in standardization_keys or key in params_to_override:
                continue
        new_lines.append(line)
    
    new_lines.append("\n# --- PRODUCTION PRECISION & COSMOLOGY OVERRIDES ---\n")
    new_lines.append(f"k_per_decade_for_pk = {k_res}\n") 
    new_lines.append("P_k_max_1/Mpc = 200.0\n") 
    new_lines.append("write_background = yes\n")
    new_lines.append("overwrite_root = yes\n")
    
    for key, val in params_to_override.items():
        new_lines.append(f"{key} = {val}\n")
        
    with open(output_path, 'w') as f:
        f.writelines(new_lines)
    return True

def load_completed_runs():
    completed = set()
    if os.path.exists(PROGRESS_LOG_FILE):
        with open(PROGRESS_LOG_FILE, 'r') as f:
            for line in f:
                if "SUCCESS" in line:
                    run_id = line.split('|')[0].strip()
                    completed.add(run_id)
    return completed

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    if not os.path.exists(CLASS_EXECUTABLE):
        print(f"CRITICAL ERROR: {CLASS_EXECUTABLE} not found!")
        return

    print(f"Starting Production. Executable: {CLASS_EXECUTABLE}")
    
    completed_runs = load_completed_runs()
    if completed_runs:
        print(f"Loaded {len(completed_runs)} previously completed runs from log.")

    # Open log in append mode so we can stream progress
    with open(PROGRESS_LOG_FILE, 'a') as prog_log:
        for n, mass, sig_hm, sig_59, sig_env in INPUT_MODELS:
            
            # --- Determine the resolution based on n ---
            current_k_res = K_RES_4 if n == 4 else K_RES_2
            
            cross_sections = {
                "halfmode": sig_hm,
                "5.9kev": sig_59,
                "envelope": sig_env
            }
            
            for rtype, sigma in cross_sections.items():
                base_params = {
                    'h': 0.7, 'Omega_b': 0.047, 'Omega_dmeff': 0.239,
                    'n_s': 0.96, 'sigma8': 0.820,
                    'npow_dmeff': n, 'm_dmeff': mass, 'sigma_dmeff': sigma
                }
                
                for g_type in ['newtonian', 'synchronous']:
                    run_id = f"idm_n{n}_m{mass}_{rtype}_{g_type}"
                    ini_path = os.path.join(OUTPUT_DIR, f"{run_id}.ini")
                    root_path = os.path.join(OUTPUT_DIR, run_id) + "_"
                    
                    # 1. Primary skip check: Is it in the log?
                    if run_id in completed_runs:
                        print(f"Skipping {run_id} (found in progress log)")
                        continue
                        
                    # 2. Fallback skip check: Are the files already in the directory?
                    if glob.glob(root_path + "*tk.dat"):
                        print(f"Skipping {run_id} (files found in output dir). Updating log...")
                        prog_log.write(f"{run_id} | SUCCESS | Time: Unknown (Recovered from dir)\n")
                        prog_log.flush()
                        completed_runs.add(run_id)
                        continue
                    
                    p = base_params.copy()
                    p['root'] = root_path
                    
                    print(f"\nProcessing {run_id} (k_res={current_k_res})...")
                    
                    # Pass the evaluated current_k_res to the ini creator
                    if create_ini_production(TEMPLATES[g_type], p, ini_path, current_k_res):
                        print(f"  > Executing CLASS...", end='', flush=True)
                        
                        start_time = time.time()
                        res = subprocess.run([CLASS_EXECUTABLE, ini_path], 
                                             capture_output=True, text=True, cwd=SCRIPT_DIR)
                        elapsed_time = time.time() - start_time
                        
                        if res.returncode == 0:
                            tks = glob.glob(root_path + "*tk.dat")
                            if tks:
                                status = "SUCCESS"
                                print(f" SUCCESS ({len(tks)} files) in {elapsed_time:.2f}s")
                                completed_runs.add(run_id)
                            else:
                                status = "FAILED"
                                print(f" FAILED (No files produced) in {elapsed_time:.2f}s")
                        else:
                            status = "ERROR"
                            print(f" ERROR (CLASS crashed) in {elapsed_time:.2f}s")
                            with open(os.path.join(LOG_DIR, f"{run_id}.err"), "w") as f:
                                f.write(res.stderr)

                        # Write to the running log immediately
                        prog_log.write(f"{run_id} | {status} | Time: {elapsed_time:.2f}s\n")
                        prog_log.flush() # Ensure it writes to disk immediately in case of external termination

    print(f"\nRuns finished. Data located in: {OUTPUT_DIR}")
    print(f"Log located in: {PROGRESS_LOG_FILE}")

if __name__ == "__main__":
    main()