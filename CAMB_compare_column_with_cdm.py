import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

# ==============================================================================
# CONFIGURATION
# ==============================================================================
FILE_STD      = "output/z99_synchronous__tk.dat"                       # Baseline
FILE_USER_IDM = "output/z99_idm_n2_m1e-2_s7.1e-24_envelope_synchronous__tk.dat" # Your Run
FILE_REF      = "data_tk/idm_n2_1e-2GeV_envelope_z99_Tk.dat"           # Reference

def load_transfer_all_cols(filename):
    if not os.path.exists(filename):
        print(f"ERROR: Could not find {filename}")
        return None, None, None
    try:
        data = np.loadtxt(filename)
        k = data[:, 0]
        # Load Columns 1 and 2
        if data.shape[1] > 2:
            t_cdm   = np.abs(data[:, 1])
            t_dmeff = np.abs(data[:, 2])
        else:
            t_cdm   = np.abs(data[:, 1])
            t_dmeff = np.abs(data[:, 1]) # Fallback
        
        mask = k > 0
        return k[mask], t_cdm[mask], t_dmeff[mask]
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None, None

def main():
    # 1. Load Data
    k_std, t_cdm_std, t_dmeff_std = load_transfer_all_cols(FILE_STD)
    k_idm, t_cdm_idm, t_dmeff_idm = load_transfer_all_cols(FILE_USER_IDM)
    k_ref, t_cdm_ref, t_dmeff_ref = load_transfer_all_cols(FILE_REF)

    if k_std is None or k_idm is None or k_ref is None: return

    # 2. Interpolate Baseline for Normalization
    f_std = interp1d(k_std, t_cdm_std, bounds_error=False, fill_value=0)
    
    # Get Baseline on User/Ref grids
    base_user = np.where(f_std(k_idm) == 0, 1e-30, f_std(k_idm))
    base_ref  = np.where(f_std(k_ref) == 0, 1e-30, f_std(k_ref))

    # 3. Calculate Ratios (Normalized by Baseline T_cdm)
    # T_cdm Ratios
    r_user_cdm = t_cdm_idm / base_user
    r_ref_cdm  = t_cdm_ref / base_ref
    
    # T_dmeff Ratios
    r_user_dmeff = t_dmeff_idm / base_user
    r_ref_dmeff  = t_dmeff_ref / base_ref

    # ==========================================================================
    # PLOTTING
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    
    # --- TOP LEFT: T_cdm Absolute ---
    ax = axes[0, 0]
    ax.loglog(k_std, t_cdm_std, 'k-', lw=2, alpha=0.3, label='Baseline')
    ax.loglog(k_idm, t_cdm_idm, 'b-', lw=1.5, label='My Class Run: T_cdm')
    ax.loglog(k_ref, t_cdm_ref, 'g--', lw=2, alpha=0.7, label='Reference: T_cdm')
    ax.set_title("Column 1: T_cdm", fontsize=14)
    ax.set_ylabel(r'$|T(k)|$')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # --- TOP RIGHT: T_dmeff Absolute ---
    ax = axes[0, 1]
    ax.loglog(k_std, t_cdm_std, 'k-', lw=2, alpha=0.3, label='Baseline')
    ax.loglog(k_idm, t_dmeff_idm, 'r-', lw=1.5, label='My Class Run: T_dmeff')
    ax.loglog(k_ref, t_dmeff_ref, 'm--', lw=2, alpha=0.7, label='Reference: T_dmeff')
    ax.set_title("Column 2: T_dmeff", fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # --- BOTTOM LEFT: T_cdm Ratio ---
    ax = axes[1, 0]
    ax.semilogx(k_idm, r_user_cdm, 'b-', lw=1.5, label='My Class Run / Baseline')
    ax.semilogx(k_ref, r_ref_cdm, 'g--', lw=2, alpha=0.7, label='Reference / Baseline')
    ax.axhline(1.0, color='k', linestyle=':')
    ax.set_title("Ratio: T_cdm / Baseline", fontsize=14)
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- BOTTOM RIGHT: T_dmeff Ratio ---
    ax = axes[1, 1]
    ax.semilogx(k_idm, r_user_dmeff, 'r-', lw=1.5, label='My Class Run / Baseline')
    ax.semilogx(k_ref, r_ref_dmeff, 'm--', lw=2, alpha=0.7, label='Reference / Baseline')
    ax.axhline(1.0, color='k', linestyle=':')
    ax.set_title("Ratio: T_dmeff / Baseline", fontsize=14)
    ax.set_xlabel(r"$k [h/Mpc]$")
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = 'compare_final_labels.png'
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()
