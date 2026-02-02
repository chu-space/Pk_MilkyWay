import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# 1. The NEW file path (Matches your 'ls' output exactly)
class_file = 'cdm_calibration_final_match/gold_final_flat_00_tk.dat'

# 2. The Reference file
ref_file = './enadler_cdm/test_transfer_z99.dat'

REDSHIFT = 99

# ==============================================================================
# ANALYSIS SCRIPT
# ==============================================================================

def load_data(path, label):
    print(f"\n[VERBOSE] Reading {label}...")
    if not os.path.exists(path):
        print(f"  !! ERROR: File not found at {path}")
        return None, None
    
    try:
        data = np.loadtxt(path)
        k = data[:, 0]
        # Column 1 is CDM transfer in 'format = camb'
        t_cdm = data[:, 1]
        
        print(f"  > Metadata: {len(k)} rows loaded.")
        print(f"  > k-range: [{k.min():.2e}, {k.max():.2e}] h/Mpc")
        return k, t_cdm
    except Exception as e:
        print(f"  !! ERROR processing {label}: {e}")
        return None, None

def main():
    print(f"--- CDM CALIBRATION COMPARISON (z={REDSHIFT}) ---")
    
    # 1. Load Data
    k_cls, t_cls = load_data(class_file, "My CLASS")
    k_ref, t_ref = load_data(ref_file, "Reference (Enadler)")

    if k_cls is None or k_ref is None:
        print("\n[ABORT] Could not load one or both files.")
        return

    # 2. Interpolate CLASS onto Reference grid
    itp = interp1d(k_cls, t_cls, kind='cubic', bounds_error=False, fill_value="extrapolate")
    t_cls_interp = itp(k_ref)
    
    # 3. Calculate Ratio
    ratio = t_cls_interp / t_ref

    # 4. Diagnostics Table
    print("\n" + "="*60)
    print(f"{'Scale (k)':<15} | {'Ratio':<20} | {'Dev %':<10}")
    print("-" * 60)
    
    test_ks = [0.01, 0.1, 1.0, 10.0, 100.0]
    for tk in test_ks:
        idx = (np.abs(k_ref - tk)).argmin()
        r_val = ratio[idx]
        dev = (r_val - 1.0) * 100
        print(f"k ~ {tk:<10.2f} | {r_val:<20.6f} | {dev:+.4f}%")

    # 5. Global Stats (0.01 < k < 200)
    mask = (k_ref > 0.01) & (k_ref < 200)
    if np.sum(mask) > 0:
        avg_dev = np.mean(np.abs(ratio[mask] - 1.0)) * 100
        print("-" * 60)
        print(f"AVG Deviation (0.01 < k < 200): {avg_dev:.4f}%")
        
        if avg_dev < 0.5:
            print("\n[PASS] Excellent match! (< 0.5% avg error)")
        else:
            print(f"\n[WARN] Average deviation is {avg_dev:.2f}%. Check parameters.")

    # 6. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top Panel: Transfer Functions
    ax1.loglog(k_ref, t_ref, 'k-', lw=3, alpha=0.3, label='Reference (Enadler)')
    ax1.loglog(k_ref, t_cls_interp, 'r--', lw=1.5, label='My CLASS')
    ax1.set_ylabel(r'$T_{cdm}(k)$', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.2)
    ax1.set_title(f'CDM Transfer Function (z={REDSHIFT})')

    # Bottom Panel: Ratio
    ax2.semilogx(k_ref, ratio, color='dodgerblue', lw=1.5)
    
    # Guidelines
    ax2.axhline(1.0, color='black', lw=1)
    ax2.axhline(1.005, color='red', linestyle=':', alpha=0.5, label='+/- 0.5%')
    ax2.axhline(0.995, color='red', linestyle=':', alpha=0.5)
    
    ax2.set_ylabel(r'Ratio (My CLASS / Ref)', fontsize=12)
    ax2.set_xlabel('k [h/Mpc]', fontsize=12)
    ax2.set_ylim(0.99, 1.01) 
    ax2.legend(loc='lower right')
    ax2.grid(True, which='both', alpha=0.1)

    out_img = 'cdm_comparison_result.png'
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    print(f"\n[INFO] Plot saved to {out_img}")

if __name__ == "__main__":
    main()
