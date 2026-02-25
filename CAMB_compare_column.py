import numpy as np
import matplotlib.pyplot as plt
import os

# Replace the filename with any .dat file that was generated via CLASS
FILENAME = "output/z99_idm_n2_m1e-2_s7.1e-24_envelope_synchronous__tk.dat"

def main():
    if not os.path.exists(FILENAME):
        print(f"\nERROR: Could not find file: {FILENAME}")
        return

    print(f"Loading: {FILENAME} ...")
    data = np.loadtxt(FILENAME)

    k       = data[:, 0]
    t_cdm   = data[:, 1]  # Col 1
    t_dmeff = data[:, 2]  # Col 2
    t_b     = data[:, 3]  # Col 3
    t_tot   = data[:, 7]  # Col 7 (Total Matter)

    # Filter positive k
    mask = k > 0
    k       = k[mask]
    t_cdm   = t_cdm[mask]
    t_dmeff = t_dmeff[mask]
    t_b     = t_b[mask]
    t_tot   = t_tot[mask]

    # Create the two competing sums
    sum_cdm_b   = t_cdm + t_b
    sum_dmeff_b = t_dmeff + t_b

    # Calculate ratios relative to T_tot
    # Avoid division by zero
    safe_tot = np.where(t_tot == 0, 1e-30, t_tot)

    ratio_cdm   = sum_cdm_b / safe_tot
    ratio_dmeff = sum_dmeff_b / safe_tot

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- TOP PANEL: Absolute Values of Sums vs Total ---
    # I use abs() because transfer functions can cross zero
    ax1.loglog(k, np.abs(t_tot),       label='T_total (Reference)', color='black', linewidth=3, alpha=0.5)
    ax1.loglog(k, np.abs(sum_cdm_b),   label='Sum: T_cdm + T_b',    linestyle='--', color='blue', alpha=0.8)
    ax1.loglog(k, np.abs(sum_dmeff_b), label='Sum: T_dmeff + T_b',  linestyle='--', color='red',  alpha=0.8)

    ax1.set_ylabel(r'$|T(k)|$', fontsize=12)
    ax1.set_title(f"Summation Check: Which sum matches T_total?", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, which="both", alpha=0.3)

    # --- BOTTOM PANEL: Ratios ---
    ax2.semilogx(k, ratio_cdm,   label='(T_cdm + T_b) / T_tot',   color='blue', linestyle='--', alpha=0.7)
    ax2.semilogx(k, ratio_dmeff, label='(T_dmeff + T_b) / T_tot', color='red',  linewidth=2)

    # Reference line at 1.0
    ax2.axhline(1.0, color='black', linestyle='-', linewidth=1)

    ax2.set_ylabel(r'Ratio to $T_{total}$', fontsize=12)
    ax2.set_xlabel(r'$k \ [h/\mathrm{Mpc}]$', fontsize=12)
    ax2.grid(True, which="both", alpha=0.3)

    # Zoom in tight on y-axis to show precision
    ax2.set_ylim(0.90, 1.10)
    ax2.legend(fontsize=10)

    out_file = 'check_sums_plot.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_file}")

    # Simple console check
    print("\n--- CHECK ---")
    print(f"Mean deviation of (CDM+b):   {np.mean(np.abs(ratio_cdm - 1.0)):.2e}")
    print(f"Mean deviation of (DMEFF+b): {np.mean(np.abs(ratio_dmeff - 1.0)):.2e}")

if __name__ == "__main__":
    main()