import numpy as np
import matplotlib.pyplot as plt
import os

# Replace the filename with any .dat file that was generated via CLASS
FILENAME = "output/z99_idm_n2_m1e-2_s7.1e-24_envelope_synchronous__tk.dat"
OMEGA_M     = 0.286
OMEGA_B     = 0.047
OMEGA_DMEFF = OMEGA_M - OMEGA_B

def main():
    if not os.path.exists(FILENAME):
        print(f"Error: {FILENAME} not found.")
        return

    print(f"Loading: {FILENAME}")
    data = np.loadtxt(FILENAME)
    k       = data[:, 0]
    t_cdm   = data[:, 1]  # Tracer
    t_dmeff = data[:, 2]  # IDM
    t_b     = data[:, 3]  # Baryons
    t_tot   = data[:, 7]  # Total Matter (Col 8)

    # Filter positive k
    mask = k > 0
    k = k[mask]; t_cdm = t_cdm[mask]; t_dmeff = t_dmeff[mask]
    t_b = t_b[mask]; t_tot = t_tot[mask]


    # I calculate the weighted matter transfer function to determine which specieis should be use

    # Attempt 1: Using the Tracer (CDM) - WRONG PHYSICS
    # (0.239 * CDM + 0.047 * Baryon) / 0.286
    T_weighted_cdm = (OMEGA_DMEFF * t_cdm + OMEGA_B * t_b) / OMEGA_M

    # Attempt 2: Using the Interacting Particle (DMEFF) - CORRECT PHYSICS
    # (0.239 * DMEFF + 0.047 * Baryon) / 0.286
    T_weighted_dmeff = (OMEGA_DMEFF * t_dmeff + OMEGA_B * t_b) / OMEGA_M


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # --- TOP PANEL: Absolute Values ---
    ax1.loglog(k, np.abs(t_tot), 'k-', lw=5, alpha=0.3, label='T_total (File Output)')
    ax1.loglog(k, np.abs(T_weighted_dmeff), 'r--', lw=2, label='Weighted Avg (DMEFF + Baryon)')
    ax1.loglog(k, np.abs(T_weighted_cdm), 'b:', lw=2, label='Weighted Avg (CDM + Baryon)')

    ax1.set_title(f"Weighted Check: ( {OMEGA_DMEFF:.3f}*Dm + {OMEGA_B:.3f}*b ) / {OMEGA_M:.3f}", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel(r'$|T(k)|$')

    # --- BOTTOM PANEL: Ratio to Total ---
    # The Red Line should be perfectly flat at 1.0
    ratio_dmeff = T_weighted_dmeff / np.where(t_tot==0, 1e-30, t_tot)
    ratio_cdm   = T_weighted_cdm   / np.where(t_tot==0, 1e-30, t_tot)

    ax2.semilogx(k, ratio_cdm, 'b:', alpha=0.5, label='CDM Weighted Ratio')
    ax2.semilogx(k, ratio_dmeff, 'r-', lw=2, label='DMEFF Weighted Ratio (Should be 1.0)')

    ax2.axhline(1.0, color='k', linestyle='-')
    ax2.set_ylabel('Ratio to Total')
    ax2.set_xlabel(r'$k \ [h/\mathrm{Mpc}]$')
    ax2.set_ylim(0.99, 1.01) # Strict zoom to prove exact match
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.savefig('check_weighted_sum.png', dpi=150)
    print("Saved plot 'check_weighted_sum.png'")

if __name__ == "__main__":
    main()
