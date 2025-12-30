import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.interpolate import InterpolatedUnivariateSpline

# Paths
cozmic_dir = "COZMIC_IDM_Tk/"
class_dir = "fixed_T2_grouped_CLASS_runs_VG_237_k300/"

# Cosmological parameters
h = 0.7  # Hubble parameter
omega_mh2 = 0.11711  # From your WDM constraint

# WDM transfer function formulas (converted to 1/Mpc)
def transfer_wdm(k_hMpc, mwdm_keV):
    """WDM transfer function from your formula, k in h/Mpc"""
    a = 0.0437
    b = -1.188
    nu = 1.049
    theta = 2.012
    eta = 0.2463
    
    alpha = a * (mwdm_keV**b) * ((omega_mh2/0.12)**eta) * ((h/0.6736)**theta)
    transfer = (1 + (alpha * k_hMpc)**(2*nu))**(-5./nu)
    return transfer

def T2_wdm_old(k_hMpc, mwdm_keV):
    """Old WDM transfer function, k in h/Mpc"""
    nu = 1.12
    lambda_fs = (0.049 * (mwdm_keV**(-1.11)) * ((omega_mh2/h/h/0.25)**(0.11)) * ((h/0.7)**1.22))
    alpha = lambda_fs
    transfer = (1 + (alpha * k_hMpc)**(2*nu))**(-10./nu)
    return transfer

# Load CLASS CDM baseline for transfer function normalization
def load_class_pk(filename):
    """Load CLASS pk.dat file, return k and P(k)"""
    data = np.loadtxt(filename)
    k_hMpc = data[:, 0]  # k in h/Mpc
    Pk = data[:, 1]  # P(k) [Mpc^3/h^3]
    k_Mpc = k_hMpc * h  # Convert to 1/Mpc
    return k_Mpc, Pk

def load_class_tk(filename, species_idx=2):
    """Load CLASS tk.dat file, return k and T for specific species"""
    data = np.loadtxt(filename)
    k_hMpc = data[:, 0]  # k in h/Mpc
    T = data[:, species_idx]  # Transfer function
    k_Mpc = k_hMpc * h  # Convert to 1/Mpc
    return k_Mpc, T

# Load COZMIC data with flexible format
def load_cozmic_pk(pk_file, k_file=None):
    """Load COZMIC .npy file, handle both 1-col and 2-col formats"""
    data = np.load(pk_file)
    
    if data.ndim == 1:
        # 1 column: just P(k)
        Pk = data
        if k_file is not None:
            k_hMpc = np.load(k_file)
        else:
            k_hMpc = np.arange(len(Pk))
    elif data.ndim == 2 and data.shape[1] == 2:
        # 2 columns: [k, P(k)]
        k_hMpc = data[:, 0]
        Pk = data[:, 1]
    else:
        raise ValueError(f"Unexpected shape in {pk_file}: {data.shape}")
    
    k_Mpc = k_hMpc * h  # Convert to 1/Mpc
    return k_Mpc, Pk

# Load baseline CDM power spectrum from CLASS
k_cdm_Mpc_baseline, Pk_cdm_baseline = load_class_pk(f"{class_dir}/cdm_baseline00_pk.dat")

# Group models by (n, m_idm) for plotting
models_by_group = {}

# Define models from your table
models = [
    # (n, m_idm, sigma0, name_suffix, sigma0_str for filename)
    (2, 1e-4, 4.2e-28, "n2_m1em04_s4.2em2801", "4.2e-28"),
    (2, 1e-4, 2.8e-27, "n2_m1em04_s2.8em2701", "2.8e-27"),
    (2, 1e-2, 1.3e-25, "n2_m1em02_s1.3em2501", "1.3e-25"),
    (2, 1e-2, 7.1e-24, "n2_m1em02_s7.1em2401", "7.1e-24"),
    (2, 1, 1.6e-23, "n2_m1ep00_s1.6em2301", "1.6e-23"),
    (2, 1, 8.0e-22, "n2_m1ep00_s8.0em2201", "8.0e-22"),
    (4, 1e-4, 2.2e-27, "n4_m1em04_s2.2em2701", "2.2e-27"),
    (4, 1e-4, 3.4e-26, "n4_m1em04_s3.4em2601", "3.4e-26"),
    (4, 1e-2, 1.7e-22, "n4_m1em02_s1.7em2201", "1.7e-22"),
    (4, 1e-2, 1.7e-19, "n4_m1em02_s1.7em1901", "1.7e-19"),
    (4, 1, 8.6e-19, "n4_m1ep00_s8.6em1901", "8.6e-19"),
    (4, 1, 2.8e-16, "n4_m1ep00_s2.8em1601", "2.8e-16")
]

# Process each model
for n, m_idm, sigma0, name_suffix, sigma0_str in models:
    print(f"Processing n={n}, m={m_idm}, σ={sigma0:.1e}")
    
    # 1. Try to load CLASS power spectrum first
    class_pattern = f"idm_run_{name_suffix}_pk.dat"
    class_files = glob.glob(f"{class_dir}/{class_pattern}")
    
    if not class_files:
        print(f"  WARNING: CLASS pk file not found: {class_pattern}")
        continue
    
    class_file = class_files[0]
    k_class_Mpc, Pk_class_idm = load_class_pk(class_file)
    
    # Interpolate CDM baseline to same k points
    Pk_cdm_interp = np.interp(k_class_Mpc, k_cdm_Mpc_baseline, Pk_cdm_baseline)
    
    # Calculate T^2 = P_idm/P_cdm (NO square root!)
    T2_ratio_class = Pk_class_idm / Pk_cdm_interp
    
    # 2. Load COZMIC data for this model
    m_str = f"{m_idm:.0e}".replace('+', '').replace('-0', '-')
    
    # IDM files
    env_idm_file = f"{cozmic_dir}/envelope_idm_{m_str}_n{n}.npy"
    env_k_file = f"{cozmic_dir}/envelope_k_idm_{m_str}_n{n}.npy"
    hm_idm_file = f"{cozmic_dir}/halfmode_idm_{m_str}_n{n}.npy"
    hm_k_file = f"{cozmic_dir}/halfmode_k_idm_{m_str}_n{n}.npy"
    
    # CDM files
    env_cdm_file = f"{cozmic_dir}/envelope_cdm_{m_str}_n{n}.npy"
    hm_cdm_file = f"{cozmic_dir}/halfmode_cdm_{m_str}_n{n}.npy"
    
    try:
        # Load COZMIC data
        k_env_Mpc, Pk_env_idm = load_cozmic_pk(env_idm_file, env_k_file)
        k_hm_Mpc, Pk_hm_idm = load_cozmic_pk(hm_idm_file, hm_k_file)
        _, Pk_env_cdm = load_cozmic_pk(env_cdm_file)
        _, Pk_hm_cdm = load_cozmic_pk(hm_cdm_file)
        
        # Calculate COZMIC T^2 ratios (NO square root!)
        T2_ratio_env = Pk_env_idm / Pk_env_cdm
        T2_ratio_hm = Pk_hm_idm / Pk_hm_cdm
        
        # Store for grouped plotting
        key = (n, m_idm)
        if key not in models_by_group:
            models_by_group[key] = []
        
        models_by_group[key].append({
            'sigma0': sigma0,
            'sigma0_str': sigma0_str,
            'k_class_Mpc': k_class_Mpc,
            'k_class_hMpc': k_class_Mpc / h,  # Keep both units
            'T2_ratio_class': T2_ratio_class,
            'k_env_Mpc': k_env_Mpc,
            'k_env_hMpc': k_env_Mpc / h,
            'T2_ratio_env': T2_ratio_env,
            'k_hm_Mpc': k_hm_Mpc,
            'k_hm_hMpc': k_hm_Mpc / h,
            'T2_ratio_hm': T2_ratio_hm
        })
        
    except FileNotFoundError as e:
        print(f"  ERROR: File not found: {e}")
        continue
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

# Create output directory
os.makedirs("transfer_comparisons", exist_ok=True)

# Create WDM constraint line
k_wdm_hMpc = np.logspace(0, np.log10(500), 1000)  # h/Mpc
k_wdm_Mpc = k_wdm_hMpc * h  # Convert to 1/Mpc
T2_wdm_new = transfer_wdm(k_wdm_hMpc, 5.9)**2  # 5.9 keV WDM
T2_wdm_old = T2_wdm_old(k_wdm_hMpc, 6.5)  # 6.5 keV WDM (old formula)

# Create plots for each (n, m_idm) group
for (n, m_idm), model_list in models_by_group.items():
    print(f"\nCreating plot for n={n}, m={m_idm}")
    
    # Sort by sigma0 for consistent colors
    model_list.sort(key=lambda x: x['sigma0'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Color cycle for different sigma0 values
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_list)))
    
    # Plot WDM constraints on both subplots
    ax1.plot(k_wdm_Mpc, T2_wdm_new, '--', color='red', lw=2, alpha=0.7, 
             label='5.9 keV WDM (new)')
    ax1.plot(k_wdm_Mpc, T2_wdm_old, '-.', color='blue', lw=2, alpha=0.7,
             label='6.5 keV WDM (old)')
    
    ax2.plot(k_wdm_Mpc, T2_wdm_new, '--', color='red', lw=2, alpha=0.7)
    ax2.plot(k_wdm_Mpc, T2_wdm_old, '-.', color='blue', lw=2, alpha=0.7)
    
    # Plot each sigma0
    for i, model in enumerate(model_list):
        color = colors[i]
        label = f"σ₀ = {model['sigma0_str']} cm²"
        
        # Plot CLASS results on top subplot
        ax1.loglog(model['k_class_Mpc'], model['T2_ratio_class'], 
                  color=color, lw=2, label=label)
        
        # Plot COZMIC envelope and halfmode on bottom subplot
        ax2.loglog(model['k_env_Mpc'], model['T2_ratio_env'], '--',
                  color=color, lw=1.5, alpha=0.8)
        ax2.loglog(model['k_hm_Mpc'], model['T2_ratio_hm'], ':',
                  color=color, lw=1.5, alpha=0.8)
    
    # Format top plot (CLASS results)
    ax1.set_ylabel(r'$T^2_{\mathrm{IDM}}(k) = P_{\mathrm{IDM}}(k)/P_{\mathrm{CDM}}(k)$', fontsize=14)
    ax1.set_title(f'CLASS: n={n}, m={m_idm:.0e} GeV', fontsize=14)
    ax1.legend(fontsize=10, loc='lower left', ncol=2)
    ax1.set_ylim([1e-3, 1.2])
    ax1.set_xlim([0.1, 500 * h])  # 500 h/Mpc converted to 1/Mpc
    
    # Format bottom plot (COZMIC fits)
    ax2.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=14)
    ax2.set_ylabel(r'$T^2_{\mathrm{IDM}}(k) = P_{\mathrm{IDM}}(k)/P_{\mathrm{CDM}}(k)$', fontsize=14)
    ax2.set_title(f'COZMIC fits: n={n}, m={m_idm:.0e} GeV', fontsize=14)
    ax2.set_ylim([1e-3, 1.2])
    
    # Add legend for line styles in bottom plot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Envelope'),
        Line2D([0], [0], color='gray', linestyle=':', lw=2, label='Half-mode'),
        Line2D([0], [0], color='red', linestyle='--', lw=2, label='5.9 keV WDM'),
        Line2D([0], [0], color='blue', linestyle='-.', lw=2, label='6.5 keV WDM')
    ]
    ax2.legend(handles=legend_elements, fontsize=10, loc='lower left')
    
    # Remove grids
    ax1.grid(False)
    ax2.grid(False)
    
    plt.tight_layout()
    
    # Save plot
    m_str_clean = f"{m_idm:.0e}".replace('+', '').replace('.0', '')
    plot_file = f"transfer_comparisons/T2_ratio_n{n}_m{m_str_clean}_grouped.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    plt.close()

# Also create individual comparison plots
print("\nCreating individual comparison plots...")
for (n, m_idm), model_list in models_by_group.items():
    for model in model_list:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot WDM constraints
        ax.plot(k_wdm_Mpc, T2_wdm_new, '--', color='red', lw=2, alpha=0.7, 
                label='5.9 keV WDM (new)')
        ax.plot(k_wdm_Mpc, T2_wdm_old, '-.', color='blue', lw=2, alpha=0.7,
                label='6.5 keV WDM (old)')
        
        # Plot CLASS result
        ax.loglog(model['k_class_Mpc'], model['T2_ratio_class'], 
                 'k-', lw=2.5, label='CLASS')
        
        # Plot COZMIC fits
        ax.loglog(model['k_env_Mpc'], model['T2_ratio_env'], 'b--', 
                 lw=2, alpha=0.8, label='COZMIC envelope')
        ax.loglog(model['k_hm_Mpc'], model['T2_ratio_hm'], 'g:', 
                 lw=2, alpha=0.8, label='COZMIC half-mode')
        
        ax.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=14)
        ax.set_ylabel(r'$T^2_{\mathrm{IDM}}(k) = P_{\mathrm{IDM}}(k)/P_{\mathrm{CDM}}(k)$', fontsize=14)
        
        title = (f'n={n}, m={m_idm:.0e} GeV, '
                f'σ₀ = {model["sigma0_str"]} cm²')
        ax.set_title(title, fontsize=14)
        
        ax.legend(fontsize=10, loc='lower left')
        ax.set_ylim([1e-3, 1.2])
        ax.set_xlim([0.1, 500 * h])  # 500 h/Mpc converted to 1/Mpc
        
        # Remove grid
        ax.grid(False)
        
        plt.tight_layout()
        
        # Save individual plot
        m_str_clean = f"{m_idm:.0e}".replace('+', '').replace('.0', '')
        sigma_str_clean = model["sigma0_str"].replace('.', 'p').replace('-', 'm').replace('+', 'p')
        plot_file = f"transfer_comparisons/T2_ratio_n{n}_m{m_str_clean}_sigma{sigma_str_clean}_individual.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

print("\nAll plots saved in 'transfer_comparisons/' directory")
