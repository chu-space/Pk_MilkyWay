import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy import interpolate

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Base directory
BASE_DIR = "/home/arifchu/"

# Data directories (relative to Pk_MilkyWay)
DATA_DIRECTORY = os.path.join(BASE_DIR, "fixed_T2_grouped_CLASS_runs_VG_237_k300")
OUTPUT_PLOT_DIR = os.path.join(BASE_DIR, "Pk_MilkyWay", "fixed_T2_grouped_plots_VG_237_fixed_k300")
COZMIC_DATA_DIR = os.path.join(BASE_DIR, "Pk_MilkyWay", "COZMIC_IDM_Tk")
INDIVIDUAL_MODELS_DIR = os.path.join(OUTPUT_PLOT_DIR, "individual_models")

HALF_MODE_THRESHOLD = 0.25

# Cosmological parameters
omega_mh2 = 0.11711; h = 0.7
a = 0.0437; b = -1.188; nu = 1.049; theta = 2.012; eta = 0.2463

# Font settings for consistency
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['legend.title_fontsize'] = 10

# COZMIC model definitions - mapping of cross-sections to models
cozmic_models = [
    # (n, mass_GeV, sigma_halfmode, sigma_5.9keV, sigma_envelope)
    # n=4 models
    (4, 1e-4, 2.20e-27, 9.91e-27, 3.40e-26),
    (4, 1e-2, 1.70e-22, 9.83e-21, 1.70e-19),
    (4, 1, 8.60e-19, 2.09e-17, 2.80e-16),

    # n=2 models  
    (2, 1e-4, 4.20e-28, 1.19e-27, 2.80e-27),
    (2, 1e-2, 1.30e-25, 1.36e-24, 7.10e-24),
    (2, 1, 1.60e-23, 1.38e-22, 8.00e-22),
]

# ==============================================================================
# WDM TRANSFER FUNCTIONS
# ==============================================================================

def transfer(k,mwdm):
    """New WDM Transfer function"""
    alpha = a*(mwdm**b)*((omega_mh2/0.12)**eta)*((h/0.6736)**theta)
    transfer = (1+(alpha*k)**(2*nu))**(-5./nu)
    return transfer

def T2_wdm_old(k,mwdm):
    """Old WDM Transfer function"""
    nu = 1.12
    lambda_fs = (0.049*(mwdm**(-1.11))*((omega_mh2/h/h/0.25)**(0.11))*((h/0.7)**1.22))
    alpha = lambda_fs
    transfer = (1+(alpha*k)**(2*nu))**(-10./nu)
    return transfer

def calculate_wdm_half_mode(m_wdm_keV, formula='new'):
    """Calculate WDM half-mode wavenumber."""
    if formula == 'new':
        alpha = a * (m_wdm_keV**b) * ((omega_mh2/0.12)**eta) * ((h/0.6736)**theta)
        rhs = 0.25**(-nu/10) - 1
        k_half_hMpc = (rhs**(1/(2*nu))) / alpha
        k_half_Mpc = k_half_hMpc * h
    else:
        nu_old = 1.12
        lambda_fs = 0.049 * (m_wdm_keV**(-1.11)) * ((omega_mh2/(h*h)/0.25)**(0.11)) * ((h/0.7)**1.22)
        alpha = lambda_fs
        rhs = 0.25**(-nu_old/10) - 1
        k_half_hMpc = (rhs**(1/(2*nu_old))) / alpha
        k_half_Mpc = k_half_hMpc * h
    
    return k_half_Mpc, k_half_hMpc

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def parse_cross_section(s_str):
    """Parse cross-section string like s2.8em2700 to 2.8e-27."""
    if s_str.startswith('s'):
        s_str = s_str[1:]
    
    if 'em' in s_str:
        base_part, exp_part = s_str.split('em', 1)
        base = float(base_part)
        
        if len(exp_part) >= 2 and exp_part[:2].isdigit():
            exponent = -int(exp_part[:2])
            return base * 10**exponent
        elif len(exp_part) >= 1 and exp_part[0].isdigit():
            exponent = -int(exp_part[0])
            return base * 10**exponent
        else:
            return base * 10**(-28)
    
    elif 'e' in s_str:
        if 'e-' in s_str or 'e+' in s_str:
            base, exp = s_str.split('e', 1)
            return float(base) * 10**float(exp)
        else:
            parts = s_str.split('e')
            if len(parts) == 2:
                return float(parts[0]) * 10**float(parts[1])
    
    try:
        return float(s_str)
    except:
        return 0.0

def extract_idm_params(filename):
    """Extract n, m, sigma from filename."""
    basename = os.path.basename(filename)
    
    params = {}
    basename = basename.replace('_pk.dat', '')
    parts = basename.split('_')
    
    for part in parts:
        if part.startswith('n'):
            try:
                n_str = part[1:]
                if n_str.isdigit():
                    params['n'] = int(n_str)
                else:
                    digits = ''
                    for char in n_str:
                        if char.isdigit():
                            digits += char
                        else:
                            break
                    if digits:
                        params['n'] = int(digits)
                    else:
                        params['n'] = None
            except (ValueError, TypeError):
                params['n'] = None
        
        elif part.startswith('m'):
            try:
                m_str = part[1:]
                if 'ep' in m_str:
                    base, exp = m_str.split('ep')
                    params['m_eV'] = float(base) * 10**float(exp)
                elif 'em' in m_str:
                    base, exp = m_str.split('em')
                    params['m_eV'] = float(base) * 10**(-float(exp))
                elif 'e' in m_str:
                    if 'e-' in m_str or 'e+' in m_str:
                        base, exp = m_str.split('e', 1)
                        params['m_eV'] = float(base) * 10**float(exp)
                    else:
                        parts = m_str.split('e')
                        if len(parts) == 2:
                            params['m_eV'] = float(parts[0]) * 10**float(parts[1])
                        else:
                            params['m_eV'] = float(m_str)
                else:
                    params['m_eV'] = float(m_str)
            except (ValueError, TypeError, IndexError):
                params['m_eV'] = 0.0
        
        elif part.startswith('s'):
            try:
                s_str = part[1:]
                params['sigma_cm2'] = parse_cross_section(s_str)
            except (ValueError, TypeError, IndexError):
                params['sigma_cm2'] = 0.0
    
    if 'n' not in params:
        params['n'] = None
    if 'm_eV' not in params:
        params['m_eV'] = 0.0
    if 'sigma_cm2' not in params:
        params['sigma_cm2'] = 0.0
    
    return params

def format_cross_section(sigma_cm2):
    """Format cross-section for display."""
    if sigma_cm2 == 0:
        return "0 cm²"
    
    if sigma_cm2 >= 1e-16:
        return f"{sigma_cm2:.2e} cm²"
    elif sigma_cm2 >= 1e-19:
        return f"{sigma_cm2/1e-19:.2f}e-19 cm²"
    elif sigma_cm2 >= 1e-22:
        return f"{sigma_cm2/1e-22:.2f}e-22 cm²"
    elif sigma_cm2 >= 1e-25:
        return f"{sigma_cm2/1e-25:.2f}e-25 cm²"
    elif sigma_cm2 >= 1e-27:
        return f"{sigma_cm2/1e-27:.2f}e-27 cm²"
    else:
        return f"{sigma_cm2:.2e} cm²"

def format_mass(mass_GeV):
    """Format mass for display."""
    if mass_GeV == 1e-4:
        return "0.0001 GeV"
    elif mass_GeV == 1e-2:
        return "0.01 GeV"
    elif mass_GeV == 1:
        return "1 GeV"
    else:
        return f"{mass_GeV:.1e} GeV"

def format_mass_short(mass_GeV):
    """Format mass for short display."""
    if mass_GeV == 1e-4:
        return "1e-4"
    elif mass_GeV == 1e-2:
        return "1e-2"
    elif mass_GeV == 1:
        return "1"
    else:
        return f"{mass_GeV:.0e}"

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_npy_data(filepath):
    """Load .npy data, handling both 1D and 2D arrays."""
    try:
        data = np.load(filepath)
        
        # Handle different shapes
        if data.ndim == 1:
            return data
        elif data.ndim == 2:
            # Use last column for Pk value
            return data[:, -1]
        else:
            # Flatten if >2D
            return data.flatten()
    except Exception as e:
        print(f"    Error loading {os.path.basename(filepath)}: {e}")
        return None

def load_cozmic_model(n_val, mass_GeV, sigma_type):
    """Load specific COZMIC model based on n, mass, and sigma type."""
    # Convert mass to string for filename
    if mass_GeV == 1e-4:
        mass_str = "1e-4"
    elif mass_GeV == 1e-2:
        mass_str = "1e-2"
    elif mass_GeV == 1:
        mass_str = "1GeV"
    else:
        return None, None
    
    try:
        # Determine which files to load based on sigma_type
        if sigma_type == "envelope":
            # Load envelope data
            k_file = os.path.join(COZMIC_DATA_DIR, f"envelope_k_idm_{mass_str}_n{n_val}.npy")
            idm_file = os.path.join(COZMIC_DATA_DIR, f"envelope_idm_{mass_str}_n{n_val}.npy")
            cdm_file = os.path.join(COZMIC_DATA_DIR, f"envelope_cdm_{mass_str}_n{n_val}.npy")
        elif sigma_type == "halfmode":
            # Load half-mode data
            k_file = os.path.join(COZMIC_DATA_DIR, f"halfmode_k_idm_{mass_str}_n{n_val}.npy")
            idm_file = os.path.join(COZMIC_DATA_DIR, f"halfmode_idm_{mass_str}_n{n_val}.npy")
            cdm_file = os.path.join(COZMIC_DATA_DIR, f"halfmode_cdm_{mass_str}_n{n_val}.npy")
        else:
            return None, None
        
        # Check if files exist
        for f in [k_file, idm_file, cdm_file]:
            if not os.path.exists(f):
                print(f"    File not found: {os.path.basename(f)}")
                return None, None
        
        # Load data
        k_data = load_npy_data(k_file)
        idm_data = load_npy_data(idm_file)
        cdm_data = load_npy_data(cdm_file)
        
        if k_data is None or idm_data is None or cdm_data is None:
            return None, None
        
        # Ensure all arrays have same length
        min_len = min(len(k_data), len(idm_data), len(cdm_data))
        k_data = k_data[:min_len]
        idm_data = idm_data[:min_len]
        cdm_data = cdm_data[:min_len]
        
        # Normalize: T² = P_IDM / P_CDM
        cdm_safe = np.where(cdm_data > 0, cdm_data, 1e-100)
        T2 = idm_data / cdm_safe
        
        # Remove any NaN or inf values
        valid_mask = np.isfinite(T2)
        if not np.all(valid_mask):
            k_data = k_data[valid_mask]
            T2 = T2[valid_mask]
        
        # Clip to reasonable range
        T2 = np.clip(T2, 0, 1.1)
        
        # Convert k from h/Mpc to Mpc⁻¹
        k_Mpc = k_data * h
        
        return k_Mpc, T2
        
    except Exception as e:
        print(f"    Error loading COZMIC {sigma_type} n={n_val}, m={mass_str}: {e}")
        return None, None

def find_class_files(data_directory, n_value):
    """Find all CLASS IDM files for a specific n value."""
    print(f"\nLooking for CLASS files with n={n_value} in: {data_directory}")
    
    # Try different patterns
    patterns = [
        f"*n{n_value}*_pk.dat",
        f"idm_run_n{n_value}*_pk.dat",
        f"idm_run_*n{n_value}*_pk.dat"
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(data_directory, pattern))
        all_files.extend(files)
    
    # Remove duplicates
    all_files = list(set(all_files))
    
    print(f"Found {len(all_files)} files matching n={n_value}")
    for f in all_files:
        print(f"  {os.path.basename(f)}")
    
    return all_files

def load_class_model(file_path, cdm_data):
    """Load a single CLASS model."""
    k_cdm, P_cdm = cdm_data
    
    try:
        filename = os.path.basename(file_path)
        params = extract_idm_params(filename)
        
        # Load IDM data
        k_idm_h, P_idm_h3 = np.loadtxt(file_path, unpack=True)
        P_idm = P_idm_h3 * (h**3)
        
        # Calculate T²
        T2 = P_idm / P_cdm
        
        # Find half-mode
        half_mode_Mpc = None
        
        for i in range(len(T2)-1):
            if T2[i] >= HALF_MODE_THRESHOLD and T2[i+1] < HALF_MODE_THRESHOLD:
                k1, k2 = k_cdm[i], k_cdm[i+1]
                T1, T2_val = T2[i], T2[i+1]
                
                k_half_hMpc = np.exp(
                    np.log(k1) + (np.log(k2) - np.log(k1)) * 
                    (np.log(HALF_MODE_THRESHOLD) - np.log(T1)) / 
                    (np.log(T2_val) - np.log(T1))
                )
                
                half_mode_Mpc = k_half_hMpc * h
                break
        
        if not half_mode_Mpc:
            idx_min = np.argmin(np.abs(T2 - HALF_MODE_THRESHOLD))
            half_mode_Mpc = k_cdm[idx_min]
        
        return {
            'filename': filename,
            'params': params,
            'k_Mpc': k_cdm.copy(),
            'T2': T2,
            'half_mode_Mpc': half_mode_Mpc,
            'sigma_cm2': params.get('sigma_cm2', 0),
            'm_eV': params.get('m_eV', 0),
            'n': params.get('n', 0)
        }
        
    except Exception as e:
        print(f"    Error loading {os.path.basename(file_path)}: {e}")
        return None

# ==============================================================================
# INDIVIDUAL MODEL PLOTTING FUNCTIONS
# ==============================================================================

def create_individual_model_plots(n_value, output_dir, cozmic_data, class_models):
    """Create individual plots for each model combination."""
    
    # Create directory for individual models
    individual_dir = os.path.join(output_dir, "individual_models")
    os.makedirs(individual_dir, exist_ok=True)
    
    # Group COZMIC data by mass
    cozmic_by_mass = {}
    for data in cozmic_data:
        mass_GeV = data['mass_GeV']
        if mass_GeV not in cozmic_by_mass:
            cozmic_by_mass[mass_GeV] = []
        cozmic_by_mass[mass_GeV].append(data)
    
    # Group CLASS models by mass
    class_by_mass = {}
    for model in class_models:
        m_eV = model['m_eV']
        m_GeV = m_eV / 1e9
        
        if abs(m_GeV - 1e-4) < 1e-5:
            mass_key = 1e-4
        elif abs(m_GeV - 1e-2) < 1e-4:
            mass_key = 1e-2
        elif abs(m_GeV - 1) < 0.1:
            mass_key = 1
        else:
            mass_key = 'other'
        
        if mass_key not in class_by_mass:
            class_by_mass[mass_key] = []
        class_by_mass[mass_key].append(model)
    
    # Calculate WDM half-modes for reference
    k_half_59_Mpc, _ = calculate_wdm_half_mode(5.9, formula='new')
    k_half_65_Mpc, _ = calculate_wdm_half_mode(6.5, formula='old')
    
    # Plot for each mass
    for mass_GeV in [1e-4, 1e-2, 1]:
        if mass_GeV not in cozmic_by_mass:
            continue
            
        print(f"\nCreating individual plots for n={n_value}, m={format_mass(mass_GeV)}...")
        
        # Get COZMIC models for this mass
        cozmic_models_mass = cozmic_by_mass[mass_GeV]
        
        # Get CLASS models for this mass
        class_models_mass = class_by_mass.get(mass_GeV, [])
        
        # Create plot with COZMIC envelope and half-mode
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot WDM constraints
        kvec_Mpc = np.logspace(0, np.log10(500), 1000)
        kvec_hMpc = kvec_Mpc / h
        
        T2_new = transfer(kvec_hMpc, 5.9)**2
        T2_old = T2_wdm_old(kvec_hMpc, 6.5)
        
        ax.plot(kvec_Mpc, T2_new, linestyle='-', color='maroon', 
                linewidth=2.0, alpha=0.8, label=f"WDM 5.9 keV (k_half={k_half_59_Mpc:.1f})")
        ax.plot(kvec_Mpc, T2_old, linestyle='-', color='navy', 
                linewidth=2.0, alpha=0.8, label=f"WDM 6.5 keV (k_half={k_half_65_Mpc:.1f})")
        
        # Plot COZMIC models with different line styles
        for data in cozmic_models_mass:
            if data['type'] == 'envelope':
                linestyle = ':'  # DOTTED for envelope
                label = f"COZMIC Envelope\nσ={format_cross_section(data['sigma'])}, k_half={data.get('k_half', 0):.1f}"
                linewidth = 2.5
                color = 'forestgreen'
            else:  # halfmode
                linestyle = '-'  # SOLID for half-mode
                label = f"COZMIC Half-mode\nσ={format_cross_section(data['sigma'])}, k_half={data.get('k_half', 0):.1f}"
                linewidth = 2.5
                color = 'darkorange'
            
            ax.plot(data['k'], data['T2'],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=0.9,
                    label=label)
        
        # Plot CLASS models if available
        if class_models_mass:
            # Sort by sigma
            class_models_sorted = sorted(class_models_mass, key=lambda x: x['sigma_cm2'])
            
            # Plot min and max sigma
            if len(class_models_sorted) > 0:
                min_model = class_models_sorted[0]
                ax.plot(min_model['k_Mpc'], min_model['T2'],
                        color='royalblue',
                        linestyle='-',
                        linewidth=2.0,
                        alpha=0.9,
                        label=f"CLASS σ_min\nσ={format_cross_section(min_model['sigma_cm2'])}, k_half={min_model['half_mode_Mpc']:.1f}")
            
            if len(class_models_sorted) > 1:
                max_model = class_models_sorted[-1]
                ax.plot(max_model['k_Mpc'], max_model['T2'],
                        color='crimson',
                        linestyle='-',
                        linewidth=2.0,
                        alpha=0.9,
                        label=f"CLASS σ_max\nσ={format_cross_section(max_model['sigma_cm2'])}, k_half={max_model['half_mode_Mpc']:.1f}")
        
        # Add reference lines
        ax.axhline(y=HALF_MODE_THRESHOLD, color='gray', linestyle='--', 
                   linewidth=1.5, alpha=0.5, label=f'Half-mode threshold (T²={HALF_MODE_THRESHOLD})')
        ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, linewidth=1.5, label='CDM (T²=1)')
        
        # Format plot
        ax.set_xscale("log")
        ax.set_ylabel(r"$T^2(k) = P_{\mathrm{IDM}}(k) / P_{\mathrm{CDM}}(k)$", 
                      fontsize=14, labelpad=10)
        ax.set_xlabel(r"Wavenumber $k$ [Mpc$^{-1}$]", fontsize=14, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.set_xlim(1, 500)
        ax.set_ylim(0, 1.1)
        
        # Add subtle grid
        ax.grid(True, which='major', axis='both', linestyle=':', alpha=0.3)
        ax.grid(True, which='minor', axis='both', linestyle=':', alpha=0.1)
        
        # Create legend
        ax.legend(fontsize=9, loc='lower left', frameon=True, framealpha=0.9, facecolor='white')
        
        # Add title and info
        ax.set_title(f"Transfer Functions: n={n_value}, m={format_mass(mass_GeV)}", 
                     fontsize=16, pad=15, weight='bold')
        
        info_text = f"h = {h}, T²_half = {HALF_MODE_THRESHOLD}\n"
        info_text += f"COZMIC models: {len(cozmic_models_mass)}\n"
        info_text += f"CLASS models: {len(class_models_mass)}"
        
        ax.text(0.98, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=3))
        
        plt.tight_layout()
        
        # Save plot
        mass_str = format_mass_short(mass_GeV)
        output_file = os.path.join(individual_dir, f"transfer_n{n_value}_m{mass_str}.pdf")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        
        # Also save PNG
        png_file = os.path.join(individual_dir, f"transfer_n{n_value}_m{mass_str}.png")
        plt.savefig(png_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

def create_model_comparison_plots(n_value, output_dir, cozmic_data, class_models):
    """Create comparison plots showing all models together."""
    
    print(f"\nCreating comparison plots for n={n_value}...")
    
    # Group COZMIC data by mass
    cozmic_by_mass = {}
    for data in cozmic_data:
        mass_GeV = data['mass_GeV']
        if mass_GeV not in cozmic_by_mass:
            cozmic_by_mass[mass_GeV] = []
        cozmic_by_mass[mass_GeV].append(data)
    
    # Group CLASS models by mass
    class_by_mass = {}
    for model in class_models:
        m_eV = model['m_eV']
        m_GeV = m_eV / 1e9
        
        if abs(m_GeV - 1e-4) < 1e-5:
            mass_key = 1e-4
        elif abs(m_GeV - 1e-2) < 1e-4:
            mass_key = 1e-2
        elif abs(m_GeV - 1) < 0.1:
            mass_key = 1
        else:
            mass_key = 'other'
        
        if mass_key not in class_by_mass:
            class_by_mass[mass_key] = []
        class_by_mass[mass_key].append(model)
    
    # Calculate WDM half-modes
    k_half_59_Mpc, _ = calculate_wdm_half_mode(5.9, formula='new')
    k_half_65_Mpc, _ = calculate_wdm_half_mode(6.5, formula='old')
    
    # Create plot with all masses
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, mass_GeV in enumerate([1e-4, 1e-2, 1]):
        ax = axes[idx]
        
        # Plot WDM constraints
        kvec_Mpc = np.logspace(0, np.log10(500), 1000)
        kvec_hMpc = kvec_Mpc / h
        
        T2_new = transfer(kvec_hMpc, 5.9)**2
        T2_old = T2_wdm_old(kvec_hMpc, 6.5)
        
        ax.plot(kvec_Mpc, T2_new, linestyle='-', color='maroon', 
                linewidth=2.0, alpha=0.8, label=f"5.9 keV (k_half={k_half_59_Mpc:.1f})")
        ax.plot(kvec_Mpc, T2_old, linestyle='-', color='navy', 
                linewidth=2.0, alpha=0.8, label=f"6.5 keV (k_half={k_half_65_Mpc:.1f})")
        
        # Plot COZMIC models for this mass
        if mass_GeV in cozmic_by_mass:
            for data in cozmic_by_mass[mass_GeV]:
                if data['type'] == 'envelope':
                    linestyle = ':'  # DOTTED
                    color = 'forestgreen'
                    label = f"Envelope\nσ={format_cross_section(data['sigma'])}"
                else:  # halfmode
                    linestyle = '-'  # SOLID
                    color = 'darkorange'
                    label = f"Half-mode\nσ={format_cross_section(data['sigma'])}"
                
                ax.plot(data['k'], data['T2'],
                        color=color,
                        linestyle=linestyle,
                        linewidth=2.0,
                        alpha=0.9,
                        label=label)
        
        # Plot CLASS models for this mass
        if mass_GeV in class_by_mass:
            models_sorted = sorted(class_by_mass[mass_GeV], key=lambda x: x['sigma_cm2'])
            
            if len(models_sorted) > 0:
                min_model = models_sorted[0]
                ax.plot(min_model['k_Mpc'], min_model['T2'],
                        color='royalblue',
                        linestyle='-',
                        linewidth=1.5,
                        alpha=0.9,
                        label=f"CLASS σ_min\nk_half={min_model['half_mode_Mpc']:.1f}")
            
            if len(models_sorted) > 1:
                max_model = models_sorted[-1]
                ax.plot(max_model['k_Mpc'], max_model['T2'],
                        color='crimson',
                        linestyle='-',
                        linewidth=1.5,
                        alpha=0.9,
                        label=f"CLASS σ_max\nk_half={max_model['half_mode_Mpc']:.1f}")
        
        # Add reference lines
        ax.axhline(y=HALF_MODE_THRESHOLD, color='gray', linestyle='--', 
                   linewidth=1.0, alpha=0.5, label=f'T²={HALF_MODE_THRESHOLD}')
        ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.3, linewidth=1.0)
        
        # Format subplot
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=12)
        if idx == 0:
            ax.set_ylabel(r"$T^2(k)$", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.set_xlim(1, 500)
        ax.set_ylim(0, 1.1)
        ax.grid(True, which='major', axis='both', linestyle=':', alpha=0.2)
        
        # Add mass label
        ax.text(0.05, 0.95, f"m = {format_mass(mass_GeV)}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=2))
        
        # Legend for first subplot only
        if idx == 0:
            ax.legend(fontsize=8, loc='lower left', frameon=True, framealpha=0.9, facecolor='white')
    
    # Main title
    plt.suptitle(f"Transfer Functions Comparison: n = {n_value}", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    output_file = os.path.join(output_dir, f"transfer_comparison_n{n_value}.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comparison plot: {output_file}")
    
    # Also save PNG
    png_file = os.path.join(output_dir, f"transfer_comparison_n{n_value}.png")
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# MAIN PLOTTING FUNCTION
# ==============================================================================

def create_plot_for_n_value(n_value, output_dir):
    """Create plot for specific n value with COZMIC and CLASS data."""
    print(f"\n{'='*70}")
    print(f"Creating plots for n={n_value}")
    print('='*70)
    
    # ==================== LOAD CDM BASELINE ====================
    print("\n1. Loading CDM baseline...")
    cdm_file = os.path.join(DATA_DIRECTORY, "cdm_baseline00_pk.dat")
    if not os.path.exists(cdm_file):
        print(f"❌ CDM file not found: {cdm_file}")
        # Try alternative
        cdm_files = glob.glob(os.path.join(DATA_DIRECTORY, "*cdm*_pk.dat"))
        if cdm_files:
            cdm_file = cdm_files[0]
            print(f"   Using alternative: {os.path.basename(cdm_file)}")
        else:
            print("   ❌ No CDM files found!")
            return 0, 0
    
    k_cdm_h, P_cdm_h3 = np.loadtxt(cdm_file, unpack=True)
    k_Mpc = k_cdm_h * h
    P_cdm_Mpc3 = P_cdm_h3 * (h**3)
    cdm_data = (k_Mpc, P_cdm_Mpc3)
    
    print(f"   ✓ CDM loaded: {len(k_cdm_h)} points, k={k_Mpc[0]:.1f}-{k_Mpc[-1]:.1f} Mpc⁻¹")
    
    # ==================== LOAD CLASS IDM MODELS ====================
    print(f"\n2. Loading CLASS IDM models for n={n_value}...")
    
    # Find CLASS files
    class_files = find_class_files(DATA_DIRECTORY, n_value)
    
    # Load CLASS models
    class_models = []
    for file_path in class_files:
        model = load_class_model(file_path, cdm_data)
        if model is not None and model['n'] == n_value:
            class_models.append(model)
    
    print(f"   ✓ Loaded {len(class_models)} CLASS models")
    
    # ==================== LOAD COZMIC MODELS ====================
    print(f"\n3. Loading COZMIC reference models for n={n_value}...")
    
    # Filter COZMIC models for this n value
    cozmic_models_n = [m for m in cozmic_models if m[0] == n_value]
    
    cozmic_data = []
    for model in cozmic_models_n:
        n_val, mass_GeV, sigma_halfmode, sigma_5_9keV, sigma_envelope = model
        
        print(f"   Loading COZMIC: n={n_val}, m={format_mass(mass_GeV)}")
        
        # Load envelope model
        k_env, T2_env = load_cozmic_model(n_val, mass_GeV, "envelope")
        if k_env is not None and T2_env is not None:
            # Calculate half-mode for envelope too
            k_half_env = None
            for i in range(len(T2_env)-1):
                if T2_env[i] >= HALF_MODE_THRESHOLD and T2_env[i+1] < HALF_MODE_THRESHOLD:
                    k1, k2 = k_env[i], k_env[i+1]
                    T1, T2_val = T2_env[i], T2_env[i+1]
                    
                    k_half_env = np.exp(
                        np.log(k1) + (np.log(k2) - np.log(k1)) * 
                        (np.log(HALF_MODE_THRESHOLD) - np.log(T1)) / 
                        (np.log(T2_val) - np.log(T1))
                    )
                    break
            
            if k_half_env is None:
                idx = np.argmin(np.abs(T2_env - HALF_MODE_THRESHOLD))
                k_half_env = k_env[idx]
            
            cozmic_data.append({
                'type': 'envelope',
                'mass_GeV': mass_GeV,
                'sigma': sigma_envelope,
                'k': k_env,
                'T2': T2_env,
                'k_half': k_half_env,
                'label': f'COZMIC Envelope'
            })
            print(f"     ✓ Envelope: σ={format_cross_section(sigma_envelope)}, k_half={k_half_env:.1f} Mpc⁻¹")
        
        # Load half-mode model
        k_hm, T2_hm = load_cozmic_model(n_val, mass_GeV, "halfmode")
        if k_hm is not None and T2_hm is not None:
            # Calculate actual k_half
            k_half_actual = None
            for i in range(len(T2_hm)-1):
                if T2_hm[i] >= HALF_MODE_THRESHOLD and T2_hm[i+1] < HALF_MODE_THRESHOLD:
                    k1, k2 = k_hm[i], k_hm[i+1]
                    T1, T2_val = T2_hm[i], T2_hm[i+1]
                    
                    k_half_actual = np.exp(
                        np.log(k1) + (np.log(k2) - np.log(k1)) * 
                        (np.log(HALF_MODE_THRESHOLD) - np.log(T1)) / 
                        (np.log(T2_val) - np.log(T1))
                    )
                    break
            
            if k_half_actual is None:
                idx = np.argmin(np.abs(T2_hm - HALF_MODE_THRESHOLD))
                k_half_actual = k_hm[idx]
            
            cozmic_data.append({
                'type': 'halfmode',
                'mass_GeV': mass_GeV,
                'sigma': sigma_halfmode,
                'k': k_hm,
                'T2': T2_hm,
                'k_half': k_half_actual,
                'label': f'COZMIC Half-mode'
            })
            print(f"     ✓ Half-mode: σ={format_cross_section(sigma_halfmode)}, k_half={k_half_actual:.1f} Mpc⁻¹")
    
    print(f"   ✓ Loaded {len(cozmic_data)} COZMIC curves")
    
    # ==================== CREATE INDIVIDUAL MODEL PLOTS ====================
    print(f"\n4. Creating individual model plots...")
    create_individual_model_plots(n_value, output_dir, cozmic_data, class_models)
    
    # ==================== CREATE COMPARISON PLOTS ====================
    print(f"\n5. Creating comparison plots...")
    create_model_comparison_plots(n_value, output_dir, cozmic_data, class_models)
    
    # ==================== SAVE SUMMARY ====================
    summary_file = os.path.join(output_dir, f"summary_n{n_value}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Transfer Functions Summary - n={n_value}\n")
        f.write("="*60 + "\n\n")
        
        f.write("WDM Constraints:\n")
        f.write("-"*40 + "\n")
        k_half_59_Mpc, _ = calculate_wdm_half_mode(5.9, formula='new')
        k_half_65_Mpc, _ = calculate_wdm_half_mode(6.5, formula='old')
        f.write(f"5.9 keV: k_half = {k_half_59_Mpc:.1f} Mpc⁻¹\n")
        f.write(f"6.5 keV: k_half = {k_half_65_Mpc:.1f} Mpc⁻¹\n\n")
        
        f.write("COZMIC REFERENCE MODELS:\n")
        f.write("-"*40 + "\n")
        for data in cozmic_data:
            mass_str = format_mass(data['mass_GeV'])
            f.write(f"{data['type'].upper()}, m={mass_str}:\n")
            f.write(f"  σ = {format_cross_section(data['sigma'])}\n")
            f.write(f"  k_half = {data.get('k_half', 0):.1f} Mpc⁻¹\n\n")
        
        f.write("\nCLASS IDM MODELS:\n")
        f.write("-"*40 + "\n")
        
        # Group CLASS models by mass
        class_by_mass = {}
        for model in class_models:
            m_eV = model['m_eV']
            m_GeV = m_eV / 1e9
            
            if abs(m_GeV - 1e-4) < 1e-5:
                mass_key = 1e-4
            elif abs(m_GeV - 1e-2) < 1e-4:
                mass_key = 1e-2
            elif abs(m_GeV - 1) < 0.1:
                mass_key = 1
            else:
                mass_key = 'other'
            
            if mass_key not in class_by_mass:
                class_by_mass[mass_key] = []
            class_by_mass[mass_key].append(model)
        
        for mass_key in [1e-4, 1e-2, 1]:
            if mass_key in class_by_mass:
                models = class_by_mass[mass_key]
                models_sorted = sorted(models, key=lambda x: x['sigma_cm2'])
                
                f.write(f"{format_mass(mass_key)}: {len(models)} models\n")
                if models_sorted:
                    f.write(f"  σ_min = {format_cross_section(models_sorted[0]['sigma_cm2'])}, k_half = {models_sorted[0]['half_mode_Mpc']:.1f} Mpc⁻¹\n")
                    if len(models_sorted) > 1:
                        f.write(f"  σ_max = {format_cross_section(models_sorted[-1]['sigma_cm2'])}, k_half = {models_sorted[-1]['half_mode_Mpc']:.1f} Mpc⁻¹\n")
                f.write("\n")
    
    print(f"✓ Summary saved: {summary_file}")
    
    return len(cozmic_data), len(class_models)

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main function."""
    print("\n" + "="*70)
    print("TRANSFER FUNCTION COMPARISON")
    print("="*70)
    print(f"Base directory: {BASE_DIR}")
    print(f"CLASS data directory: {DATA_DIRECTORY}")
    print(f"COZMIC data directory: {COZMIC_DATA_DIR}")
    print(f"Output directory: {OUTPUT_PLOT_DIR}")
    print("="*70)
    
    # Check directories
    if not os.path.exists(DATA_DIRECTORY):
        print(f"❌ CLASS data directory not found: {DATA_DIRECTORY}")
        # Try to find it
        alt_path = os.path.join(BASE_DIR, "class_public-master-new-dmeff", "fixed_T2_grouped_CLASS_runs_VG_237_k300")
        if os.path.exists(alt_path):
            print(f"   Found at alternative location: {alt_path}")
            # Update the module-level variable
            import sys
            module = sys.modules[__name__]
            module.DATA_DIRECTORY = alt_path
            print(f"   Updated DATA_DIRECTORY to: {DATA_DIRECTORY}")
        else:
            print("   ❌ Could not find CLASS data directory. Exiting.")
            return
    
    if not os.path.exists(COZMIC_DATA_DIR):
        print(f"⚠️ COZMIC data directory not found: {COZMIC_DATA_DIR}")
        print("   Will plot WDM and CLASS data only.")
    
    # Create output directory
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    
    # Get unique n values from COZMIC models
    n_values = sorted(set([m[0] for m in cozmic_models]))
    print(f"\nCOZMIC models available for n values: {n_values}")
    
    # Also check what n values we have in CLASS data
    class_files = glob.glob(os.path.join(DATA_DIRECTORY, "*_pk.dat"))
    class_n_values = set()
    
    for file in class_files:
        filename = os.path.basename(file)
        if "idm_run" in filename:
            params = extract_idm_params(filename)
            n_val = params.get('n')
            if n_val is not None:
                class_n_values.add(n_val)
    
    print(f"CLASS data available for n values: {sorted(class_n_values)}")
    
    # Create plots for all n values that have data
    all_n_values = sorted(set(list(n_values) + list(class_n_values)))
    
    if not all_n_values:
        print("❌ No data found for any n value!")
        return
    
    print(f"\nCreating plots for n values: {all_n_values}")
    
    total_plots = 0
    
    for n_val in all_n_values:
        num_cozmic, num_class = create_plot_for_n_value(n_val, OUTPUT_PLOT_DIR)
        total_plots += 1
        print(f"  Created plots for n={n_val}: {num_cozmic} COZMIC curves, {num_class} CLASS models")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Created plots for {total_plots} n-values")
    print(f"Output directory: {OUTPUT_PLOT_DIR}")
    print(f"Individual model plots: {INDIVIDUAL_MODELS_DIR}")
    print("\nFiles created for each n-value:")
    print("  • Individual model plots (one per mass)")
    print("  • Comparison plot (all masses together)")
    print("  • Summary text file")
    print("\nPlot styles:")
    print("  • Envelope models: DOTTED lines")
    print("  • Half-mode models: SOLID lines")
    print("  • WDM constraints: SOLID lines")
    print("  • CLASS models: SOLID lines (σ_min and σ_max only)")
    print("="*70)

if __name__ == "__main__":
    main()
