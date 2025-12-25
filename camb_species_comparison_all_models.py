import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.interpolate import InterpolatedUnivariateSpline
import glob

# ==============================================================================
# Configuration
# ==============================================================================

# Base directories
BASE_DIR = "/home/arifchu"
VG_CLASS_DIR = os.path.join(BASE_DIR, "class_public-master-new-dmeff/fixed_correct_middle_camb")
REF_CAMB_DIR = os.path.join(BASE_DIR, "Pk_MilkyWay/camb_data_tk")
OUTPUT_DIR = "CAMB_species_comparisons_all_models"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Column finder and data loader
# ==============================================================================

def extract_column_names(filepath):
    """Extract column names from a CLASS output file."""
    column_names = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find the last comment line with column descriptions
        header_line = None
        for line in reversed(lines):
            if line.strip().startswith('#'):
                header_line = line.strip()
                break
        
        if header_line:
            # Parse column names from header like: "# column 1:z 2:H[1/Mpc] ..."
            matches = re.findall(r'\d+:\s*([^\s]+)', header_line)
            if matches:
                column_names = matches
            else:
                # Alternative format: just space separated names after #
                names_part = header_line.lstrip('#').strip()
                column_names = names_part.split()
    except Exception as e:
        print(f"Warning: Could not extract column names from {filepath}: {e}")
    
    return column_names

def find_column_index(column_names, patterns):
    """Find column index by matching patterns."""
    for i, col_name in enumerate(column_names):
        for pattern in patterns:
            if pattern.lower() in col_name.lower():
                return i
    return -1

def load_vg_data(file_prefix, model_dir):
    """Load VG data files with given prefix."""
    # Pattern match for the processed_Tk files
    pattern = os.path.join(model_dir, f"{file_prefix}*.dat")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    # Load the data file
    data = np.loadtxt(files[0])
    
    # Try to extract column names from the first few lines
    column_names = []
    with open(files[0], 'r') as f:
        for line in f:
            if line.strip().startswith('#'):
                # Extract column names from header
                header = line.strip().lstrip('#').strip()
                # Try different parsing methods
                if ':' in header:
                    # Format: "1:k/h 2:T_cdm ..."
                    parts = header.split()
                    for part in parts:
                        if ':' in part:
                            col_name = part.split(':')[1]
                            column_names.append(col_name)
                else:
                    # Space-separated names
                    column_names = header.split()
                break
    
    return data, column_names if column_names else None

def load_reference_data(ref_file):
    """Load reference CAMB data."""
    data = np.loadtxt(ref_file)
    return data

# ==============================================================================
# Model configuration
# ==============================================================================

# Define all models based on your directory listing
models = [
    # n=2 models - halfmode
    {
        'name': 'n2_m1e-4_halfmode',
        'vg_prefix': 'processed_Tk_n2_m1e-4_s4.2e-28',
        'camb_ref': 'idm_n2_1e-4GeV_halfmode_z99_Tk.dat',
        'mass': '1e-4 GeV',
        'sigma': '4.2e-28',
        'n': 2,
        'type': 'halfmode'
    },
    {
        'name': 'n2_m1e-4_halfmode_s1.19e-27',
        'vg_prefix': 'processed_Tk_n2_m1e-4_s1.19e-27',
        'camb_ref': 'idm_n2_1e-4GeV_halfmode_z99_Tk.dat',
        'mass': '1e-4 GeV',
        'sigma': '1.19e-27',
        'n': 2,
        'type': 'halfmode'
    },
    {
        'name': 'n2_m1e-4_halfmode_s2.8e-27',
        'vg_prefix': 'processed_Tk_n2_m1e-4_s2.8e-27',
        'camb_ref': 'idm_n2_1e-4GeV_halfmode_z99_Tk.dat',
        'mass': '1e-4 GeV',
        'sigma': '2.8e-27',
        'n': 2,
        'type': 'halfmode'
    },
    {
        'name': 'n2_m0.01_halfmode_s1.3e-25',
        'vg_prefix': 'processed_Tk_n2_m0.01_s1.3e-25',
        'camb_ref': 'idm_n2_1e-2GeV_halfmode_z99_Tk.dat',
        'mass': '0.01 GeV',
        'sigma': '1.3e-25',
        'n': 2,
        'type': 'halfmode'
    },
    {
        'name': 'n2_m0.01_halfmode_s1.36e-24',
        'vg_prefix': 'processed_Tk_n2_m0.01_s1.36e-24',
        'camb_ref': 'idm_n2_1e-2GeV_halfmode_z99_Tk.dat',
        'mass': '0.01 GeV',
        'sigma': '1.36e-24',
        'n': 2,
        'type': 'halfmode'
    },
    {
        'name': 'n2_m1_halfmode_s1.6e-23',
        'vg_prefix': 'processed_Tk_n2_m1_s1.6e-23',
        'camb_ref': 'idm_n2_1GeV_halfmode_z99_Tk.dat',
        'mass': '1 GeV',
        'sigma': '1.6e-23',
        'n': 2,
        'type': 'halfmode'
    },
    {
        'name': 'n2_m1_halfmode_s8e-22',
        'vg_prefix': 'processed_Tk_n2_m1_s8e-22',
        'camb_ref': 'idm_n2_1GeV_halfmode_z99_Tk.dat',
        'mass': '1 GeV',
        'sigma': '8e-22',
        'n': 2,
        'type': 'halfmode'
    },
    {
        'name': 'n2_m1_halfmode_s1.38e-22',
        'vg_prefix': 'processed_Tk_n2_m1_s1.38e-22',
        'camb_ref': 'idm_n2_1GeV_halfmode_z99_Tk.dat',
        'mass': '1 GeV',
        'sigma': '1.38e-22',
        'n': 2,
        'type': 'halfmode'
    },
    # n=2 models - envelope
    {
        'name': 'n2_m1e-4_envelope_s2.8e-27',
        'vg_prefix': 'processed_Tk_n2_m1e-4_s2.8e-27',
        'camb_ref': 'idm_n2_1e-4GeV_envelope_z99_Tk.dat',
        'mass': '1e-4 GeV',
        'sigma': '2.8e-27',
        'n': 2,
        'type': 'envelope'
    },
    {
        'name': 'n2_m0.01_envelope_s1.36e-24',
        'vg_prefix': 'processed_Tk_n2_m0.01_s1.36e-24',
        'camb_ref': 'idm_n2_1e-2GeV_envelope_z99_Tk.dat',
        'mass': '0.01 GeV',
        'sigma': '1.36e-24',
        'n': 2,
        'type': 'envelope'
    },
    {
        'name': 'n2_m0.01_envelope_s1.7e-24',
        'vg_prefix': 'processed_Tk_n2_m0.01_s1.36e-24',  # Using available file
        'camb_ref': 'idm_n2_1e-2GeV_envelope_z99_Tk.dat',
        'mass': '0.01 GeV',
        'sigma': '1.36e-24',
        'n': 2,
        'type': 'envelope'
    },
    {
        'name': 'n2_m1_envelope_s1.38e-22',
        'vg_prefix': 'processed_Tk_n2_m1_s1.38e-22',
        'camb_ref': 'idm_n2_1GeV_envelope_z99_Tk.dat',
        'mass': '1 GeV',
        'sigma': '1.38e-22',
        'n': 2,
        'type': 'envelope'
    },
    {
        'name': 'n2_m1_envelope_s8e-22',
        'vg_prefix': 'processed_Tk_n2_m1_s8e-22',
        'camb_ref': 'idm_n2_1GeV_envelope_z99_Tk.dat',
        'mass': '1 GeV',
        'sigma': '8e-22',
        'n': 2,
        'type': 'envelope'
    },
    # n=4 models - halfmode
    {
        'name': 'n4_m1e-4_halfmode_s2.2e-27',
        'vg_prefix': 'processed_Tk_n4_m1e-4_s2.2e-27',
        'camb_ref': 'idm_1e-4GeV_halfmode_Tk.dat',
        'mass': '1e-4 GeV',
        'sigma': '2.2e-27',
        'n': 4,
        'type': 'halfmode'
    },
    {
        'name': 'n4_m1e-4_halfmode_s3.4e-26',
        'vg_prefix': 'processed_Tk_n4_m1e-4_s3.4e-26',
        'camb_ref': 'idm_1e-4GeV_halfmode_Tk.dat',
        'mass': '1e-4 GeV',
        'sigma': '3.4e-26',
        'n': 4,
        'type': 'halfmode'
    },
    {
        'name': 'n4_m1e-4_halfmode_s9.91e-27',
        'vg_prefix': 'processed_Tk_n4_m1e-4_s9.91e-27',
        'camb_ref': 'idm_1e-4GeV_halfmode_Tk.dat',
        'mass': '1e-4 GeV',
        'sigma': '9.91e-27',
        'n': 4,
        'type': 'halfmode'
    },
    {
        'name': 'n4_m0.01_halfmode_s1.7e-19',
        'vg_prefix': 'processed_Tk_n4_m0.01_s1.7e-19',
        'camb_ref': 'idm_1e-2GeV_envelope_Tk.dat',  # Using envelope as reference
        'mass': '0.01 GeV',
        'sigma': '1.7e-19',
        'n': 4,
        'type': 'halfmode'
    },
    {
        'name': 'n4_m0.01_halfmode_s1.7e-22',
        'vg_prefix': 'processed_Tk_n4_m0.01_s1.7e-22',
        'camb_ref': 'idm_1e-2GeV_envelope_Tk.dat',  # Using envelope as reference
        'mass': '0.01 GeV',
        'sigma': '1.7e-22',
        'n': 4,
        'type': 'halfmode'
    },
    {
        'name': 'n4_m0.01_halfmode_s9.83e-21',
        'vg_prefix': 'processed_Tk_n4_m0.01_s9.83e-21',
        'camb_ref': 'idm_1e-2GeV_envelope_Tk.dat',  # Using envelope as reference
        'mass': '0.01 GeV',
        'sigma': '9.83e-21',
        'n': 4,
        'type': 'halfmode'
    },
    # n=4 models - envelope
    {
        'name': 'n4_m1e-4_envelope_s9.91e-27',
        'vg_prefix': 'processed_Tk_n4_m1e-4_s9.91e-27',
        'camb_ref': 'idm_1e-4GeV_envelope_Tk.dat',
        'mass': '1e-4 GeV',
        'sigma': '9.91e-27',
        'n': 4,
        'type': 'envelope'
    },
    {
        'name': 'n4_m0.01_envelope_s9.83e-21',
        'vg_prefix': 'processed_Tk_n4_m0.01_s9.83e-21',
        'camb_ref': 'idm_1e-2GeV_envelope_Tk.dat',
        'mass': '0.01 GeV',
        'sigma': '9.83e-21',
        'n': 4,
        'type': 'envelope'
    },
    {
        'name': 'n4_m1_envelope_s2.09e-17',
        'vg_prefix': 'processed_Tk_n4_m1_s2.09e-17',
        'camb_ref': 'idm_1GeV_envelope_Tk.dat',
        'mass': '1 GeV',
        'sigma': '2.09e-17',
        'n': 4,
        'type': 'envelope'
    },
    {
        'name': 'n4_m1_envelope_s2.8e-16',
        'vg_prefix': 'processed_Tk_n4_m1_s2.8e-16',
        'camb_ref': 'idm_1GeV_envelope_Tk.dat',
        'mass': '1 GeV',
        'sigma': '2.8e-16',
        'n': 4,
        'type': 'envelope'
    },
    {
        'name': 'n4_m1_envelope_s8.6e-19',
        'vg_prefix': 'processed_Tk_n4_m1_s8.6e-19',
        'camb_ref': 'idm_1GeV_envelope_Tk.dat',
        'mass': '1 GeV',
        'sigma': '8.6e-19',
        'n': 4,
        'type': 'envelope'
    },
]

# ==============================================================================
# Data processing functions
# ==============================================================================

def identify_vg_columns(data, column_names):
    """Identify important columns in VG data."""
    # If no column names provided, use default indices
    if column_names is None or len(column_names) < 6:
        print("Warning: Using default column indices for VG data")
        return {
            'k': 0,
            'T_cdm': 1,
            'T_b': 2,
            'T_tot': 5,
            'v_cdm': 9,
            'v_b': 10
        }
    
    # Try to identify columns
    col_map = {}
    
    # Find k column
    for i, name in enumerate(column_names):
        if any(pattern in name.lower() for pattern in ['k', 'k/h']):
            col_map['k'] = i
            break
    if 'k' not in col_map:
        col_map['k'] = 0
    
    # Find T_cdm column
    for i, name in enumerate(column_names):
        if any(pattern in name.lower() for pattern in ['cdm', 'dmeff', 'd_cdm']):
            col_map['T_cdm'] = i
            break
    if 'T_cdm' not in col_map:
        col_map['T_cdm'] = 1
    
    # Find T_b column
    for i, name in enumerate(column_names):
        if any(pattern in name.lower() for pattern in ['b', 'baryon', 'd_b']):
            col_map['T_b'] = i
            break
    if 'T_b' not in col_map:
        col_map['T_b'] = 2
    
    # Find T_tot column
    for i, name in enumerate(column_names):
        if any(pattern in name.lower() for pattern in ['tot', 'total']):
            col_map['T_tot'] = i
            break
    if 'T_tot' not in col_map:
        col_map['T_tot'] = 5
    
    # Find v_cdm column
    for i, name in enumerate(column_names):
        if any(pattern in name.lower() for pattern in ['v_cdm', 'vcdm', 'theta_cdm']):
            col_map['v_cdm'] = i
            break
    if 'v_cdm' not in col_map:
        col_map['v_cdm'] = 9
    
    # Find v_b column
    for i, name in enumerate(column_names):
        if any(pattern in name.lower() for pattern in ['v_b', 'vb', 'theta_b']):
            col_map['v_b'] = i
            break
    if 'v_b' not in col_map:
        col_map['v_b'] = 10
    
    print(f"Identified columns: k={col_map['k']}, T_cdm={col_map['T_cdm']}, T_b={col_map['T_b']}, "
          f"T_tot={col_map['T_tot']}, v_cdm={col_map['v_cdm']}, v_b={col_map['v_b']}")
    
    return col_map

def process_vg_data(data, column_names):
    """Process VG data to extract relevant quantities."""
    col_map = identify_vg_columns(data, column_names)
    
    k = data[:, col_map['k']]
    T_cdm = data[:, col_map['T_cdm']]
    T_b = data[:, col_map['T_b']]
    T_tot = data[:, col_map['T_tot']]
    v_cdm = data[:, col_map['v_cdm']]
    v_b = data[:, col_map['v_b']]
    
    return {
        'k': k,
        'T_cdm': T_cdm,
        'T_b': T_b,
        'T_tot': T_tot,
        'v_cdm': v_cdm,
        'v_b': v_b
    }

def process_ref_data(data):
    """Process reference CAMB data."""
    # CAMB format typically has:
    # 0: k/h, 1: CDM, 2: baryon, 6: total, 10: v_CDM, 11: v_b
    return {
        'k': data[:, 0],
        'T_cdm': data[:, 1],
        'T_b': data[:, 2],
        'T_tot': data[:, 6],
        'v_cdm': data[:, 10],
        'v_b': data[:, 11]
    }

# ==============================================================================
# Plotting functions
# ==============================================================================

def plot_species_comparison(model_info, vg_data, ref_data):
    """Plot comparison of individual species."""
    
    model_name = model_info['name']
    model_type = model_info['type'].capitalize()
    
    # Create figure with subplots for each species
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Species to plot
    species_info = [
        {
            'name': 'CDM Transfer',
            'vg_data': vg_data['T_cdm']**2,
            'ref_data': ref_data['T_cdm']**2,
            'k_vg': vg_data['k'],
            'k_ref': ref_data['k'],
            'ylabel': '$T_{cdm}^2(k)$',
            'ax_idx': 0
        },
        {
            'name': 'Baryon Transfer',
            'vg_data': vg_data['T_b']**2,
            'ref_data': ref_data['T_b']**2,
            'k_vg': vg_data['k'],
            'k_ref': ref_data['k'],
            'ylabel': '$T_{b}^2(k)$',
            'ax_idx': 1
        },
        {
            'name': 'Total Transfer',
            'vg_data': vg_data['T_tot']**2,
            'ref_data': ref_data['T_tot']**2,
            'k_vg': vg_data['k'],
            'k_ref': ref_data['k'],
            'ylabel': '$T_{tot}^2(k)$',
            'ax_idx': 2
        },
        {
            'name': 'CDM Velocity',
            'vg_data': np.abs(vg_data['v_cdm']),
            'ref_data': np.abs(ref_data['v_cdm']),
            'k_vg': vg_data['k'],
            'k_ref': ref_data['k'],
            'ylabel': '$v_{cdm}(k)$',
            'ax_idx': 3
        },
        {
            'name': 'Baryon Velocity',
            'vg_data': np.abs(vg_data['v_b']),
            'ref_data': np.abs(ref_data['v_b']),
            'k_vg': vg_data['k'],
            'k_ref': ref_data['k'],
            'ylabel': '$v_{b}(k)$',
            'ax_idx': 4
        },
        {
            'name': 'Velocity Ratio',
            'vg_data': np.abs(vg_data['v_cdm'] / vg_data['v_b']),
            'ref_data': np.abs(ref_data['v_cdm'] / ref_data['v_b']),
            'k_vg': vg_data['k'],
            'k_ref': ref_data['k'],
            'ylabel': '$v_{cdm}/v_{b}$',
            'ax_idx': 5
        }
    ]
    
    for species in species_info:
        ax = axes[species['ax_idx']]
        
        # Interpolate to common k-range for ratio plot
        k_min = max(species['k_vg'].min(), species['k_ref'].min())
        k_max = min(species['k_vg'].max(), species['k_ref'].max())
        
        if k_min >= k_max:
            print(f"  Warning: No overlap in k-range for {species['name']}")
            continue
            
        k_common = np.logspace(np.log10(k_min), np.log10(k_max), 500)
        
        # Interpolate both datasets to common k-grid
        vg_interp = np.interp(k_common, species['k_vg'], species['vg_data'])
        ref_interp = np.interp(k_common, species['k_ref'], species['ref_data'])
        ratio = vg_interp / ref_interp
        
        # Plot transfer functions (left y-axis)
        ax.loglog(species['k_vg'], species['vg_data'], 'b-', linewidth=1.5, 
                 label='VG CLASS', alpha=0.8)
        ax.loglog(species['k_ref'], species['ref_data'], 'r--', linewidth=1.5, 
                 label='CAMB Ref', alpha=0.8)
        
        ax.set_xlabel('k [h/Mpc]', fontsize=10)
        ax.set_ylabel(species['ylabel'], fontsize=10, color='black')
        ax.set_title(species['name'], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.set_xlim(1e-3, 1e2)
        
        # Add ratio plot on right y-axis
        ax2 = ax.twinx()
        ax2.semilogx(k_common, ratio, 'g-', linewidth=1, alpha=0.7, label='Ratio')
        ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax2.set_ylabel('VG/Ref Ratio', fontsize=9, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0.1, 10)
        ax2.set_yscale('log')
    
    # Add model info to title
    title = (f'Species Comparison: n={model_info["n"]}, m={model_info["mass"]}, '
             f'{model_type}\nσ={model_info["sigma"]}, {model_info["name"]}')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    output_path = os.path.join(OUTPUT_DIR, f"species_comparison_{model_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Created species comparison plot: {output_path}")
    
    return output_path

def plot_detailed_species(model_info, vg_data, ref_data):
    """Create detailed individual plots for each species."""
    
    model_name = model_info['name']
    model_type = model_info['type'].capitalize()
    species_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(species_dir, exist_ok=True)
    
    # Individual species plots
    species_plots = [
        ('cdm_transfer', vg_data['T_cdm']**2, ref_data['T_cdm']**2, 
         '$T_{cdm}^2(k)$', vg_data['k'], ref_data['k']),
        ('baryon_transfer', vg_data['T_b']**2, ref_data['T_b']**2, 
         '$T_{b}^2(k)$', vg_data['k'], ref_data['k']),
        ('total_transfer', vg_data['T_tot']**2, ref_data['T_tot']**2, 
         '$T_{tot}^2(k)$', vg_data['k'], ref_data['k']),
        ('cdm_velocity', np.abs(vg_data['v_cdm']), np.abs(ref_data['v_cdm']), 
         '$|v_{cdm}(k)|$', vg_data['k'], ref_data['k']),
        ('baryon_velocity', np.abs(vg_data['v_b']), np.abs(ref_data['v_b']), 
         '$|v_{b}(k)|$', vg_data['k'], ref_data['k']),
        ('velocity_ratio', np.abs(vg_data['v_cdm']/vg_data['v_b']), 
         np.abs(ref_data['v_cdm']/ref_data['v_b']), 
         '$|v_{cdm}/v_{b}|$', vg_data['k'], ref_data['k']),
    ]
    
    for species_name, vg_species, ref_species, ylabel, k_vg, k_ref in species_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Interpolate to common k-range
        k_min = max(k_vg.min(), k_ref.min())
        k_max = min(k_vg.max(), k_ref.max())
        
        if k_min < k_max:
            k_common = np.logspace(np.log10(k_min), np.log10(k_max), 500)
            
            vg_interp = np.interp(k_common, k_vg, vg_species)
            ref_interp = np.interp(k_common, k_ref, ref_species)
            ratio = vg_interp / ref_interp
        else:
            k_common = k_vg
            ratio = np.ones_like(k_vg)
        
        # Plot 1: Comparison
        ax1.loglog(k_vg, vg_species, 'b-', linewidth=2, label='VG CLASS')
        ax1.loglog(k_ref, ref_species, 'r--', linewidth=2, label='CAMB Ref')
        
        ax1.set_xlabel('k [h/Mpc]', fontsize=11)
        ax1.set_ylabel(ylabel, fontsize=11)
        ax1.set_title(f'{species_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xlim(1e-3, 1e2)
        
        # Plot 2: Ratio
        ax2.semilogx(k_common, ratio, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='black', linestyle=':', alpha=0.7, linewidth=1)
        ax2.fill_between(k_common, 0.9, 1.1, color='gray', alpha=0.2, label='±10%')
        
        ax2.set_xlabel('k [h/Mpc]', fontsize=11)
        ax2.set_ylabel('VG/Ref Ratio', fontsize=11)
        ax2.set_title('Ratio Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.1, 10)
        ax2.set_xlim(1e-3, 1e2)
        ax2.set_yscale('log')
        ax2.legend(fontsize=10)
        
        # Add model info to title
        title = (f'{model_name}: {species_name.replace("_", " ").title()}\n'
                 f'n={model_info["n"]}, m={model_info["mass"]}, {model_type}, σ={model_info["sigma"]}')
        
        plt.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_path = os.path.join(species_dir, f"{species_name}_{model_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"    Created detailed plot: {species_name}")

# ==============================================================================
# Summary and statistics functions
# ==============================================================================

def calculate_statistics(vg_data, ref_data, model_info):
    """Calculate statistics for comparison."""
    # Interpolate to common k-range
    k_min = max(vg_data['k'].min(), ref_data['k'].min())
    k_max = min(vg_data['k'].max(), ref_data['k'].max())
    
    if k_min >= k_max:
        return None
    
    k_common = np.logspace(np.log10(k_min), np.log10(k_max), 1000)
    
    stats = {}
    
    # For each species
    species_list = ['T_cdm', 'T_b', 'T_tot', 'v_cdm', 'v_b']
    
    for species in species_list:
        vg_interp = np.interp(k_common, vg_data['k'], vg_data[species])
        ref_interp = np.interp(k_common, ref_data['k'], ref_data[species])
        
        # Calculate ratios
        ratio = vg_interp / ref_interp
        
        stats[f'{species}_mean_ratio'] = np.mean(ratio)
        stats[f'{species}_std_ratio'] = np.std(ratio)
        stats[f'{species}_max_ratio'] = np.max(ratio)
        stats[f'{species}_min_ratio'] = np.min(ratio)
        
        # Calculate percentage within 10%
        within_10 = np.sum(np.abs(ratio - 1) <= 0.1) / len(ratio) * 100
        stats[f'{species}_within_10%'] = within_10
    
    return stats

def create_summary_report(model_results):
    """Create a summary report of all model comparisons."""
    
    summary_file = os.path.join(OUTPUT_DIR, "comparison_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SPECIES COMPARISON SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total models processed: {len(model_results)}\n")
        f.write(f"Output directory: {OUTPUT_DIR}\n")
        f.write(f"Date: {np.datetime64('now')}\n\n")
        
        # Group by n value and type
        groups = {}
        for result in model_results:
            key = f"n={result['model_info']['n']}_{result['model_info']['type']}"
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        for group_name, results in groups.items():
            f.write(f"\n{'-' * 60}\n")
            f.write(f"GROUP: {group_name.upper()}\n")
            f.write(f"{'-' * 60}\n")
            
            for result in results:
                model = result['model_info']
                f.write(f"\nModel: {model['name']}\n")
                f.write(f"  Mass: {model['mass']}, σ: {model['sigma']}\n")
                
                if result['success']:
                    f.write(f"  Status: ✓ Success\n")
                    f.write(f"  Plot: {os.path.basename(result['plot_path'])}\n")
                    
                    if result['stats'] is not None:
                        f.write(f"  Statistics:\n")
                        for key, value in result['stats'].items():
                            if 'within_10%' in key:
                                f.write(f"    {key}: {value:.1f}%\n")
                            elif 'ratio' in key:
                                f.write(f"    {key}: {value:.3f}\n")
                else:
                    f.write(f"  Status: ✗ Failed\n")
                    f.write(f"  Error: {result['error']}\n")
    
    print(f"\nCreated summary report: {summary_file}")
    
    # Also create a CSV summary
    csv_file = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    
    with open(csv_file, 'w') as f:
        # Write header
        f.write("model_name,n,mass,sigma,type,success,plot_file,T_cdm_within_10%,T_b_within_10%,"
                "T_tot_within_10%,v_cdm_within_10%,v_b_within_10%\n")
        
        for result in model_results:
            model = result['model_info']
            success = result['success']
            plot_file = os.path.basename(result['plot_path']) if result['success'] else "N/A"
            
            # Extract statistics
            stats = result['stats'] if result['stats'] is not None else {}
            
            f.write(f"{model['name']},{model['n']},{model['mass']},{model['sigma']},{model['type']},"
                    f"{success},{plot_file},"
                    f"{stats.get('T_cdm_within_10%', 'N/A'):.1f},"
                    f"{stats.get('T_b_within_10%', 'N/A'):.1f},"
                    f"{stats.get('T_tot_within_10%', 'N/A'):.1f},"
                    f"{stats.get('v_cdm_within_10%', 'N/A'):.1f},"
                    f"{stats.get('v_b_within_10%', 'N/A'):.1f}\n")
    
    print(f"Created CSV summary: {csv_file}")

# ==============================================================================
# Main execution
# ==============================================================================

def main():
    """Main function to process and plot all models."""
    
    print("=" * 80)
    print("SPECIES COMPARISON ANALYSIS - ALL MODELS")
    print("=" * 80)
    print(f"VG CLASS directory: {VG_CLASS_DIR}")
    print(f"Reference CAMB directory: {REF_CAMB_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    model_results = []
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Processing {model['name']}...")
        
        try:
            # Load VG data
            vg_data_raw, vg_columns = load_vg_data(model['vg_prefix'], VG_CLASS_DIR)
            vg_data = process_vg_data(vg_data_raw, vg_columns)
            
            # Load reference data
            ref_file_path = os.path.join(REF_CAMB_DIR, model['camb_ref'])
            if not os.path.exists(ref_file_path):
                print(f"  ✗ Reference file not found: {ref_file_path}")
                # Try alternative naming
                alt_ref = model['camb_ref'].replace('_z99', '').replace('idm_', 'idm_n2_')
                ref_file_path = os.path.join(REF_CAMB_DIR, alt_ref)
                if os.path.exists(ref_file_path):
                    print(f"  ✓ Using alternative reference: {alt_ref}")
                else:
                    raise FileNotFoundError(f"No reference file found for {model['name']}")
            
            ref_data_raw = load_reference_data(ref_file_path)
            ref_data = process_ref_data(ref_data_raw)
            
            print(f"  ✓ Loaded data: VG k-range [{vg_data['k'].min():.2e}, {vg_data['k'].max():.2e}], "
                  f"Ref k-range [{ref_data['k'].min():.2e}, {ref_data['k'].max():.2e}]")
            
            # Create plots
            plot_path = plot_species_comparison(model, vg_data, ref_data)
            plot_detailed_species(model, vg_data, ref_data)
            
            # Calculate statistics
            stats = calculate_statistics(vg_data, ref_data, model)
            
            # Save processed data
            data_dir = os.path.join(OUTPUT_DIR, model['name'], "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Save VG data
            vg_output = os.path.join(data_dir, f"vg_processed_{model['name']}.dat")
            vg_header = "# k/h T_cdm T_b T_tot v_cdm v_b"
            np.savetxt(vg_output, np.column_stack((vg_data['k'], vg_data['T_cdm'], 
                                                  vg_data['T_b'], vg_data['T_tot'],
                                                  vg_data['v_cdm'], vg_data['v_b'])),
                      fmt='%15.6e', header=vg_header)
            
            # Save reference data
            ref_output = os.path.join(data_dir, f"ref_processed_{model['name']}.dat")
            ref_header = "# k/h T_cdm T_b T_tot v_cdm v_b"
            np.savetxt(ref_output, np.column_stack((ref_data['k'], ref_data['T_cdm'], 
                                                   ref_data['T_b'], ref_data['T_tot'],
                                                   ref_data['v_cdm'], ref_data['v_b'])),
                      fmt='%15.6e', header=ref_header)
            
            # Record results
            model_results.append({
                'model_info': model,
                'success': True,
                'plot_path': plot_path,
                'stats': stats,
                'error': None
            })
            
            print(f"  ✓ Successfully processed {model['name']}")
            
        except Exception as e:
            print(f"  ✗ Error processing {model['name']}: {e}")
            model_results.append({
                'model_info': model,
                'success': False,
                'plot_path': None,
                'stats': None,
                'error': str(e)
            })
    
    # Create summary reports
    print(f"\n{'='*80}")
    print("CREATING SUMMARY REPORTS")
    print(f"{'='*80}")
    create_summary_report(model_results)
    
    # Create overview plots
    create_overview_plots(model_results)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len([r for r in model_results if r['success']])}/{len(models)} models successfully")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nContents:")
    print(f"  - Species comparison plots (6-panel)")
    print(f"  - Detailed individual species plots")
    print(f"  - Processed data files")
    print(f"  - Summary reports (text and CSV)")
    print(f"  - Overview plots")
    print(f"{'='*80}")

def create_overview_plots(model_results):
    """Create overview plots comparing different models."""
    
    successful_models = [r for r in model_results if r['success']]
    
    if not successful_models:
        print("No successful models to create overview plots")
        return
    
    # Create overview directory
    overview_dir = os.path.join(OUTPUT_DIR, "overview")
    os.makedirs(overview_dir, exist_ok=True)
    
    # 1. Plot comparing T_tot for different n values
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Group by mass
    masses = sorted(set([m['model_info']['mass'] for m in successful_models]))
    
    for ax_idx, mass in enumerate(masses[:4]):  # Plot first 4 masses
        ax = axes[ax_idx]
        
        # Get models with this mass
        mass_models = [m for m in successful_models if m['model_info']['mass'] == mass]
        
        for model_result in mass_models:
            model = model_result['model_info']
            
            # Load data
            data_file = os.path.join(OUTPUT_DIR, model['name'], "data", 
                                    f"vg_processed_{model['name']}.dat")
            if os.path.exists(data_file):
                data = np.loadtxt(data_file)
                k = data[:, 0]
                T_tot = data[:, 3]**2  # Squared transfer function
                
                label = f"n={model['n']}, {model['type']}, σ={model['sigma']}"
                ax.loglog(k, T_tot, label=label, linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('k [h/Mpc]', fontsize=11)
        ax.set_ylabel('$T_{tot}^2(k)$', fontsize=11)
        ax.set_title(f'Mass = {mass}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1e-3, 1e2)
        ax.legend(fontsize=8, loc='best')
    
    plt.suptitle('Total Transfer Function Comparison by Mass', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(overview_dir, "overview_T_tot_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Plot statistics comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    statistics = ['T_cdm_within_10%', 'T_b_within_10%', 'T_tot_within_10%', 
                  'v_cdm_within_10%', 'v_b_within_10%']
    
    for ax_idx, stat in enumerate(statistics[:5]):
        ax = axes[ax_idx]
        
        # Extract data
        x_labels = []
        values = []
        
        for model_result in successful_models:
            if model_result['stats'] and stat in model_result['stats']:
                model = model_result['model_info']
                x_labels.append(f"n={model['n']}\n{model['type'][:1]}")
                values.append(model_result['stats'][stat])
        
        if values:
            bars = ax.bar(range(len(values)), values, color='skyblue', alpha=0.7)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_ylabel('Percentage within 10%', fontsize=10)
            ax.set_title(f'{stat.replace("_", " ").replace("within 10%", "")}', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Model Comparison Statistics (Percentage within 10% of Reference)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(overview_dir, "overview_statistics_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created overview plots in {overview_dir}")

if __name__ == "__main__":
    main()
