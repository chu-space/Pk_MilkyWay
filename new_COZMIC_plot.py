import matplotlib.pyplot as plt
import numpy as np
import glob
import os

DATA_DIRECTORY = "fixed_T2_grouped_CLASS_runs_VG_237_k300"
OUTPUT_PLOT_DIR = "fixed_T2_grouped_plots_middle_sigma"

HALF_MODE_THRESHOLD = 0.25

omega_mh2 = 0.11711; h = 0.7
a = 0.0437; b = -1.188; nu = 1.049; theta = 2.012; eta = 0.2463

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

def transfer(k,mwdm):
    """New WDM Transfer function - k in h/Mpc"""
    alpha = a*(mwdm**b)*((omega_mh2/0.12)**eta)*((h/0.6736)**theta)
    transfer = (1+(alpha*k)**(2*nu))**(-5./nu)
    return transfer

def T2_wdm_old(k,mwdm):
    """Old WDM Transfer function - k in h/Mpc"""
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

def parse_cross_section(s_str):
    """Parse cross-section string like s2.2em2700 to 2.2e-27 cm²."""
    if s_str.startswith('s'):
        s_str = s_str[1:]
    
    # Parse formats like: s2.2em2700 -> 2.2e-27 (ignore last two digits)
    if 'em' in s_str:
        try:
            # Split at 'em'
            parts = s_str.split('em', 1)
            base = float(parts[0])
            
            # The part after 'em' has the exponent
            exp_part = parts[1]
            
            # First two digits are the exponent
            if len(exp_part) >= 2:
                # Take first two digits for exponent
                exponent = -int(exp_part[:2])
                return base * 10**exponent
        except:
            return None
    
    return None

def extract_params_from_filename(filename):
    """Extract parameters directly from filename."""
    # Remove extension
    name = filename.replace('_pk.dat', '')
    
    # Initialize default values
    n = None
    m_GeV = None
    sigma_cm2 = None
    
    # Split by underscores
    parts = name.split('_')
    
    for part in parts:
        # Extract n value
        if part.startswith('n'):
            try:
                n_str = part[1:]
                # Get all digits
                digits = ''
                for char in n_str:
                    if char.isdigit():
                        digits += char
                    else:
                        break
                if digits:
                    n = int(digits)
            except:
                n = None
        
        # Extract mass in GeV
        elif part.startswith('m'):
            try:
                m_str = part[1:]
                
                # Handle m1em04 = 1 × 10⁻⁴ GeV
                if 'em' in m_str:
                    base, exp_part = m_str.split('em', 1)
                    base_val = float(base)
                    if len(exp_part) >= 2:
                        exponent = -int(exp_part[:2])
                        m_GeV = base_val * 10**exponent
                
                # Handle m1ep00 = 1 × 10⁰ = 1 GeV  
                elif 'ep' in m_str:
                    base, exp_part = m_str.split('ep', 1)
                    base_val = float(base)
                    if len(exp_part) >= 2:
                        exponent = int(exp_part[:2])
                        m_GeV = base_val * 10**exponent
                    else:
                        m_GeV = base_val  # m1ep00 -> 1 GeV
            except:
                m_GeV = None
        
        # Extract cross-section
        elif part.startswith('s'):
            try:
                sigma_cm2 = parse_cross_section(part)
            except:
                sigma_cm2 = None
    
    return n, m_GeV, sigma_cm2

def format_cross_section(sigma_cm2):
    """Format cross-section for display in scientific notation (3 sf)."""
    if sigma_cm2 is None:
        return "N/A"
    
    # Format in scientific notation with 3 significant figures
    if sigma_cm2 == 0:
        return "0"
    
    # Determine the exponent
    exponent = int(np.floor(np.log10(abs(sigma_cm2))))
    mantissa = sigma_cm2 / 10**exponent
    
    # Format mantissa with 3 significant figures
    if abs(mantissa) >= 10:
        mantissa_str = f"{mantissa:.2f}"
    elif abs(mantissa) >= 1:
        mantissa_str = f"{mantissa:.2f}"
    else:
        mantissa_str = f"{mantissa:.2f}"
    
    return f"{mantissa_str}×10^{{{exponent}}} cm²"

def format_mass(mass_GeV):
    """Format mass for display."""
    if mass_GeV is None:
        return "N/A"
    
    if abs(mass_GeV - 1e-4) < 1e-5:
        return "10⁻⁴ GeV"
    elif abs(mass_GeV - 1e-2) < 1e-3:
        return "10⁻² GeV"
    elif abs(mass_GeV - 1) < 0.1:
        return "1 GeV"
    else:
        # Format in scientific notation
        exponent = int(np.floor(np.log10(abs(mass_GeV))))
        mantissa = mass_GeV / 10**exponent
        return f"{mantissa:.2f}×10^{{{exponent}}} GeV"

def find_matching_files(n_value, mass_GeV):
    """Find files matching specific n and mass."""
    print(f"  Looking for n={n_value}, m={format_mass(mass_GeV)}")
    
    # Get all pk.dat files
    all_files = glob.glob(os.path.join(DATA_DIRECTORY, "*_pk.dat"))
    matching_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Skip CDM files
        if 'cdm_baseline' in filename:
            continue
        
        # Extract parameters
        n, m, sigma = extract_params_from_filename(filename)
        
        # Debug: print what we found
        if n == n_value:
            print(f"    Checking {filename}: n={n}, m={m} GeV, σ={sigma}")
        
        # Check if they match
        if n == n_value:
            # For mass comparison, allow some tolerance
            if m is not None and abs(m - mass_GeV) < 0.01 * max(abs(m), abs(mass_GeV)):
                matching_files.append((file_path, sigma))
                print(f"    ✓ Matched: {filename}")
                print(f"      m={format_mass(m)}, σ={format_cross_section(sigma)}")
    
    return matching_files

def calculate_half_mode(k_Mpc, T2):
    """Calculate half-mode wavenumber from transfer function."""
    if len(k_Mpc) == 0 or len(T2) == 0:
        return 0.0
    
    # Find where T2 crosses HALF_MODE_THRESHOLD
    for i in range(len(T2)-1):
        if T2[i] >= HALF_MODE_THRESHOLD and T2[i+1] < HALF_MODE_THRESHOLD:
            # Linear interpolation in log space
            k1, k2 = k_Mpc[i], k_Mpc[i+1]
            T1, T2_val = T2[i], T2[i+1]
            
            # Use linear interpolation in log space for better accuracy
            log_k_half = np.log(k1) + (np.log(k2) - np.log(k1)) * (np.log(HALF_MODE_THRESHOLD) - np.log(T1)) / (np.log(T2_val) - np.log(T1))
            k_half = np.exp(log_k_half)
            return k_half
    
    # If no crossing found, find closest point
    idx = np.argmin(np.abs(T2 - HALF_MODE_THRESHOLD))
    return k_Mpc[idx]

def load_model(file_path, cdm_k_h, cdm_P):
    """Load a single CLASS model."""
    try:
        filename = os.path.basename(file_path)
        
        # Extract parameters
        n, m_GeV, sigma_cm2 = extract_params_from_filename(filename)
        
        # Load IDM data
        k_idm_h, P_idm = np.loadtxt(file_path, unpack=True)
        
        # Convert k from h/Mpc to Mpc⁻¹
        k_cdm_Mpc = cdm_k_h * h
        k_idm_Mpc = k_idm_h * h
        
        # Interpolate IDM P(k) to CDM k grid
        P_idm_interp = np.interp(k_cdm_Mpc, k_idm_Mpc, P_idm)
        
        # Calculate T² = P_IDM / P_CDM
        P_cdm_safe = np.where(cdm_P > 0, cdm_P, 1e-100)
        T2 = P_idm_interp / P_cdm_safe
        
        # Find half-mode
        k_half_Mpc = calculate_half_mode(k_cdm_Mpc, T2)
        
        # Check T² at small k (should be ~1)
        if len(T2) > 10:
            T2_k0 = np.mean(T2[:min(10, len(T2)//10)])
        else:
            T2_k0 = T2[0] if len(T2) > 0 else 0
        
        return {
            'filename': filename,
            'n': n,
            'm_GeV': m_GeV,
            'sigma_cm2': sigma_cm2,
            'k_Mpc': k_cdm_Mpc,
            'T2': T2,
            'half_mode_Mpc': k_half_Mpc,
            'T2_k0': T2_k0
        }
        
    except Exception as e:
        print(f"    Error loading {os.path.basename(file_path)}: {e}")
        return None

def find_closest_model(models, target_sigma):
    """Find the model with sigma closest to target_sigma."""
    if not models:
        return None, float('inf')
    
    closest_model = None
    min_diff = float('inf')
    
    for model in models:
        if model['sigma_cm2'] is None:
            continue
            
        diff = abs(model['sigma_cm2'] - target_sigma) / target_sigma
        if diff < min_diff:
            min_diff = diff
            closest_model = model
    
    return closest_model, min_diff

def create_plot_for_cozmic_model(cozmic_model, output_dir):
    """Create a separate plot for each COZMIC model (n, mass)."""
    n_val, mass_GeV, sigma_halfmode, sigma_5_9keV, sigma_envelope = cozmic_model
    
    print(f"\n{'='*70}")
    print(f"Creating plot for n={n_val}, m={format_mass(mass_GeV)}")
    print(f"Target cross-sections: half-mode={sigma_halfmode:.2e}, σ_5.9keV={sigma_5_9keV:.2e}, envelope={sigma_envelope:.2e}")
    print('='*70)
    
    # ==================== LOAD CDM BASELINE ====================
    print("\n1. Loading CDM baseline...")
    cdm_file = os.path.join(DATA_DIRECTORY, "cdm_baseline00_pk.dat")
    if not os.path.exists(cdm_file):
        print(f"❌ CDM file not found: {cdm_file}")
        return False
    
    # Load CDM (k in h/Mpc, P(k) in (Mpc/h)³ or similar)
    k_cdm_h, P_cdm = np.loadtxt(cdm_file, unpack=True)
    
    # Convert k from h/Mpc to Mpc⁻¹ (for plotting)
    k_cdm_Mpc = k_cdm_h * h
    
    print(f"   ✓ CDM loaded: {len(k_cdm_h)} points")
    print(f"     k range: {k_cdm_h[0]:.3f}-{k_cdm_h[-1]:.1f} h/Mpc")
    print(f"     k range: {k_cdm_Mpc[0]:.3f}-{k_cdm_Mpc[-1]:.1f} Mpc⁻¹")
    
    # ==================== LOAD ALL CLASS MODELS FOR THIS (n, mass) ====================
    print(f"\n2. Finding CLASS models for n={n_val}, m={format_mass(mass_GeV)}...")
    
    matching_files = find_matching_files(n_val, mass_GeV)
    
    if not matching_files:
        print(f"   ❌ No matching files found")
        return False
    
    print(f"   ✓ Found {len(matching_files)} matching files")
    
    # Load all models
    all_models = []
    for file_path, sigma in matching_files:
        model = load_model(file_path, k_cdm_h, P_cdm)
        if model:
            all_models.append(model)
    
    if not all_models:
        print(f"   ❌ No models loaded successfully")
        return False
    
    print(f"   ✓ Loaded {len(all_models)} models")
    
    # Sort by cross-section
    all_models_sorted = sorted([m for m in all_models if m['sigma_cm2'] is not None], 
                               key=lambda x: x['sigma_cm2'])
    
    if not all_models_sorted:
        print(f"   ❌ No models with valid cross-sections")
        return False
    
    # Print cross-section range
    print(f"   σ range: {all_models_sorted[0]['sigma_cm2']:.2e} - {all_models_sorted[-1]['sigma_cm2']:.2e} cm²")
    
    # ==================== FIND COZMIC REFERENCE MODELS ====================
    print(f"\n3. Identifying COZMIC reference models...")
    
    reference_models = []
    
    # Find half-mode model (smallest sigma)
    print(f"   Looking for half-mode model (σ≈{sigma_halfmode:.2e} cm²)...")
    halfmode_model, halfmode_diff = find_closest_model(all_models_sorted, sigma_halfmode)
    if halfmode_model and halfmode_diff < 0.5:  # Within 50%
        print(f"     ✓ Half-mode: σ={halfmode_model['sigma_cm2']:.2e} cm² (diff={halfmode_diff*100:.1f}%)")
        print(f"        k_half={halfmode_model['half_mode_Mpc']:.1f} Mpc⁻¹, T²(k→0)={halfmode_model['T2_k0']:.3f}")
        halfmode_model['type'] = 'halfmode'
        halfmode_model['color'] = 'darkred'
        halfmode_model['label'] = f'Half-mode: σ={format_cross_section(halfmode_model["sigma_cm2"])}'
        reference_models.append(halfmode_model)
    else:
        print(f"     ⚠️ No close match for half-mode")
    
    # Find σ_5.9keV (middle) model
    print(f"   Looking for σ_5.9keV model (σ≈{sigma_5_9keV:.2e} cm²)...")
    middle_model, middle_diff = find_closest_model(all_models_sorted, sigma_5_9keV)
    if middle_model and middle_diff < 0.5:  # Within 50%
        print(f"     ✓ σ_5.9keV: σ={middle_model['sigma_cm2']:.2e} cm² (diff={middle_diff*100:.1f}%)")
        print(f"        k_half={middle_model['half_mode_Mpc']:.1f} Mpc⁻¹, T²(k→0)={middle_model['T2_k0']:.3f}")
        middle_model['type'] = 'middle'
        middle_model['color'] = 'darkorange'
        middle_model['label'] = f'σ_5.9keV: σ={format_cross_section(middle_model["sigma_cm2"])}'
        reference_models.append(middle_model)
    else:
        print(f"     ⚠️ No close match for σ_5.9keV")
    
    # Find envelope model (largest sigma)
    print(f"   Looking for envelope model (σ≈{sigma_envelope:.2e} cm²)...")
    envelope_model, envelope_diff = find_closest_model(all_models_sorted, sigma_envelope)
    if envelope_model and envelope_diff < 0.5:  # Within 50%
        print(f"     ✓ Envelope: σ={envelope_model['sigma_cm2']:.2e} cm² (diff={envelope_diff*100:.1f}%)")
        print(f"        k_half={envelope_model['half_mode_Mpc']:.1f} Mpc⁻¹, T²(k→0)={envelope_model['T2_k0']:.3f}")
        envelope_model['type'] = 'envelope'
        envelope_model['color'] = 'forestgreen'
        envelope_model['label'] = f'Envelope: σ={format_cross_section(envelope_model["sigma_cm2"])}'
        reference_models.append(envelope_model)
    else:
        print(f"     ⚠️ No close match for envelope")
    
    print(f"   ✓ Identified {len(reference_models)} reference models")
    
    # ==================== CREATE PLOT ====================
    print("\n4. Creating plot...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # ==================== PLOT WDM CONSTRAINT LINES ====================
    print("   Plotting WDM constraint lines...")
    
    # Generate k values in Mpc⁻¹
    kvec_Mpc = np.logspace(0, np.log10(500), 1000)
    
    # Calculate WDM half-modes
    k_half_59_Mpc, k_half_59_hMpc = calculate_wdm_half_mode(5.9, formula='new')
    k_half_65_Mpc, k_half_65_hMpc = calculate_wdm_half_mode(6.5, formula='old')
    
    print(f"     WDM 5.9 keV: k_half={k_half_59_Mpc:.1f} Mpc⁻¹ = {k_half_59_hMpc:.1f} h/Mpc")
    print(f"     WDM 6.5 keV: k_half={k_half_65_Mpc:.1f} Mpc⁻¹ = {k_half_65_hMpc:.1f} h/Mpc")
    
    # Convert k to h/Mpc for WDM functions
    kvec_hMpc = kvec_Mpc / h
    
    # Calculate WDM transfer functions
    T2_new = transfer(kvec_hMpc, 5.9)**2
    T2_old = T2_wdm_old(kvec_hMpc, 6.5)
    
    # Plot WDM lines with k in Mpc⁻¹
    ax.plot(kvec_Mpc, T2_new, linestyle='--', color='purple', 
            linewidth=3.0, alpha=0.9, label=f"WDM: 5.9 keV\nk_half={k_half_59_Mpc:.1f} Mpc⁻¹")
    ax.plot(kvec_Mpc, T2_old, linestyle='-.', color='darkblue', 
            linewidth=3.0, alpha=0.9, label=f"WDM: 6.5 keV\nk_half={k_half_65_Mpc:.1f} Mpc⁻¹")
    
    # Add vertical lines at WDM half-modes
    ax.axvline(x=k_half_59_Mpc, color='purple', linestyle=':', alpha=0.3, linewidth=1.5)
    ax.axvline(x=k_half_65_Mpc, color='darkblue', linestyle=':', alpha=0.3, linewidth=1.5)
    
    # ==================== PLOT ALL CLASS MODELS ====================
    print(f"   Plotting {len(all_models)} CLASS models...")
    
    # Plot all models as thin gray lines
    for model in all_models:
        if model['sigma_cm2'] is None:
            continue
            
        # Skip if this is a reference model (will be plotted separately)
        if any(ref['filename'] == model['filename'] for ref in reference_models):
            continue
            
        ax.plot(model['k_Mpc'], model['T2'],
                color='gray',
                linestyle='-',
                linewidth=0.5,
                alpha=0.3,
                label='_nolegend_')
    
    # ==================== PLOT REFERENCE MODELS ====================
    print("   Plotting reference models...")
    
    for model in reference_models:
        linestyle = '-'
        linewidth = 2.5
        
        if model['type'] == 'envelope':
            linestyle = ':'  # DOTTED for envelope
        elif model['type'] == 'halfmode':
            linestyle = '--'  # DASHED for half-mode
        elif model['type'] == 'middle':
            linestyle = '-'  # SOLID for middle/5.9keV
            linewidth = 3.0  # Thicker for emphasis
        
        # Plot with k in Mpc⁻¹
        label = f"{model['label']}\nk_half={model['half_mode_Mpc']:.1f} Mpc⁻¹"
        
        ax.plot(model['k_Mpc'], model['T2'],
                color=model['color'],
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.9,
                label=label)
        
        # Add marker at half-mode
        ax.plot(model['half_mode_Mpc'], HALF_MODE_THRESHOLD,
                marker='o', color=model['color'], markersize=8, alpha=0.8,
                label='_nolegend_')
    
    # ==================== ADD REFERENCE LINES ====================
    
    ax.axhline(y=HALF_MODE_THRESHOLD, color='gray', linestyle=':', 
               linewidth=2.0, alpha=0.6, label=f'Half-mode: T²={HALF_MODE_THRESHOLD}')
    
    ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, linewidth=1.5, label='CDM (T²=1)')
    
    # ==================== FORMAT PLOT ====================
    
    ax.set_xscale("log")
    ax.set_ylabel(r"$T^2(k) = P_{\mathrm{IDM}}(k) / P_{\mathrm{CDM}}(k)$", 
                  fontsize=16, labelpad=10)
    ax.set_xlabel(r"Wavenumber $k$ [Mpc$^{-1}$]", fontsize=16, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_xlim(1, 500)
    ax.set_ylim(0, 1.1)
    
    ax.grid(True, which='major', linestyle='-', alpha=0.1)
    ax.grid(True, which='minor', linestyle=':', alpha=0.05)
    
    # ==================== CREATE LEGEND ====================
    
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_handles = []
    
    # WDM constraints
    legend_handles.append(Patch(facecolor='white', edgecolor='none', 
                               label='=== WDM CONSTRAINTS ==='))
    legend_handles.append(Line2D([0], [0], color='purple', linestyle='--', 
                                 linewidth=3.0, label=f'5.9 keV WDM\nk_half={k_half_59_Mpc:.1f} Mpc⁻¹'))
    legend_handles.append(Line2D([0], [0], color='darkblue', linestyle='-.', 
                                 linewidth=3.0, label=f'6.5 keV WDM\nk_half={k_half_65_Mpc:.1f} Mpc⁻¹'))
    
    legend_handles.append(Patch(facecolor='white', edgecolor='none', label=''))
    
    # Reference models
    if reference_models:
        legend_handles.append(Patch(facecolor='white', edgecolor='none', 
                                   label='=== COZMIC REFERENCE MODELS ==='))
        
        for model in reference_models:
            if model['type'] == 'envelope':
                linestyle = ':'
            elif model['type'] == 'halfmode':
                linestyle = '--'
            else:
                linestyle = '-'
            
            linewidth = 3.0 if model['type'] == 'middle' else 2.5
            
            legend_handles.append(Line2D([0], [0], color=model['color'], linestyle=linestyle, 
                                         linewidth=linewidth, 
                                         label=f'{model["type"]}: σ={format_cross_section(model["sigma_cm2"])}\nk_half={model["half_mode_Mpc"]:.1f} Mpc⁻¹'))
    
    legend_handles.append(Patch(facecolor='white', edgecolor='none', label=''))
    
    # Other CLASS models
    legend_handles.append(Patch(facecolor='white', edgecolor='none', 
                               label='=== OTHER CLASS MODELS ==='))
    
    min_model = all_models_sorted[0]
    max_model = all_models_sorted[-1]
    
    if len(all_models_sorted) > 1:
        label = f'{len(all_models)} models total\nσ={format_cross_section(min_model["sigma_cm2"])} - {format_cross_section(max_model["sigma_cm2"])}'
    else:
        label = f'{len(all_models)} model\nσ={format_cross_section(min_model["sigma_cm2"])}'
    
    legend_handles.append(Line2D([0], [0], color='gray', linestyle='-', 
                                 linewidth=1.0, label=label))
    
    legend_handles.append(Patch(facecolor='white', edgecolor='none', label=''))
    
    # Reference lines
    legend_handles.append(Line2D([0], [0], color='gray', linestyle=':', 
                                 linewidth=2.0, label=f'Half-mode: T²={HALF_MODE_THRESHOLD}'))
    legend_handles.append(Line2D([0], [0], color='black', linestyle='-', 
                                 linewidth=1.5, alpha=0.3, label='CDM (T²=1)'))
    
    # Create legend
    legend = ax.legend(handles=legend_handles, fontsize=8, loc='lower left', 
                      framealpha=0.95, ncol=2, frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)
    
    # ==================== ADD TITLE AND INFO ====================
    
    title_text = f"Transfer Functions: n={n_val}, m={format_mass(mass_GeV)}\n"
    title_text += f"CLASS Models with COZMIC Reference Cross-sections"
    
    ax.set_title(title_text, fontsize=16, pad=20, weight='bold')
    
    # Add info text
    info_text = f"h = {h}, T²_half = {HALF_MODE_THRESHOLD}\n"
    info_text += f"All k in Mpc⁻¹ (converted from h/Mpc)\n"
    info_text += f"Total CLASS models: {len(all_models)}\n"
    info_text += f"Reference models found: {len(reference_models)}/3\n"
    info_text += f"WDM 5.9 keV: k_half={k_half_59_Mpc:.1f} Mpc⁻¹\n"
    info_text += f"WDM 6.5 keV: k_half={k_half_65_Mpc:.1f} Mpc⁻¹"
    
    ax.text(0.98, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    if mass_GeV == 1e-4:
        mass_str = "1e-04GeV"
    elif mass_GeV == 1e-2:
        mass_str = "1e-02GeV"
    elif mass_GeV == 1:
        mass_str = "1GeV"
    else:
        mass_str = f"{mass_GeV:.2e}GeV"
    
    output_file = os.path.join(output_dir, f"transfer_functions_n{n_val}_m{mass_str}_middle.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved: {output_file}")
    
    return True

def main():
    """Main function - create separate plots for each COZMIC model."""
    print("\n" + "="*70)
    print("TRANSFER FUNCTION COMPARISON with COZMIC Reference Cross-sections")
    print("="*70)
    print(f"CLASS data: {DATA_DIRECTORY}")
    print(f"COZMIC models: {len(cozmic_models)} total")
    print(f"Output: {OUTPUT_PLOT_DIR}")
    print("="*70)
    
    # Check directories
    if not os.path.exists(DATA_DIRECTORY):
        print(f"❌ CLASS data directory not found: {DATA_DIRECTORY}")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    
    # First test file parsing with examples
    print("\nTesting file parsing with examples:")
    test_files = [
        "idm_run_n4_m1em04_s2.2em2700_pk.dat",
        "idm_run_n4_m1em02_s1.7em1900_pk.dat", 
        "idm_run_n4_m1ep00_s2.1em1700_pk.dat",
        "idm_run_n2_m1em04_s1.2em2700_pk.dat",
        "idm_run_n2_m1em02_s1.3em2500_pk.dat",
        "idm_run_n2_m1ep00_s1.4em2200_pk.dat"
    ]
    
    for test_file in test_files:
        n, m, sigma = extract_params_from_filename(test_file)
        print(f"  {test_file}:")
        print(f"    n={n}, m={format_mass(m)} (raw: {m}), σ={format_cross_section(sigma)} (raw: {sigma})")
    
    # Create plots for each COZMIC model
    successful_plots = 0
    for i, cozmic_model in enumerate(cozmic_models):
        n_val, mass_GeV, _, sigma_5_9keV, _ = cozmic_model
        
        print(f"\n{'='*70}")
        print(f"Processing model {i+1}/{len(cozmic_models)}: n={n_val}, m={format_mass(mass_GeV)}")
        print(f"Looking for σ_5.9keV ≈ {sigma_5_9keV:.2e} cm²")
        print('='*70)
        
        success = create_plot_for_cozmic_model(cozmic_model, OUTPUT_PLOT_DIR)
        
        if success:
            successful_plots += 1
            print(f"✓ Created plot for n={n_val}, m={format_mass(mass_GeV)}")
        else:
            print(f"✗ Failed to create plot for n={n_val}, m={format_mass(mass_GeV)}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Created {successful_plots}/{len(cozmic_models)} plots")
    print(f"Output directory: {OUTPUT_PLOT_DIR}")
    
    print("\nCOZMIC models processed:")
    for i, model in enumerate(cozmic_models):
        n_val, mass_GeV, _, sigma_5_9keV, _ = model
        print(f"  {i+1}. n={n_val}, m={format_mass(mass_GeV)}, σ_5.9keV={sigma_5_9keV:.2e} cm²")
    
    print("\nKey features:")
    print("  1. Correct file parsing: m1ep00 = 1 GeV, s2.2em2700 = 2.2e-27 cm²")
    print("  2. Cross-sections formatted in scientific notation (3 sf)")
    print("  3. All k values converted to Mpc⁻¹ for consistent plotting")
    print("  4. WDM constraints: 5.9 keV and 6.5 keV")
    print("  5. Separate plots for each (n, mass) combination")
    print("="*70)

if __name__ == "__main__":
    main()