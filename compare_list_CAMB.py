import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Set up the plot grid - we need 6 plots for the envelope/halfmode models
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
axes = axes.flatten()

# Dictionary to map models for comparison - using actual file names from your directory
models_to_compare = {
    0: {
        'name': 'n=2, 1e-4 GeV, halfmode',
        'camb_ref': '/home/arifchu/Pk_MilkyWay/camb_data_tk/idm_n2_1e-4GeV_halfmode_z99_Tk.dat',
        'camb_vg': '/home/arifchu/class_public-master-new-dmeff/VG_fixed_correct_middle_camb/processed_Tk_n2_m1e-4_s4.2e-28.dat',
    },
    1: {
        'name': 'n=2, 1e-2 GeV, envelope',
        'camb_ref': '/home/arifchu/Pk_MilkyWay/camb_data_tk/idm_n2_1e-2GeV_envelope_z99_Tk.dat',
        'camb_vg': '/home/arifchu/class_public-master-new-dmeff/VG_fixed_correct_middle_camb/processed_Tk_n2_m0.01_s7.1e-24.dat',
    },
    2: {
        'name': 'n=2, 1 GeV, halfmode',
        'camb_ref': '/home/arifchu/Pk_MilkyWay/camb_data_tk/idm_n2_1GeV_halfmode_z99_Tk.dat',
        'camb_vg': '/home/arifchu/class_public-master-new-dmeff/VG_fixed_correct_middle_camb/processed_Tk_n2_m1_s1.6e-23.dat',
    },
    3: {
        'name': 'n=4, 1e-4 GeV, halfmode',
        'camb_ref': '/home/arifchu/Pk_MilkyWay/camb_data_tk/idm_1e-4GeV_halfmode_Tk.dat',
        'camb_vg': '/home/arifchu/class_public-master-new-dmeff/VG_fixed_correct_middle_camb/processed_Tk_n4_m1e-4_s2.2e-27.dat',
    },
    4: {
        'name': 'n=4, 1e-2 GeV, halfmode',
        'camb_ref': '/home/arifchu/Pk_MilkyWay/camb_data_tk/idm_1e-2GeV_halfmode_Tk.dat',  # Note: this might not exist
        'camb_vg': '/home/arifchu/class_public-master-new-dmeff/VG_fixed_correct_middle_camb/processed_Tk_n4_m0.01_s1.7e-22.dat',
    },
    5: {
        'name': 'n=4, 1 GeV, halfmode',
        'camb_ref': '/home/arifchu/Pk_MilkyWay/camb_data_tk/idm_1GeV_halfmode_Tk.dat',
        'camb_vg': '/home/arifchu/class_public-master-new-dmeff/VG_fixed_correct_middle_camb/processed_Tk_n4_m1_s8.6e-19.dat',
    }
}

# Let's also check for other envelope/halfmode models from your list
# that might not be in the original dictionary but have files
additional_models = []

# Check which models actually have both files
available_models = {}
model_counter = 0

for idx, model_info in models_to_compare.items():
    ref_exists = os.path.exists(model_info['camb_ref'])
    vg_exists = os.path.exists(model_info['camb_vg'])
    
    if ref_exists and vg_exists:
        available_models[model_counter] = model_info
        model_counter += 1
        print(f"Model '{model_info['name']}' has both files.")
    else:
        missing_files = []
        if not ref_exists:
            missing_files.append(f"ref: {model_info['camb_ref']}")
        if not vg_exists:
            missing_files.append(f"vg: {model_info['camb_vg']}")
        print(f"Model '{model_info['name']}' missing: {', '.join(missing_files)}")

# Plot only the available models
print(f"\nPlotting {len(available_models)} available models")

for idx, model in enumerate(available_models.values()):
    if idx >= len(axes):
        break
        
    ax = axes[idx]
    
    try:
        # Load COZMIC reference CAMB data
        data_ref = np.loadtxt(model['camb_ref'])
        k_ref = data_ref[:, 0]
        tk_ref = data_ref[:, 1]
        ax.loglog(k_ref, tk_ref**2, color='black', linewidth=2, label='COZMIC CAMB Ref')
        print(f"Loaded reference: {model['camb_ref']}")
    except Exception as e:
        print(f"Error loading {model['camb_ref']}: {e}")
        continue
    
    try:
        # Load VG CAMB data
        data_vg = np.loadtxt(model['camb_vg'])
        k_vg = data_vg[:, 0]
        tk_vg = data_vg[:, 1]
        ax.loglog(k_vg, tk_vg**2, color='blue', linestyle='--', linewidth=1.5, label='VG CAMB')
    except Exception as e:
        print(f"Error loading {model['camb_vg']}: {e}")
    
    ax.set_xlabel('k [h/Mpc]', fontsize=10)
    ax.set_ylabel('$T_{dm}^2(k)$', fontsize=10)
    ax.set_title(model['name'], fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Set consistent axis limits
    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(1e-8, 1e2)

# Remove any extra axes if we have fewer than 6 models
for i in range(len(available_models), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('idm_model_comparisons_envelope_halfmode.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'idm_model_comparisons_envelope_halfmode.png'")

# Check if we should create the detailed comparison
if len(available_models) > 0:
    print("\n" + "="*60)
    print("DETAILED COMPARISON FOR FIRST AVAILABLE MODEL")
    print("="*60)

    # Load all available data for the first model
    specific_model = list(available_models.values())[0]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    try:
        # Load reference CAMB
        data_ref = np.loadtxt(specific_model['camb_ref'])
        k_ref = data_ref[:, 0]
        tk_ref = data_ref[:, 1]
        ax1.loglog(k_ref, tk_ref**2, 'k-', linewidth=2, label='COZMIC CAMB Ref')
        ax2.semilogx(k_ref, np.ones_like(k_ref), 'k-', linewidth=2, label='Reference')
        
        # Load VG CAMB
        data_vg = np.loadtxt(specific_model['camb_vg'])
        k_vg = data_vg[:, 0]
        tk_vg = data_vg[:, 1]
        ax1.loglog(k_vg, tk_vg**2, 'b--', linewidth=1.5, label='VG CAMB')
        
        # Interpolate to common k-range for ratio plot
        k_min = max(k_ref.min(), k_vg.min())
        k_max = min(k_ref.max(), k_vg.max())
        k_common = np.logspace(np.log10(k_min), np.log10(k_max), 500)
        
        tk_ref_interp = np.interp(k_common, k_ref, tk_ref**2)
        tk_vg_interp = np.interp(k_common, k_vg, tk_vg**2)
        ratio_vg = tk_vg_interp / tk_ref_interp
        
        ax2.semilogx(k_common, ratio_vg, 'b--', linewidth=1.5, label='VG CAMB/Ref')
        
    except Exception as e:
        print(f"Error in detailed comparison: {e}")

    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('$T_{dm}^2(k)$')
    ax1.set_title(f'Transfer Function Comparison: {specific_model["name"]}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel('k [h/Mpc]')
    ax2.set_ylabel('Ratio to Reference')
    ax2.set_title('Ratio Plot')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0.1, 10)

    plt.tight_layout()
    detailed_filename = f'detailed_comparison_{specific_model["name"].replace(", ", "_").replace(" ", "_")}.png'
    plt.savefig(detailed_filename, dpi=150, bbox_inches='tight')
    print(f"Detailed plot saved as '{detailed_filename}'")
else:
    print("No models with both files available for plotting.")