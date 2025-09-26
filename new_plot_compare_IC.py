import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd

# Set the plot style for a "cosmic" look
plt.style.use('dark_background')
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "font.size": 16,
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "legend.edgecolor": "none",
    "legend.facecolor": (0.2, 0.2, 0.2, 0.6), # <-- The fix is here
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "grid.color": "#444444"
})

def load_data_with_dynamic_headers(filepath):
    """
    Loads data from a CLASS/CAMB output file into a pandas DataFrame.
    Dynamically finds and parses the header to name the columns.
    """
    header_line = None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if line.strip().startswith('#') and '1:' in line:
                header_line = line
                break
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return pd.DataFrame()

    if not header_line:
        print(f"Warning: Could not find header in {filepath}. Loading without column names.")
        return pd.DataFrame(np.loadtxt(filepath))

    matches = re.findall(r'\d+:\s*(.*?)(?=\s+\d+:|$)', header_line.strip().lstrip('#').strip())

    column_names = []
    for match in matches:
        raw_name = match.strip()
        # Specific name mapping
        if raw_name == '-T_cdm/k2': clean_name = 'cdm_t'
        elif raw_name == '-T_b/k2': clean_name = 'baryon_t'
        elif raw_name == '-T_tot/k2': clean_name = 'total_t'
        else:
            # General cleaning
            clean_name = re.sub(r'[\(\[].*?[\)\]]', '', raw_name).strip()
            clean_name = clean_name.replace(' ', '_').replace('-', '_').replace('.', '')
        column_names.append(clean_name)

    df = pd.read_csv(
        filepath,
        comment='#',
        delim_whitespace=True,
        header=None,
        names=column_names
    )
    return df

# --- 1. Load Data ---
try:
    data = np.loadtxt("corrected_new_camb_n_4_idm_1GeV_halfmode_Tk.dat")
    data2 = np.loadtxt("data_tk/idm_1e-4GeV_halfmode_Tk.dat")
    data3 = load_data_with_dynamic_headers("output/cdm_for_camb00_tk.dat")
except FileNotFoundError as e:
    print(f"Error: Could not find a data file. {e}")
    exit()

# --- 2. Unpack and Prepare Data ---
# Extract k-values and transfer functions
k = data[:, 0]
T_cdm = data[:, 1]

k2 = data2[:, 0]
T_cdm2 = data2[:, 1]

# Ensure data3 has the required columns before proceeding
if 'k' not in data3.columns or 'cdm_t' not in data3.columns:
    print("Error: The baseline file 'cdm_for_camb00_tk.dat' is missing 'k' or 'cdm_t' columns.")
    exit()

T_cdm_base = data3['cdm_t'].values

# --- 3. Interpolate for Ratio Calculation ---
# Interpolate the baseline CDM transfer function onto the k-grids of the other datasets
T_cdm_base_interp1 = np.interp(k, data3['k'].values, T_cdm_base)
T_cdm_base_interp2 = np.interp(k2, data3['k'].values, T_cdm_base)

# --- 4. Plotting ---
fig, ax = plt.subplots(figsize=(12, 8))

# Define vibrant colors for the plot
color1 = '#08F7FE'  # Bright Cyan
color2 = '#FE53BB'  # Bright Magenta

# Plot the ratios of the transfer functions
ax.loglog(k, T_cdm / T_cdm_base_interp1, color=color1, lw=2.5, label=r'$T_{\mathrm{cdm}} / T_{\mathrm{cdm, base}}$ (This work)')
ax.loglog(k2, T_cdm2 / T_cdm_base_interp2, color=color2, lw=2.5, linestyle='--', label=r'$T_{\mathrm{cdm}} / T_{\mathrm{cdm, base}}$ (Benchmark)')

# --- 5. Final Plot Adjustments ---
ax.set_title(r'Comparison of CDM Transfer Functions (Normalized to Baseline)', fontsize=20, pad=15)
ax.set_xlabel(r'$k \quad [h/\mathrm{Mpc}]$', fontsize=18)
ax.set_ylabel(r'$|T(k) / T_{\mathrm{cdm, base}}|$', fontsize=18)

ax.legend(fontsize=16)
ax.grid(True, which="both", ls="--", alpha=0.4)

# Set axis limits if necessary (optional, but good for focusing the view)
ax.set_xlim(1e-4, 1e2)
ax.set_ylim(0.9, 1.1) # Example: zoom in on the y-axis to see differences

plt.tight_layout()
plt.show()