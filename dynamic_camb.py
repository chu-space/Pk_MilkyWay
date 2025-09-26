import numpy as np
import pandas as pd
import re
from scipy.interpolate import InterpolatedUnivariateSpline
import sys

def load_data_with_dynamic_headers(filepath):
    """
    Loads data from a CLASS/CAMB output file into a pandas DataFrame.

    This function dynamically finds the header row in the commented section,
    parses the column names, cleans them up, and uses them to label the
    data columns. This avoids hardcoding column indices.

    Args:
        filepath (str): The path to the data file.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns named according to the file's header,
                      or None if the file cannot be read.
    """
    # --- 1. Find the header line and number of comment lines to skip ---
    header_line = None
    lines_to_skip = 0
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if line.strip().startswith('#'):
                    lines_to_skip = i + 1
                    # A valid header line should contain column definitions like '1:'
                    if '1:' in line:
                        header_line = line
                else:
                    # Stop searching once we hit the data
                    break
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return None

    if not header_line:
        print(f"Warning: Could not find a valid header row in {filepath}. Loading data without column names.", file=sys.stderr)
        data = np.loadtxt(filepath)
        return pd.DataFrame(data)

    # --- 2. Parse and clean column names ---
    header_content = header_line.strip().lstrip('#').strip()
    # This regex finds all text associated with an index (e.g., '1:'), capturing
    # the full column name until the next index or the end of the line.
    matches = re.findall(r'\d+:\s*(.*?)(?=\s+\d+:|$)', header_content)

    column_names = []
    for match in matches:
        raw_name = match.strip()
        # Apply specific cleaning rules for known complex names
        if raw_name == '-T_cdm/k2':   clean_name = 'cdm_t'
        elif raw_name == '-T_b/k2':    clean_name = 'baryon_t'
        elif raw_name == '-T_tot/k2':  clean_name = 'total_t'
        else:
            # General cleaning: remove units and replace special chars
            clean_name = re.sub(r'[\(\[].*?[\)\]]', '', raw_name).strip()
            clean_name = clean_name.replace(' ', '_').replace('-', '_').replace('.', '')
        column_names.append(clean_name)

    # --- 3. Load data with pandas ---
    try:
        df = pd.read_csv(
            filepath,
            comment='#',
            delim_whitespace=True,
            header=None,
            names=column_names
        )
        return df
    except Exception as e:
        print(f"Error loading data with pandas for {filepath}: {e}", file=sys.stderr)
        return None

# ==================================
# === Main Script Logic ===
# ==================================

# --- Constants and Input files ---
h = 0.7  # Hubble constant in units of 100 km/s/Mpc
z = 99   # Redshift at which to compute H(z)

sync_name = 'output/vanilla_new_camb_n_2_1e-2_halfmode_synchronous00'
new_name = 'output/vanilla_new_camb_n_2_1e-2_halfmode_newtonian00'
background_name = 'output/vanilla_new_camb_n_2_1e-2_halfmode_newtonian00' # Corrected to use the same base as new_name

files = {
    'sync': f'{sync_name}_tk.dat',
    'new': f'{new_name}_tk.dat',
    'back': f'{background_name}_background.dat'
}

# --- Load data using the new dynamic function ---
print("Loading data files...")
df_camb_sync = load_data_with_dynamic_headers(files['sync'])
df_class_new = load_data_with_dynamic_headers(files['new'])
df_back = load_data_with_dynamic_headers(files['back'])

# --- Exit if any file failed to load ---
if any(df is None for df in [df_camb_sync, df_class_new, df_back]):
    print("One or more data files could not be processed. Exiting.", file=sys.stderr)
    sys.exit(1)

# --- Verify that required columns exist ---
required_cols = {
    'sync': ['k', 'cdm_t', 'baryon_t', 'total_t'],
    'new': ['k', 't_cdm', 't_b'],
    'back': ['z', 'H']
}

all_cols_found = True
for name, df in [('sync', df_camb_sync), ('new', df_class_new), ('back', df_back)]:
    missing = set(required_cols[name]) - set(df.columns)
    if missing:
        print(f"Error: Missing required columns in {files[name]}: {missing}", file=sys.stderr)
        print(f"  Available columns: {list(df.columns)}", file=sys.stderr)
        all_cols_found = False

if not all_cols_found:
    print("Exiting due to missing columns.", file=sys.stderr)
    sys.exit(1)

print("All required columns found. Proceeding with calculations.")

# --- Interpolate H(z) from background data ---
# Data is flipped because the file lists redshift in descending order.
funH_z = InterpolatedUnivariateSpline(
    df_back['z'].values[::-1],
    df_back['H'].values[::-1]
)
H = funH_z(z)  # H(z) in 1/Mpc

# --- Compute velocities ---
k_h = df_class_new['k'].values * h  # Convert k to h/Mpc units
vc = (1 + z) * df_class_new['t_cdm'].values / (k_h**2 * H)  # v_cdm
vb = (1 + z) * df_class_new['t_b'].values / (k_h**2 * H)    # v_b
dummy = np.zeros_like(vc)

# --- Assemble output table using DataFrame columns ---
output_data = np.column_stack((
    df_camb_sync['k'].values / h,                   # k/h
    np.abs(df_camb_sync['cdm_t'].values),           # CDM
    np.abs(df_camb_sync['baryon_t'].values),        # baryon
    dummy, dummy, dummy,                            # photon, nu, mass_nu (unused)
    np.abs(df_camb_sync['total_t'].values),         # total
    dummy, dummy, dummy,                            # no_nu, total_de, Weyl (unused)
    np.abs(vc),                                     # v_CDM
    np.abs(vb),                                     # v_b
    dummy                                           # v_b - v_c (unused)
))

# --- Save to file ---
column_names = [
    'k/h', 'CDM', 'baryon', 'photon', 'nu', 'mass_nu',
    'total', 'no_nu', 'total_de', 'Weyl', 'v_CDM', 'v_b', 'v_b-v_c'
]
header_line = "# " + "".join(f"{name:>15s}" for name in column_names)
output_filename = 'best_new_n_2_1e-2GeV_Tk.dat'

try:
    with open(output_filename, 'w') as f:
        f.write(header_line + '\n')
        np.savetxt(f, output_data, fmt='%15.6e')
    print(f"\nSuccessfully processed data and saved to {output_filename}")
except IOError as e:
    print(f"Error saving file: {e}", file=sys.stderr)