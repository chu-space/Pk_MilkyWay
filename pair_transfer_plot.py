import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

# --- Constants ---
h = 0.7
npow_dmeff = 2
idm_mass = 1e-4                  # GeV
sig_values = [4.2e-28, 2.8e-27]  # cm^2
colors = ["#59655F", '#3CFF97']  # Dark Gray, Aqua Green

# --- Paths and Setup ---
cdm_ini_template = "base_cdm_template.ini"
idm_ini_template = "base_idm_template.ini"
output_dir = "correct_n_2_repo"
os.makedirs(output_dir, exist_ok=True)

# --- Load Precomputed Data from .npy Files ---
try:
    # This array is used as the COMMON K-GRID for interpolation
    k_halfmode_raw = np.load("COZMIC_IDM_Tk/halfmode_k_idm_1e-4_n2.npy", allow_pickle=True)
    k_envelope_raw = np.load("COZMIC_IDM_Tk/envelope_k_idm_1e-4_n2.npy", allow_pickle=True)
    wdm_data_raw = np.load('6.5_wdm_transfer.npy', allow_pickle=True)
    cdm_T_raw = np.load('6.5_cdm_transfer.npy', allow_pickle=True)

    # IDM Power spectrum for models
    rui_halfmode_T_raw = np.load("COZMIC_IDM_Tk/halfmode_idm_1e-4_n2.npy", allow_pickle=True)
    rui_envelope_T_raw = np.load("COZMIC_IDM_Tk/envelope_idm_1e-4_n2.npy", allow_pickle=True)

    # CDM Power spectrum for models
    halfmode_cdm_comparison_T_raw = np.load("COZMIC_IDM_Tk/halfmode_cdm_1e-4_n2.npy", allow_pickle=True)
    envelope_cdm_comparison_T_raw = np.load("COZMIC_IDM_Tk/envelope_cdm_1e-4_n2.npy", allow_pickle=True)

except FileNotFoundError as e:
    print(f"Error: Could not find a required data file: {e.filename}")
    exit()

# === FIX 1: Convert ALL reference K-values from h/Mpc to Mpc^-1 by multiplying by h ===
k_halfmode = k_halfmode_raw * h
k_envelope = k_envelope_raw * h
wdm_data_raw[:, 0] = wdm_data_raw[:, 0] * h # Convert WDM k-values

# --- Helper Function: Run CLASS and Read P(k) (No changes needed) ---
def run_class_and_get_pk(ini_path):
    """Runs CLASS with a given .ini file and returns the power spectrum."""
    try:
        # Note: CLASS output k is in units of h/Mpc
        result = subprocess.run(["./class", ini_path], check=True, capture_output=True, text=True)
        root = None
        with open(ini_path) as f:
            for line in f:
                if line.strip().startswith("root"):
                    root = line.split("=")[1].strip()
        if root is None:
            raise ValueError(f"Missing 'root' parameter in {ini_path}")

        pk_path = f"{root}pk.dat"
        if not os.path.exists(pk_path):
             pk_path_alt = f"{root}00_pk.dat"
             if not os.path.exists(pk_path_alt):
                raise FileNotFoundError(f"CLASS output not found. Looked for {pk_path} and {pk_path_alt}")
             pk_path = pk_path_alt

        data = np.loadtxt(pk_path)
        k_vals_h = data[:, 0]
        pk_vals = data[:, 1]
        return k_vals_h, pk_vals
    except subprocess.CalledProcessError as e:
        print(f"Error running CLASS with {ini_path}:\n{e.stderr}")
        raise

# --- Helper Function: Find Half-Mode Scale with Interpolation (No changes needed) ---
def find_halfmode_k(k_vals, T2_vals):
    """Finds a precise half-mode wavenumber k where T²(k) crosses 0.25."""
    cross_indices = np.where(T2_vals < 0.25)[0]
    if len(cross_indices) == 0: return np.nan
    idx2 = cross_indices[0]
    if idx2 == 0: return k_vals[idx2]
    idx1 = idx2 - 1
    k1, k2 = k_vals[idx1], k_vals[idx2]
    T1, T2 = T2_vals[idx1], T2_vals[idx2]

    # Check if the curve is strictly decreasing at the cross point for log-linear interp
    if T1 <= T2:
        return np.interp(0.25, [T1, T2], [k1, k2])

    log_k1, log_k2 = np.log(k1), np.log(k2)
    m = (T2 - T1) / (log_k2 - log_k1)
    if m == 0: return k1
    log_k_half = log_k1 + (0.25 - T1) / m
    return np.exp(log_k_half)

# --- 2. Run Baseline CDM model ---
print("Running baseline CDM simulation...")
cdm_run_ini = os.path.join(output_dir, "cdm_run.ini")
with open(cdm_ini_template) as f_in, open(cdm_run_ini, "w") as f_out:
    for line in f_in:
        if line.strip().startswith("root"):
            f_out.write(f"root = {output_dir}/cdm_run\n")
        else:
            f_out.write(line)

k_cdm_raw_h, pk_cdm_raw = run_class_and_get_pk(cdm_run_ini)

# FIX 2: Correctly convert k from h/Mpc to Mpc^-1 by MULTIPLYING by h
k_cdm_raw = k_cdm_raw_h * h

pk_cdm_interp = np.interp(k_halfmode, k_cdm_raw, pk_cdm_raw)

# --- 3. Setup Plot ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xscale('log')
ax.set_xlabel(r'$k \ [\mathrm{Mpc}^{-1}]$', fontsize=16)
ax.set_ylabel(r'$T^2(k) = P_{\mathrm{model}}(k) / P_{\mathrm{CDM}}(k)$', fontsize=16)
ax.set_xlim([1e-3, 1e3])
ax.set_ylim([0, 1.05])
ax.grid(True, which="both", linestyle='--', linewidth=0.5)

# --- 4. Plot Reference Data (using new, converted k-values) ---
ax.plot(wdm_data_raw[:, 0], wdm_data_raw[:, 1] / cdm_T_raw, label="WDM 6.5 keV (Simulated)", lw=2.5, color='dodgerblue')
ax.plot(k_halfmode, rui_halfmode_T_raw[:, 1] / halfmode_cdm_comparison_T_raw, label="Rui (Half-mode)", lw=2.5, color='crimson', linestyle='--')
ax.plot(k_envelope, rui_envelope_T_raw[:, 1] / envelope_cdm_comparison_T_raw, label="Rui (Envelope)", lw=2.5, color='orange', linestyle=':')

# --- 5. Run IDM models and Plot ---
print("Running IDM simulations...")
for idx, sig in enumerate(sig_values):
    ini_path = os.path.join(output_dir, f"idm_sig_{sig:.1e}.ini")
    with open(idm_ini_template) as f_in, open(ini_path, "w") as f_out:
        for line in f_in:
            if line.strip().startswith("sigma_dmeff"): f_out.write(f"sigma_dmeff = {sig}\n")
            elif line.strip().startswith("m_dmeff"): f_out.write(f"m_dmeff = {idm_mass}\n")
            elif line.strip().startswith("npow_dmeff"): f_out.write(f"npow_dmeff = {npow_dmeff}\n")
            elif line.strip().startswith("root"):
                sig_str = f"{sig:.1e}".replace("-", "m").replace("+", "p")
                f_out.write(f"root = {output_dir}/idm_run_s{sig_str}\n")
            else: f_out.write(line)

    k_idm_raw_h, pk_idm_raw = run_class_and_get_pk(ini_path)

    # FIX 3: Correctly convert k from h/Mpc to Mpc^-1 by MULTIPLYING by h
    k_idm_raw = k_idm_raw_h * h

    pk_idm_interp = np.interp(k_halfmode, k_idm_raw, pk_idm_raw)
    T2 = pk_idm_interp / pk_cdm_interp

    if idx == 0:
        label_text = fr'IDM $\sigma = 4.2\text{{e}}-28\ \text{{cm}}^2$'
    else:
        label_text = fr'IDM $\sigma = 2.8\text{{e}}-27\ \text{{cm}}^2$'

    ax.plot(k_halfmode, T2, color=colors[idx], linewidth=3,
            label=label_text)

    k_half = find_halfmode_k(k_halfmode, T2)
    if not np.isnan(k_half):
        ax.axvline(k_half, color=colors[idx], linestyle=':', alpha=0.8, linewidth=2)
        print(f"  -> IDM σ={sig:.2e} cm²: k_halfmode ≈ {k_half:.3f} 1/Mpc")
    else:
        print(f"  -> IDM σ={sig:.2e} cm²: Half-mode not found in k-range.")

# --- Final Touches ---
ax.axhline(1.0, color='mediumspringgreen', linestyle='-', lw=1.5)
ax.axhline(0.25, color='gray', linestyle='-', alpha=1.0, lw=1.5, label=r'Half-mode ($\mathrm{T}^2 = 0.25$)')
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig('n_2_1e-4GeV_idm_wdm_transfer_comparison_final.pdf', dpi=300)
print("\nPlot saved to n_2_1e-4GeV_idm_wdm_transfer_comparison_final.pdf")
plt.show()
