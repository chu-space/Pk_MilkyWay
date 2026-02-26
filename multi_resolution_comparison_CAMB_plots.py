import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import PchipInterpolator, InterpolatedUnivariateSpline
import os
import glob

BASE_DIR = os.getcwd()
CLASS_OUTPUT_DIR = os.path.join(BASE_DIR, "output_fast_precision_k200_bare_A_s_sigma8")
REF_CAMB_DIR = "/home/arifchu/Pk_MilkyWay/camb_data_tk"
Z_TARGET = 99.0
H_PARAM  = 0.7 

OUTPUT_PLOT_DIR = "PLOTS_RAW_AMPLITUDE_k200_bare_A_s_sigma8"
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

def get_col_map(fname):
    with open(fname, 'r') as f:
        for line in f:
            if '1:k' in line:
                parts = line.strip().split()
                return {p.split(':')[1].lower(): int(p.split(':')[0]) - 1 for p in parts if ':' in p}
    return None

def get_h_at_z(search_base):
    bfiles = glob.glob(os.path.join(CLASS_OUTPUT_DIR, f"{search_base}*background.dat"))
    if bfiles:
        data = np.loadtxt(sorted(bfiles)[0])
        z_vals, h_vals = data[:, 0], data[:, 3]
        if z_vals[0] > z_vals[-1]: z_vals, h_vals = np.flip(z_vals), np.flip(h_vals)
        return InterpolatedUnivariateSpline(z_vals, h_vals)(Z_TARGET)
    return 0.1267

def verify_model(model):    
    sync_file = os.path.join(CLASS_OUTPUT_DIR, f"{model['id']}_synchronous_00_tk.dat")
    newt_file = os.path.join(CLASS_OUTPUT_DIR, f"{model['id']}_newtonian_00_tk.dat")
    ref_path  = os.path.join(REF_CAMB_DIR, model['ref_file'])

    if not all(os.path.exists(f) for f in [sync_file, newt_file, ref_path]):
        print(f"  Error: Missing data files for {model['id']}")
        return

    s_map, n_map = get_col_map(sync_file), get_col_map(newt_file)
    s_data, n_data, ref_data = np.loadtxt(sync_file), np.loadtxt(newt_file), np.loadtxt(ref_path)
    Hz = get_h_at_z(f"{model['id']}_synchronous")

    k_vg = s_data[:, 0]
    k_ref = ref_data[:, 0]
    
    cdm_key = '-t_cdm/k2' if '-t_cdm/k2' in s_map else 't_cdm'
    b_key   = '-t_b/k2'   if '-t_b/k2'   in s_map else 't_b'
    vg_t2_cdm = s_data[:, s_map[cdm_key]]**2
    vg_t2_b   = s_data[:, s_map[b_key]]**2

    # Cubic spline interpolator to smoothen out the transfer functions for the velocity calculations
    th_cdm_int = PchipInterpolator(n_data[:, 0], n_data[:, n_map['t_cdm']])(k_vg)
    th_b_int   = PchipInterpolator(n_data[:, 0], n_data[:, n_map['t_b']])(k_vg)
    vg_v_cdm = np.abs((1 + Z_TARGET) * th_cdm_int / ((k_vg * H_PARAM)**2 * Hz))
    vg_v_b   = np.abs((1 + Z_TARGET) * th_b_int   / ((k_vg * H_PARAM)**2 * Hz))

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(4, 2, height_ratios=[3, 1.5, 3, 1.5], hspace=0.4, wspace=0.2)

    def add_panel(row, col, k_v, val_v, k_r, val_r, title):
        ax = fig.add_subplot(gs[row, col])
        ax_res = fig.add_subplot(gs[row+1, col], sharex=ax)
        
        val_r_int = PchipInterpolator(k_r, val_r)(k_v)

        # Loglog transfer function comparison
        ax.loglog(k_v, val_v, 'b-', label='CLASS (Raw)', lw=2)
        ax.loglog(k_r, val_r, 'r--', label='Reference', lw=1.5, alpha=0.8)
        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.grid(True, which="both", ls="-", alpha=0.1)
        if col == 0: ax.legend(fontsize=12)
        plt.setp(ax.get_xticklabels(), visible=False)

        # Residual plot below regular plot
        res = 100 * (val_v / val_r_int - 1)
        ax_res.semilogx(k_v, res, color='red', lw=1.5) 
        ax_res.axhline(0, color='black', lw=1.2)
        
        ax_res.set_ylim(-100, 100) 
        ax_res.yaxis.set_major_locator(MultipleLocator(20))
        ax_res.yaxis.set_minor_locator(MultipleLocator(5))
        
        ax_res.set_ylabel("Raw Diff %", fontsize=12)
        ax_res.grid(True, which="major", ls="-", alpha=0.3)
        ax_res.grid(True, which="minor", ls=":", alpha=0.2)
        if row == 2: ax_res.set_xlabel("k [1/Mpc]", fontsize=13)

    add_panel(0, 0, k_vg, vg_t2_cdm, k_ref, ref_data[:,1]**2, "CDM $T^2$")
    add_panel(0, 1, k_vg, vg_t2_b,   k_ref, ref_data[:,2]**2, "Baryon $T^2$")
    add_panel(2, 0, k_vg, vg_v_cdm,  k_ref, np.abs(ref_data[:,10]), "CDM $|v|$")
    add_panel(2, 1, k_vg, vg_v_b,    k_ref, np.abs(ref_data[:,11]), "Baryon $|v|$")

    plt.suptitle(f"Raw Amplitude Verification (No Pivot) | z={Z_TARGET}\nModel: {model['name']}", 
                 fontsize=20, y=0.97, fontweight='bold')
    
    save_path = os.path.join(OUTPUT_PLOT_DIR, f"raw_residual_{model['name']}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  -> Plot saved: {save_path}")

models = [
    {'name': 'n2_envelope', 'id': 'idm_n2_envelope', 'ref_file': 'idm_n2_1e-2GeV_envelope_z99_Tk.dat'},
    {'name': 'n4_halfmode', 'id': 'idm_n4_halfmode', 'ref_file': 'idm_1e-4GeV_halfmode_Tk.dat'}
]

for m in models:
    try: verify_model(m)
    except Exception as e: print(f"  Error: {e}")