import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
import re
from scipy.interpolate import InterpolatedUnivariateSpline

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "output_improved_wmap9_micro")  # Target folder
REF_DIR = os.path.abspath(os.path.join(BASE_DIR, "../Pk_MilkyWay/camb_data_tk"))
PLOT_DIR = os.path.join(BASE_DIR, "PLOTS_IMPROVED_WMAP9_micro")

os.makedirs(PLOT_DIR, exist_ok=True)

# FIXED: Matches the actual file names in your output_wmap9_micro folder
RUNS = {
    'Run_Fixed_Helium': {
        'label': 'WMAP9 Fixed Helium (YHe=0.24)',
        'color': 'red',
        'ls': '-', 
        'alpha': 1.0
    }
}

MODELS = [(4, "1e-4", "2.2e-27", "halfmode")]

# ==============================================================================
# ROBUST LOADER FUNCTIONS
# ==============================================================================
def find_col_by_name(filepath, possible_names):
    """Scans header for names and returns 0-based index."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('#') and ('1:' in line or 'k ' in line):
                    header = line.strip().lstrip('#').strip()
                    if ':' in header:
                        cols = [m.lower() for m in re.findall(r'\d+:\s*([^\s]+)', header)]
                    else:
                        cols = [x.lower() for x in header.split()]
                    for t in possible_names:
                        for i, c in enumerate(cols):
                            if t.lower() in c: return i
                    return -1
    except: pass
    return -1

def validate_index(shape, idx, default):
    return idx if (idx != -1 and idx < shape[1]) else default

def find_file(run_base, pattern_suffix):
    # Updated to handle the double underscore suffix seen in your directory
    matches = glob.glob(os.path.join(DATA_DIR, f"*{run_base}*{pattern_suffix}"))
    if matches:
        return sorted(matches, key=os.path.getmtime, reverse=True)[0]
    return None

def load_run_data(run_base):
    # Locate Files using your specific output naming convention
    f_sync = find_file(run_base, "synchronous__tk.dat")
    f_newt = find_file(run_base, "newtonian__tk.dat")
    f_back = find_file(run_base, "synchronous__background.dat")
    
    if not f_sync: f_sync = find_file(run_base, "newtonian__tk.dat")
    if not f_sync: return None

    try:
        d_s = np.loadtxt(f_sync)
        d_n = np.loadtxt(f_newt) if f_newt else None
        d_b = np.loadtxt(f_back) if f_back else None
    except: return None

    # Density (Using index 7 for Total based on your previous header info)
    ic = find_col_by_name(f_sync, ['dmeff','d_idm'])
    ib = find_col_by_name(f_sync, ['baryon','b'])
    it = find_col_by_name(f_sync, ['tot', 'total'])
    
    ic = validate_index(d_s.shape, ic, 2)
    ib = validate_index(d_s.shape, ib, 3)
    it = validate_index(d_s.shape, it, 7) 
    
    k = d_s[:,0]
    tc = np.abs(d_s[:,ic])
    tb = np.abs(d_s[:,ib])
    tt = np.abs(d_s[:,it])

    # Velocity conversion using BestFit h=0.6932
    vc, vb = np.zeros_like(k), np.zeros_like(k)
    if d_n is not None and d_b is not None:
        ivc = find_col_by_name(f_newt, ['dmeff','t_dmeff'])
        ivb = find_col_by_name(f_newt, ['baryon','t_b'])
        ivc = validate_index(d_n.shape, ivc, 3)
        ivb = validate_index(d_n.shape, ivb, 2)
        
        kn = d_n[:,0]
        thc = d_n[:,ivc]
        thb = d_n[:,ivb]
        
        # Interpolate H(z=99)
        z_col = d_b[:,0]
        if z_col[0] < z_col[-1]: func = InterpolatedUnivariateSpline(z_col, d_b[:,3])
        else: func = InterpolatedUnivariateSpline(np.flip(z_col), np.flip(d_b[:,3]))
        Hz = func(99)
        
        h_sim = 0.6932
        with np.errstate(divide='ignore', invalid='ignore'):
            vcr = (100 * thc) / ((kn * h_sim)**2 * Hz)
            vbr = (100 * thb) / ((kn * h_sim)**2 * Hz)
        
        vc = np.interp(k, kn, np.nan_to_num(vcr))
        vb = np.interp(k, kn, np.nan_to_num(vbr))

    return {'k':k, 'T_cdm':tc, 'T_b':tb, 'T_tot':tt, 'v_cdm':vc, 'v_b':vb}

def load_ref(n, m, t):
    names = [f"idm_{m}GeV_{t}_Tk.dat", f"idm_n{n}_{m}GeV_{t}_Tk.dat"]
    for name in names:
        p = os.path.join(REF_DIR, name)
        if os.path.exists(p):
            d = np.loadtxt(p)
            return {'k': d[:,0], 'T_cdm': d[:,1], 'T_b': d[:,2], 'T_tot': d[:,6], 'v_cdm': d[:,10], 'v_b': d[:,11]}
    return None

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f"Generating WMAP9 Micro Comparison Plots...")
    for n, m, s, t in MODELS:
        ref = load_ref(n, m, t)
        if not ref: 
            print(f"  [SKIP] Reference not found for n={n}, m={m}")
            continue
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 3, height_ratios=[3, 1, 3, 1], hspace=0.3)
        panels = [
            {'t':'CDM', 'k':'T_cdm', 'sq':1, 'r':0, 'c':0}, 
            {'t':'Baryon', 'k':'T_b', 'sq':1, 'r':0, 'c':1}, 
            {'t':'Total', 'k':'T_tot', 'sq':1, 'r':0, 'c':2},
            {'t':'v_CDM', 'k':'v_cdm', 'sq':0, 'r':2, 'c':0}, 
            {'t':'v_Baryon', 'k':'v_b', 'sq':0, 'r':2, 'c':1}
        ]
        
        data_found = False
        for name, cfg in RUNS.items():
            base_id = f"idm_{name}_n{n}_m{m}_s{s}_{t}"
            dat = load_run_data(base_id)
            if not dat: 
                print(f"  [WARN] Data missing for {base_id}")
                continue
            data_found = True
            
            for p in panels:
                ax = fig.add_subplot(gs[p['r'], p['c']])
                axr = fig.add_subplot(gs[p['r']+1, p['c']], sharex=ax)
                
                yr = ref[p['k']]**2 if p['sq'] else np.abs(ref[p['k']])
                yd = dat[p['k']]**2 if p['sq'] else np.abs(dat[p['k']])
                
                ax.loglog(ref['k'], yr, 'k-', alpha=0.2, lw=4, label='Reference')
                ax.loglog(dat['k'], yd, color=cfg['color'], ls=cfg['ls'], label=cfg['label'])
                
                ci = np.logspace(np.log10(0.01), np.log10(10), 2000)
                ri, di = np.interp(ci, ref['k'], yr), np.interp(ci, dat['k'], yd)
                
                idx = (np.abs(ci-0.1)).argmin()
                norm = ri[idx]/di[idx] if (di[idx]!=0 and ri[idx]!=0) else 1.0
                
                axr.semilogx(ci, (di*norm)/ri, color='green', lw=1.5)
                axr.axhline(1, c='gray', ls=':')
                axr.set_ylim(0.98, 1.02)
                axr.set_xlim(0.01, 10)
                
                if p['c']==0 and p['r']==0: ax.legend()
                ax.set_title(p['t'])

        if data_found:
            out_name = f"WMAP9_MicroComp_n{n}_{m}.png"
            plt.savefig(os.path.join(PLOT_DIR, out_name), bbox_inches='tight')
            print(f"  -> Saved: {out_name}")
        else:
            print("  [ERROR] No valid data found. Check RUNS names vs Filenames.")

if __name__ == "__main__":
    main()

