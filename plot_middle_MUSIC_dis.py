import numpy as np
import matplotlib.pyplot as plt
import os
import pynbody

# --- Configuration ---
plt.rcParams.update({
    'figure.facecolor': 'w',
    'axes.linewidth': 1.5,
    'legend.frameon': False,
    'text.usetex': False
})

COLORS = {
    'middle': '#9C27B0',      # Purple
    'ethan_env': '#1b9e77',    # Green (Ref)
    'ethan_hm': '#d95f02'      # Orange (Ref)
}

BASE_PATH_USER = "/resnick/groups/carnegie_poc/achu"
BASE_PATH_ETHAN = "/resnick/groups/carnegie_poc/enadler/ncdm_resims/Halo004"

def load_ic_data(ic_path):
    if not ic_path or not os.path.exists(ic_path):
        return None, None
    try:
        f = pynbody.load(ic_path)
        highres_mass = np.min(np.unique(f['mass']))
        mask = f['mass'] == highres_mass
        rho = f['rho'][mask]
        delta = (rho - np.mean(rho)) / np.mean(rho)
        hist, edges = np.histogram(delta, bins=150, density=True)
        x = 0.5 * (edges[1:] + edges[:-1])
        p = hist / np.trapz(hist, x)
        return x, p
    except:
        return None, None

def find_ic(base, folder_name):
    path = os.path.join(base, folder_name)
    for sub in ['ic/ic_gadget_dist', 'ic_gadget_dist', '']:
        loc = os.path.join(path, sub)
        if os.path.exists(loc) and not os.path.isdir(loc):
            return loc
    return None

def create_middle_plot(n, mass_id, mass_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check both common variants found in your screenshot
    variants = [f"1e-{mass_id}", mass_id]
    ethan_prefix = f"idm_n2_{mass_id}" if n == 2 else f"idm_{mass_id}"

    print(f"Generating plot for n={n}, m={mass_label}...")

    # Plot my  Middle MUSIC plots (Solid Purple)
    for v in variants:
        path = find_ic(BASE_PATH_USER, f"n{n}_{v}_middle_fixed")
        x, p = load_ic_data(path)
        if x is not None:
            ax.plot(x, p, label=f"Arif Middle ({v})", color=COLORS['middle'], lw=2)
            break # Stop once we find one valid version

    # Plot Reference Data (Dashed)
    ref_env = find_ic(BASE_PATH_ETHAN, f"{ethan_prefix}_envelope")
    ref_hm = find_ic(BASE_PATH_ETHAN, f"{ethan_prefix}_halfmode")
    
    xr_e, pr_e = load_ic_data(ref_env)
    if xr_e is not None: ax.plot(xr_e, pr_e, label="Ethan Env (Ref)", color=COLORS['ethan_env'], ls='--', lw=2)
    
    xr_h, pr_h = load_ic_data(ref_hm)
    if xr_h is not None: ax.plot(xr_h, pr_h, label="Ethan HM (Ref)", color=COLORS['ethan_hm'], ls='--', lw=2)

    ax.set_title(f"IDM Middle vs Reference: n={n}, m={mass_label}")
    ax.set_yscale('log')
    ax.set_xlim(-0.5, 0.8)
    ax.set_ylim(1e-4, 10)
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'$P(\delta)$')
    ax.legend(loc='upper right')
    
    save_name = f"middle_only_n{n}_{mass_id}.pdf"
    plt.savefig(save_name, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_name}")

if __name__ == "__main__":
    for n in [2, 4]:
        for m_id, m_lab in [('1GeV', '1 GeV'), ('1e-2GeV', '10^{-2} GeV'), ('1e-4GeV', '10^{-4} GeV')]:
            create_middle_plot(n, m_id, m_lab)
