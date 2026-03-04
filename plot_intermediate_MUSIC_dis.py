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

C_CDM = 'black'
LS_CDM = ':'
C_WDM = '#A0522D'  # Brown
LS_WDM = '--'
C_IDM = 'blue'
LS_IDM = '-'

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
    except Exception as e:
        print(f"  Error loading {ic_path}: {e}")
        return None, None

def find_ic(base, folder_name):
    path = os.path.join(base, folder_name)
    for sub in ['ic/ic_gadget_dist', 'ic_gadget_dist', '']:
        loc = os.path.join(path, sub)
        if os.path.exists(loc) and not os.path.isdir(loc):
            return loc
    return None

def create_intermediate_plot(n, mass_id, mass_label, model_type):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Load and Plot References (CDM & WDM 6.5 keV)
    cdm_path = find_ic(BASE_PATH_ETHAN, 'cdm')
    wdm_path = find_ic(BASE_PATH_ETHAN, 'wdm_6.5')
    
    x_cdm, p_cdm = load_ic_data(cdm_path)
    if x_cdm is not None:
        ax.plot(x_cdm, p_cdm, label='CDM', color=C_CDM, ls=LS_CDM, lw=1.5)
        
    x_wdm, p_wdm = load_ic_data(wdm_path)
    if x_wdm is not None:
        ax.plot(x_wdm, p_wdm, label='WDM 6.5 keV', color=C_WDM, ls=LS_WDM, lw=1.5)

    # 2. Load and Plot the Intermediate IDM Model
    folder_name = f"n{n}_{mass_id}_intermediate_{model_type}_fixed"
    idm_path = find_ic(BASE_PATH_USER, folder_name)
    
    x_idm, p_idm = load_ic_data(idm_path)
    if x_idm is not None:
        label_type = "int. envelope" if model_type == "envelope" else "int. halfmode"
        label = f"IDM n={n}, {mass_label} ({label_type})"
        ax.plot(x_idm, p_idm, label=label, color=C_IDM, ls=LS_IDM, lw=1.5)
    else:
        print(f"Missing IDM data: {folder_name}")
        plt.close(fig) # Close the figure without saving to prevent blank PDFs
        return

    # 3. Styling
    label_type = "int. envelope" if model_type == "envelope" else "int. halfmode"
    ax.set_title(f"IDM n={n}, {mass_label} ({label_type})", fontsize=14)
    ax.axvline(0, color='gray', ls='--', alpha=0.3, lw=1.5) 
    
    ax.set_yscale('log')
    ax.set_xlim(-0.45, 0.8)
    ax.set_ylim(3e-4, 5) 
    
    ax.set_xlabel(r'$\delta$', fontsize=14)
    ax.set_ylabel(r'$P(\delta)$', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    
    save_name = f"intermediate_{model_type}_n{n}_{mass_id}.pdf"
    plt.savefig(save_name, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_name}")

if __name__ == "__main__":
    tasks = [
        ('1e-1GeV', '10^{-1} GeV'),
        ('1e-3GeV', '10^{-3} GeV')
    ]
    
    for n in [2, 4]:
        for m_id, m_lab in tasks:
            print(f"Processing n={n}, {m_id}...")
            create_intermediate_plot(n, m_id, m_lab, 'envelope')
            create_intermediate_plot(n, m_id, m_lab, 'halfmode')
