import numpy as np
import matplotlib.pyplot as plt
import os
import pynbody
import pickle

def get_plot_data(base_path):
    for sub in ['ic/ic_gadget_dist', 'ic_gadget_dist', '']:
        full_path = os.path.join(base_path, sub)
        if os.path.isfile(full_path):
            f = pynbody.load(full_path)
            mask = f['mass'] == np.min(np.unique(f['mass']))
            rho = f['rho'][mask]
            delta = (rho - np.mean(rho)) / np.mean(rho)
            hist, edges = np.histogram(delta, bins=150, density=True)
            x = 0.5 * (edges[1:] + edges[:-1])
            return x, hist / np.trapz(hist, x)
    return None, None

# Change these paths based on where you have stored your MUSIC IC output and reference directories (these should be copied over or accessed remotely as it is too large for this repo to store

path_env = "/resnick/groups/carnegie_poc/achu/n2_1e-2GeV_envelope_fixed"
path_hm = "/resnick/groups/carnegie_poc/achu/n2_1e-2GeV_halfmode_fixed"
ref_env = "/resnick/groups/carnegie_poc/enadler/ncdm_resims/Halo004/idm_n2_1e-2GeV_envelope"
ref_hm = "/resnick/groups/carnegie_poc/enadler/ncdm_resims/Halo004/idm_n2_1e-2GeV_halfmode"

x_env, p_env = get_plot_data(path_env)
x_hm, p_hm = get_plot_data(path_hm)
x_renv, p_renv = get_plot_data(ref_env)
x_rhm, p_rhm = get_plot_data(ref_hm)

x_wdm, p_wdm = None, None
try:
    with open('/central/groups/carnegie_poc/enadler/ncdm_resims/analysis/ic_density_velocity.bin', 'rb') as f:
        ic_data = pickle.load(f, encoding='latin1')
        counts, edges = ic_data['Halo004']['wdm_6.5'][0]
        x_wdm = 0.5 * (edges[1:] + edges[:-1])
        p_wdm = counts / np.trapz(counts, x_wdm)
except:
    pass

fig, ax = plt.subplots(figsize=(8, 5))

if x_wdm is not None:
    ax.plot(x_wdm, p_wdm, color='gray', ls=':', lw=2, alpha=0.7, label="WDM 6.5 keV")
if x_renv is not None:
    ax.plot(x_renv, p_renv, color='#1b9e77', ls='--', lw=2, label="Ethan's Envelope")
if x_rhm is not None:
    ax.plot(x_rhm, p_rhm, color='#d95f02', ls='--', lw=2, label="Ethan's Halfmode")
if x_env is not None:
    ax.plot(x_env, p_env, color='#1E88E5', ls='-', lw=2, label="Arif's Envelope")
if x_hm is not None:
    ax.plot(x_hm, p_hm, color='#FF5722', ls='-', lw=2, label="Arif's Halfmode")

ax.axvline(0, color='gray', ls='--', alpha=0.3)
ax.set_title("IDM Comparison: n=2, m=$10^{-2}$ GeV")
ax.set_xlabel('δ = (ρ - ρ̄)/ρ̄')
ax.set_ylabel('P(δ)')
ax.set_yscale('log')
ax.set_xlim(-0.5, 0.8)
ax.set_ylim(1e-4, 10)
ax.grid(True, alpha=0.2, linestyle='--')
ax.legend()

plt.tight_layout()
plt.savefig("comparison_n2_1e-2GeV_all.pdf", bbox_inches='tight')
