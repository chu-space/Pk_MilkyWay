import numpy as np

# Load the base files 
wdm_65 = np.load('6.5_wdm_transfer.npy', allow_pickle=True)
cdm_pk = np.load('cdm_transfer.npy', allow_pickle=True)

k = wdm_65[:, 0]
cdm_vals = cdm_pk

# The Vogel & Abazajian 23 parameterisation was adopted for 5.9 keV WDM (our new WDM bounds) according to COZMIC 1
m_wdm = 5.9
omega_mh2 = 0.11711
h = 0.7
A = 0.0437
B = -1.188
nu = 1.049
theta = 2.012
eta = 0.2463

# Calculate the WDM transfer function ratio
alpha = A * (m_wdm**B) * ((omega_mh2 / 0.12)**eta) * ((h / 0.6736)**theta)
Pk_ratio = (1 + (alpha * k)**(2 * nu))**(-5.0 / nu)

# Multiply the CDM transfer function by the cutoff ratio to get WDM
wdm_59_vals = cdm_vals * Pk_ratio

# Stack the k-values and new WDM transfer function together and ensure format matches original
wdm_59 = np.column_stack((k, wdm_59_vals)).astype(np.float64)

# Save to the new NPY file safely
np.save('5.9_wdm_transfer.npy', wdm_59)

