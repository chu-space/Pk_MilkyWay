import pynbody
import numpy as np
#Load z = 0 particle snapshot for Halo004 WDM 3 keV
f = pynbody.load('/central/groups/carnegie_poc/enadler/ncdm_resims/Halo004/wdm_3/output_wdm_3/snapshot_235')
pos = f['pos']
mass = f['mass']
#Filter to only get high-resolution particles
pos = pos[mass==np.min(mass)]
mass = mass[mass==np.min(mass)]

print(pos)
print(mass)

