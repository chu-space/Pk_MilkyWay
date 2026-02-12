import numpy as np
from scipy.interpolate import Akima1DInterpolator
import matplotlib.pyplot as plt

index = 1
offset = 0


plt.figure()
#plt.title('n=2, 1e-2GeV, envelope')

#dmeff:
#data_camb = np.loadtxt('data_tk/idm_n2_1e-2GeV_envelope_z99_Tk.dat')
data_class = np.loadtxt('output/n2_1e-2GeV_7.1e-24_camb_tk.dat')

#cdm:  
data_camb = np.loadtxt('data_tk/test_transfer_z99.dat')
#data_class = np.loadtxt('output/explanatoryCDM_tk.dat')

kc = data_camb[:,0]
tkc = data_camb[:, index]

k = data_class[:,0]
tk = data_class[:, index+offset]
tk_dmeff = data_class[:, index+1]

#### now Pk
#data_class_pk_dmeff = np.loadtxt('output/n2_1e-2GeV_7.1e-24_pk.dat')
#data_class_pk_cdm = np.loadtxt('output/explanatoryCDM_pk.dat')

#k_dmeff = data_class_pk_dmeff[:,0]
#pk_dmeff = data_class_pk_dmeff[:,1]

#k_cdm = data_class_pk_cdm[:,0]
#pk_cdm = data_class_pk_cdm[:,1]
#pk_cdm_interp = np.exp(np.interp(np.log(k_dmeff), np.log(k_cdm), np.log(pk_cdm)))



# Hybrid interpolation: use log-log for positive data, Akima for oscillatory/negative
# Clip k to valid range to prevent extrapolation artifacts at boundaries
k_min, k_max = kc.min(), kc.max()
k_clipped = np.clip(k, k_min, k_max)

# Check if all values are positive (use log-log interpolation if so)
if np.all(tkc > 0) and np.all(kc > 0):
    # Log-log interpolation: best for smooth positive transfer functions
    tkc_interp = np.exp(np.interp(np.log(k_clipped), np.log(kc), np.log(tkc)))
else:
    # Akima interpolation: handles negative values and oscillatory behavior
    sort_idx = np.argsort(kc)
    kc_sorted = kc[sort_idx]
    tkc_sorted = tkc[sort_idx]
    akima_interp = Akima1DInterpolator(kc_sorted, tkc_sorted)
    tkc_interp = akima_interp(k_clipped)

#plt.loglog(kc, tkc**2, color='blue',label='camb')
#plt.loglog(k, tk**2, color='red', label='class')

#plt.semilogx(k,(tk/tkc_interp)**2, color='k', label='ratio')
#plt.ylim(0.5, 1.5)

#column comparison:
#plt.loglog(k,(tk/tkc_interp)**2, color='k', label='cdm column')
#plt.loglog(k,(tk_dmeff/tkc_interp)**2, color='red', label='dmeff column')

plt.semilogx(k,np.abs(tk_dmeff-tk)**2/tkc_interp**2, color='k', label='(dmeff-cdm)^2/real_cdm^2')
plt.xlim(0.1,200)
#plt.ylim(0.5, 1.5)

#power spectra
#plt.semilogx(k_dmeff,pk_dmeff/pk_cdm_interp, color='k', label='ratio')
#plt.loglog(k_dmeff,pk_dmeff*k_dmeff**3/2./np.pi**2, color='k', label='pk')
#plt.loglog(k, (tk)**2, color='blue', label='tk')



plt.xlabel('k')
#plt.ylabel('P_IDM/P_CDM')
plt.legend()
plt.show()
#plt.savefig('n2_1e-2GeV_7.1e-24_pk_ratio.pdf', dpi=300)

