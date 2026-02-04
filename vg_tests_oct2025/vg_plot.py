import numpy as np
from scipy.interpolate import Akima1DInterpolator
import matplotlib.pyplot as plt

index = 1
offset = 1

#cdm = True; dmeff = False
dmeff = True; cdm = False

plt.figure()
#plt.title('n=2, 1e-2GeV, envelope')

if dmeff:
    data_camb = np.loadtxt('data_tk/idm_n2_1e-2GeV_envelope_z99_Tk.dat')
    data_class = np.loadtxt('output/explanatory01_tk.dat')
if cdm:
    data_camb = np.loadtxt('data_tk/test_transfer_z99.dat')
    data_class = np.loadtxt('output/explanatoryCDM_tk.dat')
    #data_class = np.loadtxt('output/calibration_run-vg07_tk.dat')


kc = data_camb[:,0]
tkc = data_camb[:, index]

k = data_class[:,0]
tk = data_class[:, index+offset]


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

plt.loglog(kc, tkc**2, color='blue',label='camb')
plt.loglog(k, tk**2, color='red', label='class')

#plt.semilogx(k,(tkc_interp/tk)**2, color='k', label='ratio')
#plt.ylim(0.5, 1.5)



#assert (tkc==tkc2).all()
plt.xlabel('k')
plt.ylabel('t_dm^2 comparison')
plt.legend()
plt.show()

