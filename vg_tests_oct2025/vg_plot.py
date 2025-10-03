import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.title('n=2, 1e-2GeV, envelope')

data_camb = np.loadtxt('data_tk/idm_n2_1e-2GeV_envelope_z99_Tk.dat')
kc = data_camb[:,0]
tkc = data_camb[:, 1]
plt.loglog(kc, tkc**2, color='k',label='camb')

#assert (tkc==tkc2).all()

data_class = np.loadtxt('output/explanatory00_tk.dat')
k = data_class[:,0]
tk = data_class[:,1]
plt.loglog(k, tk**2, color='red', label='class')

plt.xlabel('k')
plt.ylabel('t_dm^2')

plt.legend()

plt.show()

