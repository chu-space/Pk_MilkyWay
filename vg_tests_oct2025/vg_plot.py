import numpy as np
import matplotlib.pyplot as plt

index = 1
offset = 0

plt.figure()
#plt.title('n=2, 1e-2GeV, envelope')

#data_camb = np.loadtxt('data_tk/idm_n2_1e-2GeV_envelope_z99_Tk.dat')
data_camb = np.loadtxt('data_tk/test_transfer_z99.dat')
data_class = np.loadtxt('output/explanatory01_tk.dat')


kc = data_camb[:,0]
tkc = data_camb[:, index]

k = data_class[:,0]
tk = data_class[:,index+offset]

tkc_interp = np.interp(k, kc, tkc)

plt.loglog(kc, tkc**2, color='blue',label='camb')
plt.loglog(k, tk**2, color='red', label='class')

#plt.semilogx(k,(tkc_interp/tk)**2, color='k', label='ratio')
#plt.ylim(0.5, 1.5)



#assert (tkc==tkc2).all()
plt.xlabel('k')
plt.ylabel('t_dm^2 comparison')
plt.legend()
plt.show()

