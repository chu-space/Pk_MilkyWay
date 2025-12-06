import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

files = ['idm_run_n2_m1ep00_s1.6e-23_halfmode_synchronous00_tk.dat','idm_run_n2_m1ep00_s1.6e-23_halfmode_newtonian00_tk.dat','idm_run_n2_m1ep00_s1.6e-23_halfmode_newtonian00_background.dat']
data_camb_sync = np.loadtxt(files[0])
data_class_new=np.loadtxt(files[1])
data_back=np.loadtxt(files[2])

funH_z=InterpolatedUnivariateSpline(np.flip(data_back[:,0]),np.flip(data_back[:,3])) #data_back[:,0] = z, data_back[:,3] = H[1/Mpc]
z=99
H=funH_z(z) #1/Mpc

header = '{0:^15s} {1:^15s} {2:^15s} {3:^15s} {4:^15s} {5:^15s} {6:^15s} {7:^15s} {8:^15s} {9:^15s} {10:^15s} {11:^15s} {12:^15s}'.format('k/h','CDM','baryon','photon','nu','mass_nu','total','no_nu','total_de','Weyl','v_CDM','v_b','v_b-v_c')

h=0.7

dummy=[]
vc=[]
vb=[]
for i in range(len(data_class_new[:,0])):
    dummy.append(0.0)
    vc.append((1+z)*data_class_new[i,3]/((data_class_new[i,0]*h)**2*H))
    vb.append((1+z)*data_class_new[i,2]/((data_class_new[i,0]*h)**2*H))

# Data by the index
#0th: data_camb_sync[:,0] = 1:k(h/Mpc) #'k/h'
#1st: np.abs(data_camb_sync[:,1]) = 2:|-T_cdm/k2| = T_cdm/k2 # 'CDM' # Is this supposed to be probing the dmeff column or cdm?, where data_camb_sync[:,2] is the dmeff column
#2nd: np.abs(data_camb_sync[:,3]) = 4:|-T_b/k2| = T_b/k2 # 'baryon'
#3-5th: dummy variablesi # 'photon','nu','mass_nu'
#6th: np.abs(data_camb_sync[:,7] = 8:t_tot # 'total'
#7-9th: dummy variables 'no_nu','total_de','Weyl'
#10th: vc = (1+z)*data_class_new[i,3] (4:t_cdm)/((data_class_new[i,0]*h)**2*H) # 'v_CDM'
#11th: vb = (1+z)*data_class_new[i,2] (3:t_b)/((data_class_new[i,0]*h)**2*H) # 'v_b'
#12th: dummy variable 'v_b-v_c'

np.savetxt('CAMB_COZMIC_test_n2_1GeV_Tk.dat', np.column_stack((data_camb_sync[:,0], np.abs(data_camb_sync[:,1]), np.abs(data_camb_sync[:,3]),dummy,dummy,dummy,np.abs(data_camb_sync[:,7]),dummy,dummy,dummy,np.abs(vc),np.abs(vb),dummy)),fmt='%15.6e', header=header)
