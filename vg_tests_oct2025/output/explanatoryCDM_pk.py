import matplotlib.pyplot as plt
import numpy as np
import itertools

files = ['/Users/veragluscevic/research/repositories/arif/Pk_MilkyWay/vg_tests_oct2025/output/explanatoryCDM_pk.dat', '/Users/veragluscevic/research/repositories/arif/Pk_MilkyWay/vg_tests_oct2025/output/n2_1e-2GeV_7.1e-24_pk.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['explanatoryCDM_pk', 'n2_1e-2GeV_7']

fig, ax = plt.subplots()
y_axis = ['P(Mpc/h)^3']
tex_names = ['P (Mpc/h)^3']
x_axis = 'k (h/Mpc)'
ax.set_xlabel('k (h/Mpc)', fontsize=16)
plt.show()