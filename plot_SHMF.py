from helpers.SimulationAnalysis import readHlist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import beyond_CDM_SHMF

# Base paths for COZMIC 1 host haloes
BASE_PATH_HALO4 = '/central/groups/carnegie_poc/enadler/ncdm_resims/Halo004/'
BASE_PATH_HALO113 = '/central/groups/carnegie_poc/enadler/ncdm_resims/Halo113/'
BASE_PATH_HALO023 = '/central/groups/carnegie_poc/enadler/ncdm_resims/Halo023/'

# Accessing the three host halos from the COZMIC 1 suite of simulations
HALO_LIST = {'Halo004', 'Halo113', 'Halo23'}
HALO_TYPES = {'wdm_3', 'wdm_4', 'wdm_5', 'wdm_6', 'wdm_6.5', 'wdm_10'}
HALO_IDS = {1889038, 3365592, 1801126}

#take in a list of halos, then for each halo go through each id

halos_wdm3_004 = readHlist(BASE_PATH_HALO4 + 'wdm_3/output_wdm_3/rockstar/hlists/hlist_1.00000.list')
host_wdm3_004 = halos_wdm3_004[halos_wdm3_004['id'] == 1889038]
subhalos_wdm3_004 = halos_wdm3_004[halos_wdm3_004['upid']==host_wdm3_004['id']]

halos_wdm3_113 = readHlist(BASE_PATH_HALO113 + 'wdm_3/output_wdm_3/rockstar/hlists/hlist_1.00000.list')
host_wdm3_113 = halos_wdm3_113[halos_wdm3_113['id'] == 3365592]
subhalos_wdm3_113 = halos_wdm3_113[halos_wdm3_113['upid']==host_wdm3_113['id']]

halos_wdm3_023 = readHlist(BASE_PATH_HALO023 + 'wdm_3/output_wdm_3/rockstar/hlists/hlist_1.00000.list')
host_wdm3_023 = halos_wdm3_023[halos_wdm3_023['id'] == 1801126]
subhalos_wdm3_023 = halos_wdm3_023[halos_wdm3_023['upid']==host_wdm3_023['id']]

halos_wdm4 = readHlist(BASE_PATH_HALO4 + 'wdm_4/output_wdm_4/rockstar/hlists/hlist_1.00000.list')
host_wdm4 = halos_wdm4[halos_wdm4['id'] == 2474249]
subhalos_wdm4 = halos_wdm4[halos_wdm4['upid']==host_wdm4['id']]

halos_wdm5 = readHlist(BASE_PATH_HALO4 + 'wdm_5/output_wdm_5/rockstar/hlists/hlist_1.00000.list')
host_wdm5 = halos_wdm5[halos_wdm5['id'] == 2992840]
subhalos_wdm5 = halos_wdm5[halos_wdm5['upid']==host_wdm5['id']]

halos_wdm6 = readHlist(BASE_PATH_HALO4 + 'wdm_6/output_wdm_6/rockstar/hlists/hlist_1.00000.list')
host_wdm6 = halos_wdm6[halos_wdm6['id'] == 3498758]
subhalos_wdm6 = halos_wdm6[halos_wdm6['upid']==host_wdm6['id']]

halos_wdm6_5 = readHlist(BASE_PATH_HALO4 + 'wdm_6.5/output_wdm_6.5/rockstar/hlists/hlist_1.00000.list')
host_wdm6_5 = halos_wdm6_5[halos_wdm6_5['id'] == 3748000]
subhalos_wdm6_5 = halos_wdm6_5[halos_wdm6_5['upid']==host_wdm6_5['id']]

halos_wdm10 = readHlist(BASE_PATH_HALO4 + 'wdm_10/output_wdm_10/rockstar/hlists/hlist_1.00000.list')
host_wdm10 = halos_wdm10[halos_wdm10['id'] == 5142328]
subhalos_wdm10 = halos_wdm10[halos_wdm10['upid']==host_wdm10['id']]

# Constants: {Cuts, h_cdm, mass threshold}
analysis = 'Mpeak'
cut = 'Mvir'
h_cdm = 0.7
threshold = 1.2*10**8*h_cdm

# Filtering all subhaloes based on what we are analysing: Mpeak - the peak mass
mass_peak_sub_wdm3_004 = subhalos_wdm3_004[:][analysis]
mass_peak_sub_wdm3_113 = subhalos_wdm3_113[:][analysis]
mass_peak_sub_wdm3_023 = subhalos_wdm3_023[:][analysis]

# Making a cut based on: Mvir - virial mass
mass_subhalos_wdm3_Mvir_004 = subhalos_wdm3_004[:][cut]
mass_subhalos_wdm3_Mvir_113 = subhalos_wdm3_113[:][cut]
mass_subhalos_wdm3_Mvir_023 = subhalos_wdm3_023[:][cut]

# Getting a binary condition if the mass of subhaloes surpass a threshold
cond_wdm3_Mvir_cut_004 = mass_subhalos_wdm3_Mvir_004 > threshold
cond_wdm3_Mvir_cut_113 = mass_subhalos_wdm3_Mvir_113 > threshold
cond_wdm3_Mvir_cut_023 = mass_subhalos_wdm3_Mvir_023 > threshold

# Applying the condition so that we only get the peak masses of subhaloes with a certain virial mass
mass_peak_with_Mvir_cut_004 = mass_peak_sub_wdm3_004[cond_wdm3_Mvir_cut_004]
mass_peak_with_Mvir_cut_113 = mass_peak_sub_wdm3_113[cond_wdm3_Mvir_cut_113]
mass_peak_with_Mvir_cut_023 = mass_peak_sub_wdm3_023[cond_wdm3_Mvir_cut_023]

# Define mass bins
log_bins = np.linspace(7, 12, 10)
bins = 10**log_bins

# Ethan's mass log bin choice - upper limit is 0.5*M_vir,host
diff_log_bins_004 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
diff_log_bins_113 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_113[cut][0]/(2.*0.7)),10)
diff_log_bins_023 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_023[cut][0]/(2.*0.7)),10)

# Calculating differential SHMF
diff_hist_1, test_1 = np.histogram(np.log10(mass_peak_with_Mvir_cut_004/h_cdm), bins=np.log10(diff_log_bins_004), density = False)
diff_hist_2, test_2 = np.histogram(np.log10(mass_peak_with_Mvir_cut_113/h_cdm), bins=np.log10(diff_log_bins_113), density = False)
diff_hist_3, test_3 = np.histogram(np.log10(mass_peak_with_Mvir_cut_023/h_cdm), bins=np.log10(diff_log_bins_023), density = False)

# Differential SHMF Mean
diff_mean = np.mean([diff_hist_1, diff_hist_2, diff_hist_3], axis=0)
diff_mean_log_bin = np.mean([diff_log_bins_004, diff_log_bins_113, diff_log_bins_023], axis = 0)
diff_bin_centers = 0.5 * (diff_mean_log_bin[1:] + diff_mean_log_bin[:-1])


test_mean_bins = np.mean([test_1, test_2, test_3], axis=0)
print('test_mean_bins: ', 10**diff_log_bins_023[1:])

# Cummulative log bin choice
cumm_log_bins = np.linspace(7.5,11,10)
cumm_bins = 10**cumm_log_bins
cumm_bin_centers = cumm_bins[1:]

cumm_hist_1, cumm_base_1 = np.histogram(mass_peak_with_Mvir_cut_004/h_cdm, bins=cumm_bins)
cumm_hist_2, cumm_base_2 = np.histogram(mass_peak_with_Mvir_cut_113/h_cdm, bins=cumm_bins)
cumm_hist_3, cumm_base_3 = np.histogram(mass_peak_with_Mvir_cut_023/h_cdm, bins=cumm_bins)

# Cumulative SHMF
# Instead of reverse cumulative sum, subtract the length of the array from the histogram
# How many subhaloes meet the cut
# CDM8K_cumulative = np.cumsum(CDM8K_values)
# len(sim_data[halo_num]['cdm'][2]['Mpeak'][ind])
# -CDM8K_cumulative histogram

#     CDM8K_values, CDM8K_base = np.histogram(np.log10(sim_data[halo_num]['cdm'][2]['Mpeak'][ind]/0.7), bins=bins)
#     CDM8K_cumulative = np.cumsum(CDM8K_values)
#     cdm_shmf.append(len(sim_data[halo_num]['cdm'][2]['Mpeak'][ind])-CDM8K_cumulative)

# cdm_shmf = np.array(cdm_shmf)

# cdm = ax.plot(CDM8K_base[1:], np.mean(cdm_shmf,axis=0),
#            'k',ls=':',lw=2,label='$\mathrm{CDM}$')

total_cumm_mean = np.mean([cumm_hist_1, cumm_hist_2, cumm_hist_3], axis=0)
cumm_mean_reverse = np.cumsum(total_cumm_mean[::-1])[::-1]
print('cumm_mean_reverse: ', cumm_mean_reverse)

# COZMIC 1 paper data
y_cumm = [37.        , 36.33333333, 29.33333333, 19.33333333, 12.66666667,
         6.33333333,  3.        ,  0.33333333,  0.33333333]
# x_cumm matches
x_cumm = [7.74263683e+07, 1.89573565e+08, 4.64158883e+08, 1.13646367e+09,
       2.78255940e+09, 6.81292069e+09, 1.66810054e+10, 4.08423865e+10,
       1.00000000e+11]

# diff mean matches
y_diff = [3.33333333, 11.        ,  8.        ,  6.33333333,  5.33333333,
         2.        ,  0.66666667,  0.33333333,  0.        ]
# NOTE: USES HOST023 DATA
x_diff = [1.92306942e+08, 4.93881087e+08, 1.26838130e+09, 3.25744628e+09,
       8.36574640e+09, 2.14848402e+10, 5.51771877e+10, 1.41705594e+11,
       3.63927126e+11]

# Next steps:
# Continue to work on adjusting the cumulative plot
# Plot Figure 11
# Streamline pipeline for all of the DM models

# Plot results
plt.figure(figsize=(10, 6))

# Differential plot
ax1 = plt.subplot(1, 2, 1)
ax1.plot(10**diff_log_bins_023[1:], diff_mean, label="Differential SHMF")
ax1.plot(x_diff, y_diff, label="Cozmic Differential")
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$M_{\mathrm{sub}}$')
ax1.set_ylabel(r'$dN/dM$')
ax1.legend()

print('cumm_bin_centers: ', cumm_bin_centers)
print('cumulative sum: ', np.cumsum(diff_mean[::-1])[::-1])
print('x_cumm: ', x_cumm)
print('y_cumm: ', y_cumm)

# Cumulative plot
ax2 = plt.subplot(1, 2, 2)
#ax2.plot(cumm_bin_centers, cumm_mean_reverse, label="Cumulative SHMF")
ax2.plot(cumm_bin_centers, np.cumsum(diff_mean[::-1])[::-1], label="Differential SHMF Cumulative Summed")
ax2.plot(x_cumm,y_cumm, label="Cozmic Cumulative")

# Cumulative plot style
ax2.set_xlim(10**8.0,10**10.5)
ax2.set_ylim([1,150])
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_yticks([1,3,10,30,100])
ax2.set_yticklabels([r'$1$',r'$3$',r'$10$',r'$30$',r'$100$'],fontsize=17)
ax2.set_xlabel(r'$M_{\mathrm{sub}}$')
ax2.set_ylabel(r'$N(>M)$')
ax2.legend()

plt.tight_layout()
plt.show()
plt.savefig('Diff_Cumm_SHMF_plot.png')
