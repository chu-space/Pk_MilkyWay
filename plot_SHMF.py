from helpers.SimulationAnalysis import readHlist
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH_HALO4 = '/central/groups/carnegie_poc/enadler/ncdm_resims/Halo004/'
BASE_PATH_HALO113 = '/central/groups/carnegie_poc/enadler/ncdm_resims/Halo113/'
BASE_PATH_HALO023 = '/central/groups/carnegie_poc/enadler/ncdm_resims/Halo023/'


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

analysis = 'Mpeak'
cut = 'Mvir'
h_cdm = 0.7
threshold = 1.2*10**8*h_cdm

mass_peak_sub_wdm3_004 = subhalos_wdm3_004[:][analysis]
#mass_peak_sub_wdm3_113 = subhalos_wdm3_113[:][analysis]
#mass_peak_sub_wdm3_023 = subhalos_wdm3_023[:][analysis]

#cond_wdm3_004 = mass_sub_wdm3_004 > threshold
#cond_wdm3_113 = mass_sub_wdm3_113 > threshold_1
#cond_wdm3_023 = mass_sub_wdm3_023 > threshold_1

#mean_count_wdm3 = (count_wdm3_004+count_wdm3_113+count_wdm3_023)/3
#print("Number of averaged subhaloes masses of 3keV WDM that surpassed the " + str(analysis) + " threshold " + str(threshold) + " is: " + str(mean_count_wdm3))

# Filter subhaloes
cond_wdm3_Mvir_cut = subhalos_wdm3_004[:][cut] > threshold
mass_peak_with_Mvir_cut = mass_peak_sub_wdm3_004[cond_wdm3_Mvir_cut]

# Define mass bins (log scale)
log_bins = np.linspace(7, 11, 10)
bins = 10**log_bins
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Differential SHMF
hist, _ = np.histogram(mass_peak_with_Mvir_cut, bins=bins)
differential_shmf = hist / np.diff(log_bins)  # Normalize by bin width (log space)

# Cumulative SHMF
cumulative_shmf = np.cumsum(hist[::-1])[::-1]  # Reverse cumulative sum

# Plot results
plt.figure(figsize=(10, 6))

# Differential plot
plt.subplot(1, 2, 1)
plt.plot(bin_centers, differential_shmf, label="Differential SHMF")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\mathrm{sub}}$')
plt.ylabel(r'$dN/dM$')
plt.legend()

# Cumulative plot
plt.subplot(1, 2, 2)
plt.plot(bin_centers, cumulative_shmf, label="Cumulative SHMF")
plt.xscale('log')
plt.xlabel(r'$M_{\mathrm{sub}}$')
plt.ylabel(r'$N(>M)$')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('Diff_Cumm_SHMF_plot.png')

#f = plt.figure()
#plt.plot(range(10**8, 10**11,10), count_wdm3)
#plt.show()
#f.savefig("SHMF_WDM.pdf", bbox_inches='tight')

