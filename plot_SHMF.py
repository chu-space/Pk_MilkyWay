from helpers.SimulationAnalysis import readHlist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import beyond_CDM_SHMF

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
mass_peak_sub_wdm3_113 = subhalos_wdm3_113[:][analysis]
mass_peak_sub_wdm3_023 = subhalos_wdm3_023[:][analysis]

#cond_wdm3_004 = mass_sub_wdm3_004 > threshold
#cond_wdm3_113 = mass_sub_wdm3_113 > threshold_1
#cond_wdm3_023 = mass_sub_wdm3_023 > threshold_1

#mean_count_wdm3 = (count_wdm3_004+count_wdm3_113+count_wdm3_023)/3
#print("Number of averaged subhaloes masses of 3keV WDM that surpassed the " + str(analysis) + " threshold " + str(threshold) + " is: " + str(mean_count_wdm3))

# Filter subhaloes
mass_subhalos_wdm3_Mvir_004 = subhalos_wdm3_004[:][cut]
mass_subhalos_wdm3_Mvir_113 = subhalos_wdm3_113[:][cut]
mass_subhalos_wdm3_Mvir_023 = subhalos_wdm3_023[:][cut]

cond_wdm3_Mvir_cut_004 = mass_subhalos_wdm3_Mvir_004 > threshold
mass_peak_with_Mvir_cut_004 = mass_peak_sub_wdm3_004[cond_wdm3_Mvir_cut_004]
cond_wdm3_Mvir_cut_113 = mass_subhalos_wdm3_Mvir_113 > threshold
mass_peak_with_Mvir_cut_113 = mass_peak_sub_wdm3_113[cond_wdm3_Mvir_cut_113]
cond_wdm3_Mvir_cut_023 = mass_subhalos_wdm3_Mvir_023 > threshold
mass_peak_with_Mvir_cut_023 = mass_peak_sub_wdm3_023[cond_wdm3_Mvir_cut_023]

# Differential SHMF
df1 = DataFrame(mass_peak_with_Mvir_cut_004)
df2 = DataFrame(mass_peak_with_Mvir_cut_113)
df3 = DataFrame(mass_peak_with_Mvir_cut_023)
df_merged = pd.concat([df1, df2, df3])
by_row_index = df_merged.groupby(df_merged.index)
df_merged_mean = by_row_index.mean()

df_merged_Mvir = pd.concat([DataFrame(mass_subhalos_wdm3_Mvir_004), DataFrame(mass_subhalos_wdm3_Mvir_113), DataFrame(mass_subhalos_wdm3_Mvir_023)])
#print(df_merged_Mvir[0][0][0])
# Define mass bins (log scale between 10**7 to 11)
#log_bins = np.linspace(7, 12, 12)
log_bins = np.linspace(7, 12, 10)
bins = 10**log_bins
#upper limit is 1/2 mvir host
diff_log_bins = np.logspace(np.log10(1.2e8),np.log10(1.2e11),10)
#diff_bins = 10**diff_log_bins
diff_bin_centers = 0.5 * (diff_log_bins[1:] + diff_log_bins[:-1])

x3 = np.log10(diff_log_bins)

print('diff_log_bins: ', diff_log_bins)
print('df_merged_mean.values: ', df_merged_mean.values)
diff_hist, _ = np.histogram(df_merged_mean.values, bins=bins)
differential_shmf = diff_hist / (np.diff(log_bins))  # Normalize by bin width (log space)

cumm_log_bins = np.linspace(7.5,11,10)
cumm_bins = 10**cumm_log_bins
cumm_bin_centers = 0.5 * (cumm_bins[1:] + cumm_bins[:-1])

cumm_hist, _ = np.histogram(df_merged_mean.values, bins=cumm_bins)

# Cumulative SHMF
cumulative_shmf = np.cumsum(cumm_hist[::-1])[::-1]  # Reverse cumulative sum
print("cumulative_shmf: ", cumulative_shmf)
# Differential WDM SHMF
beyond_CDM_SHMF.f_beyond_CDM(df_merged.values, 10, 2.5, 0.9, 1.0)

x_cumm = [102562564.77081062, 189508862.66108668, 466442535.165321, 887775605.6633633, 1679198194.49203, 2780024594.2117653, 6724184096.911058, 16411385971.133947, 25370856514.077957]
y_cumm = [36.37097470752195, 36.63411357730507, 29.065641353815458, 21.70185477434134, 15.994894342266116, 12.61964315649359, 6.319753166844106, 3.011238449807771, 1.015283760417585]

y_diff = [37.56934306569343, 36.819951338199516, 25.83698296836983, 18.54501216545012, 9.722627737226276]
x_diff = [10**8,
2362256814.4919534,
12646716360.5118,
16218909592.18256,
19791102823.853317]

# Plot results
plt.figure(figsize=(10, 6))

# Differential plot
plt.subplot(1, 2, 1)
plt.plot(diff_bin_centers, differential_shmf, label="Differential SHMF")
plt.plot(x_cumm,y_cumm, label="Cozmic Differential")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\mathrm{sub}}$')
plt.ylabel(r'$dN/dM$')
plt.legend()

# Cumulative plot
plt.subplot(1, 2, 2)
plt.plot(cumm_bin_centers, cumulative_shmf, label="Cumulative SHMF")
plt.plot(x_diff,y_diff, label="Cozmic Cumulative")
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
