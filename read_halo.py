from helpers.SimulationAnalysis import readHlist
import numpy as np

BASE_PATH_HALO4 = '/central/groups/carnegie_poc/enadler/ncdm_resims/Halo004/'
halos_wdm3 = readHlist(BASE_PATH_HALO4 + 'wdm_3/output_wdm_3/rockstar/hlists/hlist_1.00000.list')
host_wdm3 = halos_wdm3[halos_wdm3['id'] == 1889038]
subhalos_wdm3 = halos_wdm3[halos_wdm3['upid']==host_wdm3['id']]

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

analysis = 'Mvir'
count_wdm3 = 0
h_cdm = 0.7
threshold = 1.2*10**8*h_cdm

mass_sub_wdm3 = subhalos_wdm3[:][analysis]
cond_wdm3 = mass_sub_wdm3 > threshold
count_wdm3 = sum(cond_wdm3)
print("Number of subhaloes of 3keV WDM that surpassed the " + str(analysis) + " threshold " + str(threshold) + " is: " + str(count_wdm3))

count_wdm4 = 0
mass_sub_wdm4 = subhalos_wdm4[:][analysis]
cond_wdm4 = mass_sub_wdm4 > threshold
count_wdm4 = sum(cond_wdm4)
print("Number of subhaloes of 4keV WDM that surpassed the " + str(analysis) + " threshold " + str(threshold) + " is: " + str(count_wdm4))

count_wdm5 = 0
mass_sub_wdm5 = subhalos_wdm5[:][analysis]
cond_wdm5 = mass_sub_wdm5 > threshold
count_wdm5 = sum(cond_wdm5)
print("Number of subhaloes of 5keV WDM that surpassed the " + str(analysis) + " threshold " + str(threshold) + " is: " + str(count_wdm5))

count_wdm6 = 0
mass_sub_wdm6 = subhalos_wdm6[:][analysis]
cond_wdm6 = mass_sub_wdm6 > threshold
count_wdm6 = sum(cond_wdm6)
print("Number of subhaloes of 6keV WDM that surpassed the " + str(analysis) + " threshold " + str(threshold) + " is: " + str(count_wdm6))

count_wdm6_5 = 0
mass_sub_wdm6_5 = subhalos_wdm6_5[:][analysis]
cond_wdm6_5 = mass_sub_wdm6_5 > threshold
count_wdm6_5 = sum(cond_wdm6_5)
print("Number of subhaloes of 6.5keV WDM that surpassed the " + str(analysis) + " threshold " + str(threshold) + " is: " + str(count_wdm6_5))


