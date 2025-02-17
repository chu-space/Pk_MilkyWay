from helpers.SimulationAnalysis import readHlist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import beyond_CDM_SHMF
import pickle
from sklearn.neighbors import KernelDensity
from scipy.integrate import trapezoid
import scipy

masses = {'wdm_3': 3., 'wdm_4': 4., 'wdm_5': 5., 'wdm_6': 6., 'wdm_6.5': 6.5, 'wdm_10': 10.,'cdm':5000.}
halo_nums = ['Halo004', 'Halo113', 'Halo023']
models = ['wdm_3', 'wdm_4', 'wdm_5', 'wdm_6', 'wdm_6.5', 'wdm_10', 'cdm']

# Base paths for COZMIC 1 host haloes
BASE_PATH = '/central/groups/carnegie_poc/enadler/ncdm_resims/'
# MCMC and simulation data
df_mcmc = pd.read_pickle('/central/groups/carnegie_poc/enadler/ncdm_resims/analysis/df_wdm.pkl')
with open("/central/groups/carnegie_poc/enadler/ncdm_resims/analysis/sim_data.bin", "rb") as f:
    sim_data = pickle.load(f, encoding='latin1')

def dNdlogMpeak(a,b,alpha,beta,gamma,x2,integrals,host,key):
    return (1.-np.array(integrals))*(a/100)*(10.**x2/(host['mvir'][0]/0.7))**(-1.*b)*np.exp(-50.*(10.**x2/(host['mvir'][0]/0.7))**(4.))*(1.+(alpha*Mhm(masses[key])/10.**x2)**beta)**(-1.*gamma)

def Mhm_k(khm):
    lambda_hm = 2*np.pi/khm
    return (4./3.)*np.pi*rho_crit*(lambda_hm/(2.*0.7))**3.

def Mhm(mwdm):
    return 4.3e8 * (mwdm/3.)**(-3.564)

def mwdm(Mhm):
    alpha = 0.049
    if mwdm >= 3.:
        alpha = 0.045
    elif mwdm > 6.:
        alpha = 0.043
    lambda_hm = 2.*(((3./4)*Mhm/(np.pi*rho_crit))**(1./3))
    lambda_fs = lambda_hm/13.93
    mwdm = (h*lambda_fs/(alpha*((omega_m/0.25)**(0.11))*((h/0.7)**1.22)))**(1./-1.11)
    return mwdm

def transfer(k,mwdm):
    nu = 1.12
    p = 0.049
    if mwdm >= 3.:
        p = 0.045
    elif mwdm > 6.:
        p = 0.043
    lambda_fs = (p*(mwdm**(-1.11))*((omega_m/0.25)**(0.11))*((h/0.7)**1.22))
    alpha = lambda_fs
    transfer = (1+(alpha*k)**(2*nu))**(-5./nu)
    return transfer

integrals = {}
integrals_coarse = {}
integrals_measured = {}

for num in halo_nums:
    integrals[num] = {}
    integrals_coarse[num] = {}
    integrals_measured[num] = {}
    for model in models:
        data = np.array([np.log10(sim_data[num][model][2]['Mpeak']/0.7),
                      np.log10(sim_data[num][model][2]['Mvir']/(0.7))]).T

        # Perform KDE
        kde = KernelDensity(bandwidth=0.2)
        kde.fit(data,sample_weight=np.log10(sim_data[num][model][2]['Mpeak']/(0.7)))

        # Define the x values for which you want to estimate the integral
        x_values = np.linspace(np.log10(1.2e8),np.log10(sim_data[num][model][0]['mvir'][0]/(2.*0.7)),100)
        x_values_coarse = np.linspace(np.log10(1.2e8),np.log10(sim_data[num][model][0]['mvir'][0]/(2.*0.7)),10)

        # Specify the y value up to which you want to integrate
        specified_y = np.log10(1.2e8)

        # Function to estimate the integral at a given x value
        def estimate_integral(x,data):
            if x > np.max(data):
                return 0.
            else:
                # Generate y values for the specified x
                y_values = np.linspace(min(data[:,1]), specified_y, 100)
                y_values_all = np.linspace(min(data[:, 1]), max(data[:, 1]), 100)

                # Compute the PDF values at each (x, y) pair
                xy_pairs = np.column_stack([np.full_like(y_values, x), y_values])
                log_pdf_values = kde.score_samples(xy_pairs)
                pdf_values = np.exp(log_pdf_values)

                # Integrate the PDF values along the y-axis
                integral = trapezoid(pdf_values, y_values)

                ###

                xy_pairs = np.column_stack([np.full_like(y_values_all, x), y_values_all])
                log_pdf_values = kde.score_samples(xy_pairs)
                pdf_values = np.exp(log_pdf_values)

                # Integrate the PDF values along the y-axis
                integral_tot = trapezoid(pdf_values, y_values_all)

                return integral/integral_tot

        # Estimate integrals for each x value
        ind = (sim_data[num][model][2]['Mvir']/0.7 > 1.2e8)
        integrals[num][model] = [estimate_integral(x,data) for x in x_values]
        integrals_coarse[num][model] = [estimate_integral(x,data) for x in x_values_coarse]
        integrals_measured[num][model] = [estimate_integral(x,data) for x in np.log10(sim_data[num][model][2]['Mpeak'][ind]/0.7)]

        # Example: Print estimated integral at x = 0
        x_index = np.abs(x_values - 9).argmin()

sim_all = {}
sim_all_diff = {}
pred_all = {}
pred_all_diff = {}
pred_all_diff_per_halo = {}
sim_all_diff_per_halo = {}

for model in models:
    sim_all[model] = []
    pred_all[model] = []
    sim_all_diff[model] = []
    pred_all_diff[model] = []
    pred_all_diff_per_halo[model] = {}
    sim_all_diff_per_halo[model] = {}
    for num in halo_nums:
        pred_all_diff_per_halo[model][num] = []
        sim_all_diff_per_halo[model][num] = []

for i,num in enumerate(halo_nums):
    ncdm_all = []
    ###
    plt.figure(figsize=(8,6))
    plt.yscale('log')
    for j,model in enumerate(models):
        pred = []
        diff = []
        for rand_int in np.random.randint(len(df_mcmc),size=500):
            if i == 0:
                a = df_mcmc['$a_1$'][rand_int]
                b = df_mcmc['$b_1$'][rand_int]
            elif i == 1:
                a = df_mcmc['$a_2$'][rand_int]
                b = df_mcmc['$b_2$'][rand_int]
            elif i == 2:
                a = df_mcmc['$a_3$'][rand_int]
                b = df_mcmc['$b_3$'][rand_int]
            ###
            x2 = np.logspace(np.log10(1.2e8),np.log10(sim_data[num][model][0]['mvir'][0]/(2.*0.7)),100)
            tot_model = dNdlogMpeak(a,b,df_mcmc[r'$\alpha$'][rand_int],df_mcmc[r'$\beta$'][rand_int],
                                    df_mcmc[r'$\gamma$'][rand_int],np.log10(x2),integrals[num][model],
                                    sim_data[num][model][0],model)
            x2 = np.log10(x2)
            tot = scipy.integrate.cumulative_trapezoid(tot_model,x2)
            ###
            diff.append(tot_model)
            pred.append(tot[-1]-tot)
            pred_all[model].append(tot[-1]-tot)
            ###
            x3 = np.logspace(np.log10(1.2e8),np.log10(sim_data[num][model][0]['mvir'][0]/(2.*0.7)),10)
            tot_model = dNdlogMpeak(a,b,df_mcmc[r'$\alpha$'][rand_int],df_mcmc[r'$\beta$'][rand_int],
                                    df_mcmc[r'$\gamma$'][rand_int],np.log10(x3),integrals_coarse[num][model],
                                    sim_data[num][model][0],model)
            pred_all_diff[model].append(tot_model)
            pred_all_diff_per_halo[model][num].append(tot_model)

halos_wdm3_004 = readHlist(BASE_PATH + halo_nums[0] + '/wdm_3/output_wdm_3/rockstar/hlists/hlist_1.00000.list')
host_wdm3_004 = halos_wdm3_004[halos_wdm3_004['id'] == 1889038]
subhalos_wdm3_004 = halos_wdm3_004[halos_wdm3_004['upid']==host_wdm3_004['id']]

halos_wdm3_113 = readHlist(BASE_PATH + halo_nums[1] + '/wdm_3/output_wdm_3/rockstar/hlists/hlist_1.00000.list')
host_wdm3_113 = halos_wdm3_113[halos_wdm3_113['id'] == 3365592]
subhalos_wdm3_113 = halos_wdm3_113[halos_wdm3_113['upid']==host_wdm3_113['id']]

halos_wdm3_023 = readHlist(BASE_PATH + halo_nums[2] + '/wdm_3/output_wdm_3/rockstar/hlists/hlist_1.00000.list')
host_wdm3_023 = halos_wdm3_023[halos_wdm3_023['id'] == 1801126]
subhalos_wdm3_023 = halos_wdm3_023[halos_wdm3_023['upid']==host_wdm3_023['id']]

halos_wdm6_5_004 = readHlist(BASE_PATH + halo_nums[0] + '/wdm_6.5/output_wdm_6.5/rockstar/hlists/hlist_1.00000.list')
host_wdm6_5_004 = halos_wdm6_5_004[halos_wdm6_5_004['id'] == 3748000]
subhalos_wdm6_5_004 = halos_wdm6_5_004[halos_wdm6_5_004['upid']==host_wdm6_5_004['id']]

halos_wdm6_5_113 = readHlist(BASE_PATH + halo_nums[1] + '/wdm_6.5/output_wdm_6.5/rockstar/hlists/hlist_1.00000.list')
host_wdm6_5_113 = halos_wdm6_5_113[halos_wdm6_5_113['id'] == 6642869]
subhalos_wdm6_5_113 = halos_wdm6_5_113[halos_wdm6_5_113['upid']==host_wdm6_5_113['id']]

halos_wdm6_5_023 = readHlist(BASE_PATH + halo_nums[2] + '/wdm_6.5/output_wdm_6.5/rockstar/hlists/hlist_1.00000.list')
host_wdm6_5_023 = halos_wdm6_5_023[halos_wdm6_5_023['id'] == 3689954]
subhalos_wdm6_5_023 = halos_wdm6_5_023[halos_wdm6_5_023['upid']==host_wdm6_5_023['id']]

# Constants: {Cuts, h_cdm, mass threshold}
analysis = 'Mpeak'
cut = 'Mvir'
h_cdm = 0.7
threshold = 1.2*10**8*h_cdm

# Recursion through each host halo can begin here:
# Filtering all subhaloes based on what we are analysing: Mpeak - the peak mass
mass_peak_sub_wdm3_004 = subhalos_wdm3_004[:][analysis]
mass_peak_sub_wdm3_113 = subhalos_wdm3_113[:][analysis]
mass_peak_sub_wdm3_023 = subhalos_wdm3_023[:][analysis]
mass_peak_sub_wdm6_5_004 = subhalos_wdm6_5_004[:][analysis]
mass_peak_sub_wdm6_5_113 = subhalos_wdm6_5_113[:][analysis]
mass_peak_sub_wdm6_5_023 = subhalos_wdm6_5_023[:][analysis]

# Making a cut based on: Mvir - virial mass
mass_subhalos_wdm3_Mvir_004 = subhalos_wdm3_004[:][cut]
mass_subhalos_wdm3_Mvir_113 = subhalos_wdm3_113[:][cut]
mass_subhalos_wdm3_Mvir_023 = subhalos_wdm3_023[:][cut]
mass_subhalos_wdm6_5_Mvir_004 = subhalos_wdm6_5_004[:][cut]
mass_subhalos_wdm6_5_Mvir_113 = subhalos_wdm6_5_113[:][cut]
mass_subhalos_wdm6_5_Mvir_023 = subhalos_wdm6_5_023[:][cut]

# Getting a binary condition if the mass of subhaloes surpass a threshold
cond_wdm3_Mvir_cut_004 = mass_subhalos_wdm3_Mvir_004 > threshold
cond_wdm3_Mvir_cut_113 = mass_subhalos_wdm3_Mvir_113 > threshold
cond_wdm3_Mvir_cut_023 = mass_subhalos_wdm3_Mvir_023 > threshold
cond_wdm6_5_Mvir_cut_004 = mass_subhalos_wdm6_5_Mvir_004 > threshold
cond_wdm6_5_Mvir_cut_113 = mass_subhalos_wdm6_5_Mvir_113 > threshold
cond_wdm6_5_Mvir_cut_023 = mass_subhalos_wdm6_5_Mvir_023 > threshold

# Applying the condition so that we only get the peak masses of subhaloes with a certain virial mass
mass3_peak_with_Mvir_cut_004 = mass_peak_sub_wdm3_004[cond_wdm3_Mvir_cut_004]
mass3_peak_with_Mvir_cut_113 = mass_peak_sub_wdm3_113[cond_wdm3_Mvir_cut_113]
mass3_peak_with_Mvir_cut_023 = mass_peak_sub_wdm3_023[cond_wdm3_Mvir_cut_023]
mass6_5_peak_with_Mvir_cut_004 = mass_peak_sub_wdm6_5_004[cond_wdm6_5_Mvir_cut_004]
mass6_5_peak_with_Mvir_cut_113 = mass_peak_sub_wdm6_5_113[cond_wdm6_5_Mvir_cut_113]
mass6_5_peak_with_Mvir_cut_023 = mass_peak_sub_wdm6_5_023[cond_wdm6_5_Mvir_cut_023]

# Ethan's mass log bin choice - upper limit is 0.5*M_vir,host
diff3_log_bins_004 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
diff3_log_bins_113 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_113[cut][0]/(2.*0.7)),10)
diff3_log_bins_023 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_023[cut][0]/(2.*0.7)),10)
diff6_5_log_bins_004 = np.logspace(np.log10(1.2e8),np.log10(host_wdm6_5_004[cut][0]/(2.*0.7)),10)
diff6_5_log_bins_113 = np.logspace(np.log10(1.2e8),np.log10(host_wdm6_5_113[cut][0]/(2.*0.7)),10)
diff6_5_log_bins_023 = np.logspace(np.log10(1.2e8),np.log10(host_wdm6_5_023[cut][0]/(2.*0.7)),10)

# Calculating differential SHMF
diff_hist_1_wdm3, test_1 = np.histogram(np.log10(mass3_peak_with_Mvir_cut_004/h_cdm), bins=np.log10(diff3_log_bins_004), density = False)
diff_hist_2_wdm3, test_2 = np.histogram(np.log10(mass3_peak_with_Mvir_cut_113/h_cdm), bins=np.log10(diff3_log_bins_113), density = False)
diff_hist_3_wdm3, test_3 = np.histogram(np.log10(mass3_peak_with_Mvir_cut_023/h_cdm), bins=np.log10(diff3_log_bins_023), density = False)
diff_hist_1_wdm6_5, test_4 = np.histogram(np.log10(mass6_5_peak_with_Mvir_cut_004/h_cdm), bins=np.log10(diff6_5_log_bins_004), density = False)
diff_hist_2_wdm6_5, test_5 = np.histogram(np.log10(mass6_5_peak_with_Mvir_cut_113/h_cdm), bins=np.log10(diff6_5_log_bins_113), density = False)
diff_hist_3_wdm6_5, test_6 = np.histogram(np.log10(mass6_5_peak_with_Mvir_cut_023/h_cdm), bins=np.log10(diff6_5_log_bins_023), density = False)


# Set figure size
fig = plt.figure(figsize=(16,12))

ax = fig.add_subplot(221)

ax.set_xscale('log')
ax.set_yscale('log')
diff_mean_wdm3 = np.mean([diff_hist_1_wdm3, diff_hist_2_wdm3, diff_hist_3_wdm3], axis=0)
diff_mean_wdm6_5 = np.mean([diff_hist_1_wdm6_5, diff_hist_2_wdm6_5, diff_hist_3_wdm6_5], axis=0)

# For n=2 envelope interacting dark matter models
idms = []
# for model in models:
#     if model == 'cdm':
#         continue
#     if 'n2' in model:
#         for method in ['envelope']:
x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)
            # idms.append(ax.errorbar(10.**(0.5*(x3[:-1]+x3[1:])),np.mean(sim_all_diff[model][method],axis=0),
            #              yerr=np.sqrt(np.mean(sim_all_diff[model][method],axis=0))/np.sqrt(3.),
            #                          linestyle='none',markersize=5,marker='o',capsize=3,color=sim_colors[model],
            #             label=labels_idm[model]))

# 3keV and 6.5keV warm dark matter models
wdms = []
# for model in ['wdm_6.5','wdm_3']:
#     if model == 'cdm':
#         continue
x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)
wdms.append(ax.plot(10.**x3, np.diff(x3)[0]*np.mean(pred_all_diff[models[0]],axis=0), label = '3keV WDM n=2 Envelope'))
x6_5 = np.logspace(np.log10(1.2e8),np.log10(host_wdm6_5_004[cut][0]/(2.*0.7)),10)
x6_5 = np.log10(x6_5)
wdms.append(ax.plot(10.**x6_5, np.diff(x6_5)[0]*np.mean(pred_all_diff[models[4]],axis=0), label = '6.5keV WDM N=2 Envelope'))

    # ax.fill_between(10.**x3,np.diff(x3)[0]*np.percentile(pred_all_diff[model],16,axis=0),
    #                  np.diff(x3)[0]*np.percentile(pred_all_diff[model],84,axis=0),
    #                     alpha=0.15,facecolor=sim_colors[model])

    # ax.fill_between(10.**x3,np.diff(x3)[0]*np.percentile(pred_all_diff[model],2.5,axis=0),
    #                  np.diff(x3)[0]*np.percentile(pred_all_diff[model],97.5,axis=0),
    #                     alpha=0.15,facecolor=sim_colors[model])

# first_legend = plt.legend(handles=[idms[0],idms[1],idms[2]], loc=1, fontsize=16, frameon=False)
# plt.gca().add_artist(first_legend)
# plt.legend(handles=[wdms[0][0],wdms[1][0]], loc='lower center', fontsize=16,frameon=False,framealpha=1.0,ncol=2)

x3_2=  [1.20000000e+08, 3.08183075e+08, 7.91473396e+08, 2.03265587e+09,
       5.22025114e+09, 1.34066087e+10, 3.44307492e+10, 8.84247846e+10,
       2.27091850e+11, 5.83215539e+11]
x3_3=  [1.03926452e+01, 2.10497140e+01, 1.90139712e+01, 1.18689033e+01,
 6.85807859e+00, 3.23047468e+00, 1.50502530e+00, 7.00359404e-01,
 3.03818389e-01, 6.73508442e-03]
x3_2_2=  [1.20000000e+08, 3.08183075e+08, 7.91473396e+08, 2.03265587e+09,
       5.22025114e+09, 1.34066087e+10, 3.44307492e+10, 8.84247846e+10,
       2.27091850e+11, 5.83215539e+11]
x3_3_2=  [2.86092682e+00, 6.80556244e+00, 8.24544935e+00, 8.48360507e+00,
 5.78084068e+00, 2.89849526e+00, 1.47082713e+00, 6.93985826e-01,
 3.02947662e-01, 6.72580219e-03]

wdms.append(ax.plot(np.array(x3_2), np.array(x3_3), label = 'COZMIC fit for 3keV', lw = 2))
wdms.append(ax.plot(np.array(x3_2_2), np.array(x3_3_2), label = 'COZMIC fit for 6.5keV', lw = 2))

ax.set_xlim(1e8,10**(10.5))
ax.set_ylim(1,40)

ax.set_xticks([1e8,1e9,1e10])
ax.set_xticklabels([r'$10^8$',r'$10^9$',r'$10^{10}$'],fontsize=20)
ax.set_yticks([1e0,1e1])
ax.set_yticklabels([r'$10^0$',r'$10^1$'],fontsize=20)
ax.text(10.**9.65,13.2,r'$n=2,\ \mathrm{envelope}$',fontsize=20)
ax.set_xlabel(r'$M_{\rm{sub,peak}}\ [M_{\mathrm{\odot}}]$',fontsize=26,labelpad=8)
ax.set_ylabel(r'$\mathrm{d}N/\mathrm{d}\log M_{\mathrm{sub,peak}}$', fontsize=26, labelpad=12)
ax.legend()
###

ax = fig.add_subplot(222)

ax.set_xscale('log')
ax.set_yscale('log')

# Interacting dark matter for n=2 half mode
idms = []
# for model in models:
#     if model == 'cdm':
#         continue
#     if 'n2' in model:
#         for method in ['halfmode']:
x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)
    # idms.append(ax.errorbar(10.**(0.5*(x3[:-1]+x3[1:])),np.mean(sim_all_diff[model][method],axis=0),
    #                      yerr=np.sqrt(np.mean(sim_all_diff[model][method],axis=0))/np.sqrt(3.),
    #                                  linestyle='none',markersize=5,marker='o',capsize=3,color=sim_colors[model],
    #                     label=labels_idm[model]))

# 3keV and 6.5keV warm dark matter models
# wdms = []
# for model in ['wdm_6.5','wdm_3']:
#     if model == 'cdm':
#         continue
x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)
wdms.append(ax.plot(10.**x3, np.diff(x3)[0]*np.mean(pred_all_diff[models[0]],axis=0), label = '3keV WDM N=2 Half-Mode'))
x6_5 = np.logspace(np.log10(1.2e8),np.log10(host_wdm6_5_004[cut][0]/(2.*0.7)),10)
x6_5 = np.log10(x6_5)
wdms.append(ax.plot(10.**x6_5, np.diff(x6_5)[0]*np.mean(pred_all_diff[models[4]],axis=0), label = '6.5keV WDM N=2 Half-Mode'))

    # ax.fill_between(10.**x3,np.diff(x3)[0]*np.percentile(pred_all_diff[model],16,axis=0),
    #                  np.diff(x3)[0]*np.percentile(pred_all_diff[model],84,axis=0),
    #                     alpha=0.15,facecolor=sim_colors[model])

    # ax.fill_between(10.**x3,np.diff(x3)[0]*np.percentile(pred_all_diff[model],2.5,axis=0),
    #                  np.diff(x3)[0]*np.percentile(pred_all_diff[model],97.5,axis=0),
    #                     alpha=0.15,facecolor=sim_colors[model])

x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)

# first_legend = plt.legend(handles=[idms[0],idms[1],idms[2]], loc=1, fontsize=16, frameon=False)
# plt.gca().add_artist(first_legend)
# plt.legend(handles=[wdms[0][0],wdms[1][0]], loc='lower center', fontsize=16,frameon=False,framealpha=1.0,ncol=2)

ax.set_xlim(1e8,10**(10.5))
ax.set_ylim(1,40)

ax.set_xticks([1e8,1e9,1e10])
ax.set_xticklabels([r'$10^8$',r'$10^9$',r'$10^{10}$'],fontsize=20)
ax.set_yticks([1e0,1e1])
ax.set_yticklabels([r'$10^0$',r'$10^1$'],fontsize=20)
ax.text(10.**9.65,13.2,r'$n=2,\ \mathrm{half}$-$\mathrm{mode}$',fontsize=20)
ax.set_xlabel(r'$M_{\rm{sub,peak}}\ [M_{\mathrm{\odot}}]$',fontsize=26,labelpad=8)
ax.set_ylabel(r'$\mathrm{d}N/\mathrm{d}\log M_{\mathrm{sub,peak}}$', fontsize=26, labelpad=12)

### n = 4 envelope

ax = fig.add_subplot(223)

ax.set_xscale('log')
ax.set_yscale('log')

idms = []
# for model in models:
#     if model == 'cdm':
#         continue
#     if 'n2' in model:
#         continue
#     for method in ['envelope']:
x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)
        # idms.append(ax.errorbar(10.**(0.5*(x3[:-1]+x3[1:])),np.mean(sim_all_diff[model][method],axis=0),
        #              yerr=np.sqrt(np.mean(sim_all_diff[model][method],axis=0))/np.sqrt(3.),
        #                          linestyle='none',markersize=5,marker='o',capsize=3,color=sim_colors[model],
        #             label=labels_idm[model]))

wdms = []
# for model in ['wdm_6.5','wdm_3']:
#     if model == 'cdm':
#         continue
x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)
wdms.append(ax.plot(10.**x3, np.diff(x3)[0]*np.mean(pred_all_diff[models[0]],axis=0), label = '3keV WDM N=4 Envelope'))
x6_5 = np.logspace(np.log10(1.2e8),np.log10(host_wdm6_5_004[cut][0]/(2.*0.7)),10)
x6_5 = np.log10(x6_5)
wdms.append(ax.plot(10.**x6_5, np.diff(x6_5)[0]*np.mean(pred_all_diff[models[4]],axis=0), label = '6.5keV WDM N=4 Envelope'))

    # ax.fill_between(10.**x3,np.diff(x3)[0]*np.percentile(pred_all_diff[model],16,axis=0),
    #                  np.diff(x3)[0]*np.percentile(pred_all_diff[model],84,axis=0),
    #                     alpha=0.15,facecolor=sim_colors[model])

    # ax.fill_between(10.**x3,np.diff(x3)[0]*np.percentile(pred_all_diff[model],2.5,axis=0),
    #                  np.diff(x3)[0]*np.percentile(pred_all_diff[model],97.5,axis=0),
    #                     alpha=0.15,facecolor=sim_colors[model])

x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)


# first_legend = plt.legend(handles=[idms[0],idms[1],idms[2]], loc=1, fontsize=16, frameon=False)
# plt.gca().add_artist(first_legend)
# plt.legend(handles=[wdms[0][0],wdms[1][0]], loc='lower center', fontsize=16,frameon=False,framealpha=1.0,ncol=2)

ax.set_xlim(1e8,10**(10.5))
ax.set_ylim(1,40)

ax.set_xticks([1e8,1e9,1e10])
ax.set_xticklabels([r'$10^8$',r'$10^9$',r'$10^{10}$'],fontsize=20)
ax.set_yticks([1e0,1e1])
ax.set_yticklabels([r'$10^0$',r'$10^1$'],fontsize=20)
ax.text(10.**9.65,13.2,r'$n=4,\ \mathrm{envelope}$',fontsize=20)
ax.set_xlabel(r'$M_{\rm{sub,peak}}\ [M_{\mathrm{\odot}}]$',fontsize=26,labelpad=8)
ax.set_ylabel(r'$\mathrm{d}N/\mathrm{d}\log M_{\mathrm{sub,peak}}$', fontsize=26, labelpad=12)

### N = 4 half-mode

ax = fig.add_subplot(224)

ax.set_xscale('log')
ax.set_yscale('log')

idms = []
# for model in models:
#     if model == 'cdm':
#         continue
#     if 'n2' in model:
#         continue
#     for method in ['halfmode']:
x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)
        # idms.append(ax.errorbar(10.**(0.5*(x3[:-1]+x3[1:])),np.mean(sim_all_diff[model][method],axis=0),
        #              yerr=np.sqrt(np.mean(sim_all_diff[model][method],axis=0))/np.sqrt(3.),
        #                          linestyle='none',markersize=5,marker='o',capsize=3,color=sim_colors[model],
        #             label=labels_idm[model]))

wdms = []
# for model in ['wdm_6.5','wdm_3']:
#     if model == 'cdm':
#         continue
x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)
wdms.append(ax.plot(10.**x3, np.diff(x3)[0]*np.mean(pred_all_diff[models[0]],axis=0), label = '3keV WDM N=4 Half-Mode'))
x6_5 = np.logspace(np.log10(1.2e8),np.log10(host_wdm6_5_004[cut][0]/(2.*0.7)),10)
x6_5 = np.log10(x6_5)
wdms.append(ax.plot(10.**x6_5, np.diff(x6_5)[0]*np.mean(pred_all_diff[models[4]],axis=0), label = '6.5keV WDM N=4 Half-Mode'))
    # ax.fill_between(10.**x3,np.diff(x3)[0]*np.percentile(pred_all_diff[model],16,axis=0),
    #                  np.diff(x3)[0]*np.percentile(pred_all_diff[model],84,axis=0),
    #                     alpha=0.15,facecolor=sim_colors[model])

    # ax.fill_between(10.**x3,np.diff(x3)[0]*np.percentile(pred_all_diff[model],2.5,axis=0),
    #                  np.diff(x3)[0]*np.percentile(pred_all_diff[model],97.5,axis=0),
    #                     alpha=0.15,facecolor=sim_colors[model])

x3 = np.logspace(np.log10(1.2e8),np.log10(host_wdm3_004[cut][0]/(2.*0.7)),10)
x3 = np.log10(x3)


# first_legend = plt.legend(handles=[idms[0],idms[1],idms[2]], loc=1, fontsize=16, frameon=False)
# plt.gca().add_artist(first_legend)
# plt.legend(handles=[wdms[0][0],wdms[1][0]], loc='lower center', fontsize=16,frameon=False,framealpha=1.0,ncol=2)

ax.set_xlim(1e8,10**(10.5))
ax.set_ylim(1,40)

ax.set_xticks([1e8,1e9,1e10])
ax.set_xticklabels([r'$10^8$',r'$10^9$',r'$10^{10}$'],fontsize=20)
ax.set_yticks([1e0,1e1])
ax.set_yticklabels([r'$10^0$',r'$10^1$'],fontsize=20)
ax.text(10.**9.65,13.2,r'$n=4,\ \mathrm{half}$-$\mathrm{mode}$',fontsize=20)
ax.set_xlabel(r'$M_{\rm{sub,peak}}\ [M_{\mathrm{\odot}}]$',fontsize=26,labelpad=8)
ax.set_ylabel(r'$\mathrm{d}N/\mathrm{d}\log M_{\mathrm{sub,peak}}$', fontsize=26, labelpad=12)

###
plt.tight_layout()
plt.subplots_adjust(wspace = 0.2)
plt.savefig('IDM_SHMF_Suppresion_plot')
