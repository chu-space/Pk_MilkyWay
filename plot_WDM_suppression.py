import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base_mpeak = np.logspace(np.log10(3e5*300/0.7),np.log10(9e5*1.3e5/0.7),100)
x=0.5*(base_mpeak[1:]+base_mpeak[:-1])
this_work = plt.plot(base_mpeak[:-1],-1.*np.ones(len(base_mpeak[:-1])),label=r'$\mathrm{This\ work}$', c = "orange")
lovell = plt.plot(base_mpeak[:-1],-1.*np.ones(len(base_mpeak[:-1])),lw=1.5,ls='--', label=r'$\mathrm{Lovell\ et\ al.\ (2014)}$', c = "blue")

def Mhm(mwdm):
    return 4.3e8 * (mwdm/3.)**(-3.564)

first_legend = plt.legend(handles=[this_work[0],lovell[0]],ncol=2,fontsize=16,frameon=False,loc=1)
plt.gca().add_artist(first_legend)

df_mcmc = pd.read_pickle('/central/groups/carnegie_poc/enadler/ncdm_resims/analysis/df_wdm.pkl')
pred_dict = {}
wdm_plot = []
pred = []
for rand_int in np.random.randint(len(df_mcmc),size=500):
    pred.append((1.+(df_mcmc[r"$\alpha$"][rand_int]*Mhm(3.)/x)**df_mcmc[r"$\beta$"][rand_int])**(-1.*df_mcmc[r"$\gamma$"][rand_int]))
#pred_dict[model]
pred_dict[0] = np.mean(pred,axis=0)
wdm_plot.append(plt.plot(base_mpeak[:-1],np.mean(pred,axis=0),lw=2, label='WDM 3.0keV', c = "orange"))

#plt.plot(base_mpeak[:-1],np.mean(pred,axis=0),lw=2)
    # plt.fill_between(base_mpeak[:-1],np.percentile(pred,16,axis=0),np.percentile(pred,84,axis=0),
    #                 alpha=0.15,facecolor=sim_colors[model])
    # plt.fill_between(base_mpeak[:-1],np.percentile(pred,2.5,axis=0),np.percentile(pred,97.5,axis=0),
    #                 alpha=0.15,facecolor=sim_colors[model])

pred = []
# The formula (involving constants in the paper)
a_max, b_max, alpha_max, beta_max, gamma_max = 1.34, 0.959, 2.7, 1, 0.99
pred.append((1.+(alpha_max*Mhm(3.0)/x)**beta_max)**(-1.*gamma_max))
plt.plot(base_mpeak[:-1],np.mean(pred,axis=0),ls='--',lw=1.5, c = "blue")

y_supp = [0.16153803,0.16880832,0.17639995,0.18432238,0.19258464,0.20119522
,0.21016199,0.21949205,0.22919166,0.23926613,0.24971968,0.26055535
,0.27177489,0.28337866,0.29536551,0.30773269,0.32047576,0.33358851
,0.34706289,0.36088892,0.37505467,0.3895462,0.40434754,0.41944066
,0.43480549,0.45041993,0.46625985,0.48229921,0.49851007,0.51486269
,0.53132568,0.54786607,0.56444953,0.58104051,0.59760243,0.61409796
,0.63048919,0.64673795,0.66280606,0.67865563,0.69424935,0.70955081
,0.72452481,0.7391376,0.75335723,0.76715361,0.78049873,0.79336666
,0.80573391,0.81758008,0.82888931,0.83965163,0.84986352,0.85952752
,0.86865149,0.87724778,0.88533242,0.89292419,0.9000437,0.9067126
,0.91295301,0.91878712,0.92423697,0.92932434,0.93407058,0.93849658
,0.94262253,0.94646782,0.95005093,0.95338933,0.95649952,0.95939696
,0.96209618,0.96461074,0.96695336,0.96913588,0.97116938,0.97306418
,0.97482991,0.97647554,0.97800941,0.9794393,0.98077244,0.98201556
,0.98317492,0.98425634,0.98526523,0.98620662,0.98708517,0.98790525
,0.98867087,0.9893858,0.99005353,0.99067728,0.99126007,0.99180471
,0.99231379,0.99278973,0.99323478]
x_supp = [1.28571429e+08, 1.38228770e+08, 1.48611499e+08, 1.59774103e+08,
       1.71775160e+08, 1.84677648e+08, 1.98549277e+08, 2.13462840e+08,
       2.29496600e+08, 2.46734698e+08, 2.65267596e+08, 2.85192549e+08,
       3.06614118e+08, 3.29644718e+08, 3.54405208e+08, 3.81025523e+08,
       4.09645361e+08, 4.40414911e+08, 4.73495644e+08, 5.09061158e+08,
       5.47298093e+08, 5.88407104e+08, 6.32603923e+08, 6.80120483e+08,
       7.31206137e+08, 7.86128970e+08, 8.45177203e+08, 9.08660705e+08,
       9.76912621e+08, 1.05029112e+09, 1.12918127e+09, 1.21399707e+09,
       1.30518362e+09, 1.40321942e+09, 1.50861896e+09, 1.62193533e+09,
       1.74376319e+09, 1.87474187e+09, 2.01555871e+09, 2.16695267e+09,
       2.32971824e+09, 2.50470956e+09, 2.69284494e+09, 2.89511167e+09,
       3.11257119e+09, 3.34636467e+09, 3.59771900e+09, 3.86795322e+09,
       4.15848544e+09, 4.47084032e+09, 4.80665700e+09, 5.16769776e+09,
       5.55585725e+09, 5.97317243e+09, 6.42183326e+09, 6.90419420e+09,
       7.42278654e+09, 7.98033174e+09, 8.57975563e+09, 9.22420384e+09,
       9.91705826e+09, 1.06619548e+10, 1.14628025e+10, 1.23238041e+10,
       1.32494777e+10, 1.42446811e+10, 1.53146369e+10, 1.64649599e+10,
       1.77016867e+10, 1.90313074e+10, 2.04607994e+10, 2.19976643e+10,
       2.36499673e+10, 2.54263791e+10, 2.73362218e+10, 2.93895180e+10,
       3.15970427e+10, 3.39703804e+10, 3.65219858e+10, 3.92652491e+10,
       4.22145661e+10, 4.53854142e+10, 4.87944331e+10, 5.24595124e+10,
       5.63998856e+10, 6.06362306e+10, 6.51907788e+10, 7.00874312e+10,
       7.53518841e+10, 8.10117641e+10, 8.70967726e+10, 9.36388423e+10,
       1.00672304e+11, 1.08234068e+11, 1.16363816e+11, 1.25104211e+11,
       1.34501120e+11, 1.44603856e+11, 1.55465436e+11]
plt.plot(x_supp, y_supp, label="COZMIC WDM Suppression", lw=0.5, c = "green")

# Figure 8  SHMF Supression Plot Style
plt.xscale('log')
plt.yscale('log')
# Dotted line
plt.plot(np.linspace(1e8,5e9,10),np.ones(10),'k--',lw=0.5)
plt.xlim(1e8,10**10)
plt.xticks([10**8,10**9,10**10],[r'$10^8$',r'$10^9$',r'$10^{10}$'])
plt.ylim(0.075,1.15)
plt.yticks([0.2,0.4,0.6,0.8,1.0],[r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'])
plt.xlabel(r'$M_{\rm{sub,peak}}\ [M_{\mathrm{\odot}}]$',labelpad=8)
plt.ylabel(r'$f_{\mathrm{WDM}}$', labelpad=12)
plt.legend(handles=[wdm_plot[0][0]], loc=4,frameon=False,framealpha=1.0,ncol=1)
plt.tight_layout()
plt.legend()
plt.show()
plt.savefig('WDM_SHMF_Suppresion_plot.png')

