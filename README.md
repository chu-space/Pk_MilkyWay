# Pk_MilkyWay
Reconstructing  linear matter power spectrum P(k) using observations of the Milky Way satellite population for scales smaller than 1 megaparsec.

The first step of this constraining problem is to identify the cross-sections at which the velocity-dependent interacting dark matter (IDM) models sufficiently match or is suppressed below the Warm Dark Matter (WDM). 

You do this by finding the envelope cross-section, at which the transfer function or the ratio between the matter power spectrum and the benchmark cold dark matter (CDM) power spectrum. I have matched the cosmological parameters with the transfer function of the IDM spectra first - each normalised by the CDM power spectra.

With this directory, you can gain access information to transfer functions which tell the story of matter suppression of different models for either the intermediate mass (researching in between the previous suspected masses of IDM) or interpolated (a cross-section evaluated at the latest 5.9kev WDM observational constraint line) cross-sections match the expected transfer function behaviour we expect. Both of these new searches provides new insight in the momentum and cross-section dependence a particle model of dark matter would have. 

Next, I compare with reference transfer functions from the results of COZMIC 1 inside COZMIC_IDM_Tk, access the reference CAMB data from camb_data_tk. Naming conventions for directories are as follows: Tk means CAMB formatted transfer functions, whereeas Pk represents the power spectrum. CLASS outputs are titled output and plots derived from these searches are capitalised and remain the proof that these ICs are fitting for a new suite of simulations to expand out knowledge of a broad range of particle dark matter models and other sectors of standard model particles.

*1. Pk*: Once again the first step of this search is the plotting of the power spectra, the relevant files are as follows:
CLASS plotting script, VG_short_cdm_Pk.ini and VG_short_idm_Pk.ini are utilised as template files to plot the transfer function or the ratio of the Pks. The halfmode models are found by finding the appropriate cross-section for a unique mass and velocity dependence where *T**2 = 0.25* or *T = 0.5* where *T* is the ratio of the IDM matter power spectra to the bnechmark CDM model's spectra. 

*2. Tk*: Next, the same CLASS parameters that were used in the fitting exercise above are utilised for the CAMB output:
If CAMB outputs are desired, we would need both the synchronous and newtonian gauge. Hence, VG_correct_params_idm_newtonian.ini and VG_correct_params_idm_synchronous.ini are used as template files to have their parameters adjusted depending on the model we are interested in and are configured apprioriately to produce the necessary format for CAMB files. The dynamic_camb python file utilises the unique naming conventions of the gauge outputs, thus satisfying the required CAMB format used in multi-scale simulation initial conditions (MUSIC).

*3.CAMB*

