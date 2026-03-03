#!/usr/bin/env python3
import os
import glob
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

# Cosmological Parameters
Z_VAL = 99     # Redshift to evaluate
H_VAL = 0.7    # Hubble parameter (h)

INPUT_DIRS = [
    'output_reference_5.9kev_cross_sections',
    'output_reference_new_masses_cross_sections'
]

DIR_59KEV = '5.9kev_CAMB_Tk'
DIR_COZMIC = 'COZMIC_CAMB_Tk'
DIR_NEW_MASSES = 'new_masses_CAMB_Tk'

# CAMB Header format
CAMB_HEADER = '{0:^15s} {1:^15s} {2:^15s} {3:^15s} {4:^15s} {5:^15s} {6:^15s} {7:^15s} {8:^15s} {9:^15s} {10:^15s} {11:^15s} {12:^15s}'.format(
    'k/h','CDM','baryon','photon','nu','mass_nu','total','no_nu','total_de','Weyl','v_CDM','v_b','v_b-v_c'
)

def process_directory(input_dir):
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    # Finding all synchronous gauge tk files
    sync_files = glob.glob(os.path.join(input_dir, '*_synchronous__tk.dat'))
    
    if not sync_files:
        print(f"No synchronous Tk files found in {input_dir}.")
        return

    for sync_path in sync_files:
        # Extract base name depending on naming scheme
        base_name = os.path.basename(sync_path).replace('_synchronous__tk.dat', '')

        # Sorting my output into different directories for individual simulation aims
        if '5.9kev' in base_name:
            out_dir = DIR_59KEV
        elif 'halfmode' in base_name or 'envelope' in base_name:
            if 'new_masses' in input_dir:
                out_dir = DIR_NEW_MASSES
            else:
                out_dir = DIR_COZMIC
        else:
            out_dir = 'outlying_names_Tk'
            
        os.makedirs(out_dir, exist_ok=True)

        # Depends on your CLASS root naming scheme you have chosen to adopt - for me its like this
        newt_path = os.path.join(input_dir, f"{base_name}_newtonian__tk.dat")
        back_path = os.path.join(input_dir, f"{base_name}_newtonian__background.dat") 
        
        if not (os.path.exists(newt_path) and os.path.exists(back_path)):
            print(f"Skipping {base_name}: Missing matching newtonian or background files.")
            continue
            
        print(f"Processing: {base_name} -> saving to {out_dir}/")
        
        data_camb_sync = np.loadtxt(sync_path)
        data_class_new = np.loadtxt(newt_path)
        data_back = np.loadtxt(back_path)
        
        # Interpolate H(z) -> data_back[:,0] is z, data_back[:,3] is H [1/Mpc]
        funH_z = InterpolatedUnivariateSpline(np.flip(data_back[:,0]), np.flip(data_back[:,3]))
        H = funH_z(Z_VAL) # 1/Mpc
        
        # Initialize output array (Rows = number of k values, Cols = 13 CAMB columns)
        num_k = len(data_class_new[:,0])
        out_data = np.zeros((num_k, 13))
        
        k_h = data_class_new[:, 0]
        
        # 0th: data_camb_sync[:,0] = 1:k(h/Mpc) #'k/h'
        out_data[:, 0] = data_camb_sync[:, 0]
        
        # 1st: np.abs(data_camb_sync[:,1]) = 2:|-T_cdm/k2| = T_cdm/k2 # 'CDM' 
        out_data[:, 1] = np.abs(data_camb_sync[:, 1]) 
        
        # 2nd: np.abs(data_camb_sync[:,3]) = 4:|-T_b/k2| = T_b/k2 # 'baryon'
        out_data[:, 2] = np.abs(data_camb_sync[:, 3]) 
        
        # 3-5th: dummy variables # 'photon','nu','mass_nu' (Left as 0.0)
        
        # 6th: np.abs(data_camb_sync[:,7]) = 8:t_tot # 'total'
        out_data[:, 6] = np.abs(data_camb_sync[:, 7])
        
        # 7-9th: dummy variables 'no_nu','total_de','Weyl' (Left as 0.0)
        
        # 10th: vc 
        out_data[:, 10] = np.abs((1 + Z_VAL) * data_class_new[:, 3] / ((k_h * H_VAL)**2 * H))
        
        # 11th: vb 
        out_data[:, 11] = np.abs((1 + Z_VAL) * data_class_new[:, 2] / ((k_h * H_VAL)**2 * H))
        
        # 12th: dummy variable 'v_b-v_c' (Left as 0.0)
        
        output_file = os.path.join(out_dir, f"{base_name}_CAMB_format.dat")
        
        # Removed comments='' to allow default '#' prefix and matched fmt string
        np.savetxt(output_file, out_data, fmt='%15.6e', header=CAMB_HEADER)

def main():
    for input_dir in INPUT_DIRS:
        print(f"\n--- Scanning input directory: {input_dir} ---")
        process_directory(input_dir)
        
    print("\n--- All files processed and sorted! ---")

if __name__ == "__main__":
    main()
