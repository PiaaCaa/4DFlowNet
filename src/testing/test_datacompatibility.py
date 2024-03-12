import os
import h5py
import numpy as np

def check_csv_patch_compatibility(csv_file, data_dir = 'data/CARDIAC'):
    """
    Check if the the pathes (and shapes) of the .h5 files loaded in the csv file are compatible 
    """
    
    print(f"--------- Checking csv file {os.path.basename(csv_file)} ---------")
    df = pd.read_csv(csv_file)
    lr_files = df['source'].unique()
    hr_files = df['target'].unique()

    vel_colnames = ['u', 'v','w']
    mag_colnames = [ 'mag_u', 'mag_v', 'mag_w']


    lst_missmatch = []

    assert len(lr_files) == len(hr_files), "Number of low resolution and high resolution files do not match"
    for lr, hr in zip(lr_files, hr_files):
        with h5py.File(f'{data_dir}/{lr}', mode = 'r' ) as p1:
            with h5py.File(f'{data_dir}/{hr}', mode = 'r' ) as p2:

                reference_shape = p1['u'].shape
                for vel, mag in zip(vel_colnames, mag_colnames):
                    if p1[vel].shape != reference_shape:
                        print("Shape of", lr, "do not match for", vel, p1[vel].shape,  "vs", reference_shape)
                        if lr not in lst_missmatch:
                            lst_missmatch.append(lr)
                    if p1[mag].shape != reference_shape:
                        print("Shape of", lr, "do not match for", mag, p1[mag].shape,  "vs", reference_shape)
                        if lr not in lst_missmatch:
                            lst_missmatch.append(lr)
                    if p2[vel].shape != reference_shape:
                        print("Shape of", hr, "do not match for", vel, p1[vel].shape, "vs", reference_shape)
                        if hr not in lst_missmatch:
                            lst_missmatch.append(hr)
                    if p2[mag].shape != reference_shape:
                        print("Shape of", hr, "do not match for", mag, p1[mag].shape, "vs", reference_shape)
                        if hr not in lst_missmatch:
                            lst_missmatch.append(hr)
                
                lr_mask = np.array(p1['mask']).squeeze()
                hr_mask = np.array(p2['mask']).squeeze()

                if lr_mask.shape != reference_shape:
                    print("Shape of", lr, "do not match for mask", p1['mask'].shape, "vs", reference_shape)
                    if lr not in lst_missmatch:
                        lst_missmatch.append(lr)
                if hr_mask.shape != reference_shape:
                    print("Shape of", hr, "do not match for mask", p2['mask'].shape, "vs", reference_shape)
                    if hr not in lst_missmatch:
                        lst_missmatch.append(hr)

    return lst_missmatch
