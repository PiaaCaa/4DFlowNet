import numpy as np
import h5py
import PatchData as pd
import os
import argparse

def load_data_shape(input_filepath):
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        t, x, y, z = hdf5['u'].shape

    print(f"Dataset of size: {t, x, y, z} ")
    return t, x, y, z 



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--lrdata", type=str, help="Optional argument to pass the name of LR data")
    parser.add_argument("--hrdata", type=str, help="Optional argument to pass the name of HR data")
    args = parser.parse_args()

    if args.lrdata is not None and args.hrdata is not None:
        print(f"Data model is {args.lrdata}")
        lr_file = args.lrdata
        hr_file = args.hrdata
    else:
        hr_file = 'M4_2mm_step2_cloudmagnRot_toeger_HRfct.h5'       #HiRes velocity data
        lr_file = 'M4_2mm_step2_cloudmagnRot_toeger_LRfct_noise.h5' #LowRes velocity data 

    # Parameters
    temporal_preparation = True
    patch_size = 16 # Patch size, this will be checked to make sure the generated patches do not go out of bounds
    n_patch = 20    # number of patch per time frame
    n_empty_patch_allowed = 1 # max number of empty patch per frame
    all_rotation = False # When true, include 90,180, and 270 rotation for each patch. When False, only include 1 random rotation. (not possible for temporal sampling)
    reverse = True
    mask_threshold = 0.5 # Threshold for non-binary mask 
    minimum_coverage = 0.2 # Minimum fluid region within a patch. Any patch with less than this coverage will not be taken. Range 0-1

    
    base_path = 'data/CARDIAC'

    output_filename = f'{base_path}/Temporal{patch_size}MODEL{hr_file[1]}_2mm_step2_cloudmagnRot_toeger.csv'


    #TODO check the compatibility in the test iteratoor
    
    # Load the data
    input_filepath = f'{base_path}/{lr_file}'
    T, X, Y, Z = load_data_shape(input_filepath)
  
    # Check if the files exist  
    assert(os.path.isfile(f'{base_path}/{hr_file}'))    # HR file does not exist
    assert(os.path.isfile(f'{base_path}/{lr_file}'))    # LR file does not exist 

    # Prepare the CSV output
    if temporal_preparation:
        pd.write_header_temporal(output_filename)
    else:
        pd.write_header(output_filename)

    # because the data is homogenous in 1 table, we only need the first data
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        mask = np.asarray(hdf5['mask']).squeeze()
        if (len(mask.shape) == 4 and mask.shape[0]==1) or len(mask.shape) == 3:  
            mask = pd.create_temporal_mask(mask, T)
    
        frames = hdf5["u"].shape[0]
        t_lr, x_lr, y_lr, z_lr = np.array(hdf5['u']).shape


    with h5py.File(f'{base_path}/{lr_file}', 'r') as hf:
        t_hr, x_hr, y_hr, z_hr = np.array(hf['u']).shape

    assert((x_lr, y_lr, z_lr) == (x_hr, y_hr, z_hr))

    # check on temporal aspect
    if t_hr == t_lr:
        step_t = 2 #or adjust this to downsampling size, default is factor 2
    else:
        print('Set step size to 1, means it is expected that downsampling is already done in the data and not on the fly.')
        step_t = 1

    # We basically need the mask on the lowres data, the patches index are retrieved based on the LR data.
    print("Overall shape", mask.shape)

    # Do the thresholding
    binary_mask = (mask >= mask_threshold) * 1

    if temporal_preparation:
        
        axis = [0, 1, 2]
        for a in axis:

            if a == 0: 
                print("______Create patches for (t, y, z) slices_____________")
                for idx in range(1, X):
                    pd.generate_temporal_random_patches_all_axis(lr_file, hr_file, output_filename,a, idx,  n_patch, binary_mask, patch_size, minimum_coverage, n_empty_patch_allowed, reverse, step_t)
            elif a == 1:
                print("______Create patches for (t, x, z) slices_____________")
                for idx in range(1, Y):
                    pd.generate_temporal_random_patches_all_axis(lr_file, hr_file, output_filename,a, idx,  n_patch, binary_mask, patch_size, minimum_coverage, n_empty_patch_allowed, reverse, step_t)
            elif a == 2:
                print("______Create patches for (t, x, y) slices_____________")
                for idx in range(1, Z):
                    pd.generate_temporal_random_patches_all_axis(lr_file, hr_file, output_filename,a, idx,  n_patch, binary_mask, patch_size, minimum_coverage, n_empty_patch_allowed, reverse, step_t)
    else:
        # Generate random patches for all time frames
        for index in range(0, T):
            print('Generating patches for row', index)
            pd.generate_random_patches(lr_file, hr_file, output_filename, index, n_patch, binary_mask, patch_size, minimum_coverage, n_empty_patch_allowed, all_rotation)

    
    print(f'Done. File saved in {output_filename}')