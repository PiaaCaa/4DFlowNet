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
        hr_file = 'M1_2mm_step2_cs_invivoP01_hr.h5'       #HiRes velocity data
        lr_file = 'M1_2mm_step2_cs_invivoP01_lr_hr_data_TBD.h5' #LowRes velocity data 

    # Parameters
    temporal_preparation = True
    spatial_patch_size = 16 # Patch size, this will be checked to make sure the generated patches do not go out of bounds
    
    n_patch = 4    # number of patch per time frame
    n_empty_patch_allowed = 0 # max number of empty patch per frame
    all_rotation = False # When true, include 90,180, and 270 rotation for each patch. When False, only include 1 random rotation. (not possible for temporal sampling)
    reverse_1 = False
    extended_data_augmentation = True

    # extended data augmentation parameters
    if extended_data_augmentation:
        temporal_patch_size = 4
        all_rotation = True
        swap_velocity_components = True
        change_sign_velocity_components = True
        # if reverese_1 is true we enable both flipping in vertical and horizontal direction
        save_nonaugmented_patch= True
        n_patches_augmented_from_original_patch = 1
        only_choose_apply_one_augmentation_technique=False
        sign_change_on_all_components = True


    
    mask_threshold = 0.5 # Threshold for non-binary mask 
    minimum_coverage = 0.5 # Minimum fluid region within a patch. Any patch with less than this coverage will not be taken. Range 0-1

    base_path = 'data/CARDIAC'#'/mnt/c/Users/piacal/Code/Supon4DFlowMRI/Temporal4DFlowNet/data/CARDIAC'#'data/CARDIAC'
    output_filename = f'{base_path}/csv_files/Temporal{spatial_patch_size}MODEL{hr_file[1]}_2mm_step2_cs_invivomagn_exclfirst2frames_highcoverage_HRHR_step1_TBD_.csv'

    # Check if the files exist  
    assert(os.path.isfile(f'{base_path}/{hr_file}'))    # HR file does not exist
    assert(os.path.isfile(f'{base_path}/{lr_file}'))    # LR file does not exist 


    # Prepare the CSV output
    if temporal_preparation:
        if extended_data_augmentation:
            pd.write_header_temporal_extended_data_augmentation(output_filename)
        else:
            pd.write_header_temporal(output_filename)
    else:
        pd.write_header(output_filename)

    # because the data is homogenous in 1 table, we only need the first data
    with h5py.File(f'{base_path}/{lr_file}', mode = 'r' ) as hdf5:
        frames, X, Y, Z = hdf5["u"].shape
        mask_lr = np.asarray(hdf5['mask']).squeeze()
        if (len(mask_lr.shape) == 4 and mask_lr.shape[0]==1) or len(mask_lr.shape) == 3:  
            mask_lr = pd.create_temporal_mask(mask_lr, frames)
    
        t_lr, x_lr, y_lr, z_lr = np.array(hdf5['u']).shape


    with h5py.File(f'{base_path}/{hr_file}', 'r') as hf:
        t_hr, x_hr, y_hr, z_hr = np.array(hf['u']).shape
        mask_hr = np.asarray(hf['mask']).squeeze()

    # check on temporal aspect
    if t_hr == t_lr:
        step_t = 2 # or adjust this to downsampling size, default is factor 2
        check_t = 1
    else:
        print('Set step size to 1, means it is expected that downsampling is already done in the data and not on the fly.')
        step_t = 1
        check_t = 2

    assert((x_lr, y_lr, z_lr) == (x_hr, y_hr, z_hr)) # for temporal downsampling we need the same spatial shape
    assert np.sum(np.abs(mask_lr - mask_hr[::check_t])) == 0 # mask of lr and hr should be the same after downsampling

    # We basically need the mask on the lowres data, the patches index are retrieved based on the LR data.
    print("Overall shape", mask_lr.shape)

    # Do the thresholding
    binary_mask = (mask_lr >= mask_threshold) * 1

    if extended_data_augmentation:
        pd.save_csv_file_settings_json(lr_file, hr_file, output_filename, n_patch, binary_mask, spatial_patch_size,temporal_patch_size, minimum_coverage, n_empty_patch_allowed, 
                                        reverse_1,  all_rotation, swap_velocity_components, change_sign_velocity_components, step_t ,
                                        save_nonaugmented_patch, n_patches_augmented_from_original_patch, only_choose_apply_one_augmentation_technique, sign_change_on_all_components)

    if temporal_preparation:
        if extended_data_augmentation:

            for a in [0, 1, 2]:
                if a == 0: 
                    print("______Create patches for (t, y, z) slices_____________")
                    for idx in range(1, X):
                        pd.generate_temporal_random_patches_extended_data_augmentation(lr_file, hr_file, output_filename,a, idx,  n_patch, binary_mask, spatial_patch_size,temporal_patch_size, minimum_coverage, n_empty_patch_allowed, 
                                                                flipping= reverse_1,  all_rotation = all_rotation, swap_velocity_components = swap_velocity_components, change_sign_velocity_components = change_sign_velocity_components,  step_t =step_t,
                                                                save_nonaugmented_patch= save_nonaugmented_patch, n_patches_augmented_from_original_patch =n_patches_augmented_from_original_patch, only_choose_apply_one_augmentation_technique=only_choose_apply_one_augmentation_technique, sign_change_on_all_components=sign_change_on_all_components)
                elif a == 1:
                    print("______Create patches for (t, x, z) slices_____________")
                    for idx in range(1, Y):
                        pd.generate_temporal_random_patches_extended_data_augmentation(lr_file, hr_file, output_filename,a, idx,  n_patch, binary_mask, spatial_patch_size,temporal_patch_size, minimum_coverage, n_empty_patch_allowed, 
                                                                flipping= reverse_1,  all_rotation = all_rotation, swap_velocity_components = swap_velocity_components, change_sign_velocity_components = change_sign_velocity_components,  step_t =step_t,
                                                                save_nonaugmented_patch= save_nonaugmented_patch, n_patches_augmented_from_original_patch =n_patches_augmented_from_original_patch, only_choose_apply_one_augmentation_technique=only_choose_apply_one_augmentation_technique, sign_change_on_all_components=sign_change_on_all_components)
                elif a == 2:
                    print("______Create patches for (t, x, y) slices_____________")
                    for idx in range(1, Z):
                        pd.generate_temporal_random_patches_extended_data_augmentation(lr_file, hr_file, output_filename,a, idx,  n_patch, binary_mask, spatial_patch_size,temporal_patch_size, minimum_coverage, n_empty_patch_allowed, 
                                                                flipping= reverse_1,  all_rotation = all_rotation, swap_velocity_components = swap_velocity_components, change_sign_velocity_components = change_sign_velocity_components,  step_t =step_t,
                                                                save_nonaugmented_patch= save_nonaugmented_patch, n_patches_augmented_from_original_patch =n_patches_augmented_from_original_patch, only_choose_apply_one_augmentation_technique=only_choose_apply_one_augmentation_technique, sign_change_on_all_components=sign_change_on_all_components)
        else:
            for a in [0, 1, 2]:
                if a == 0: 
                    print("______Create patches for (t, y, z) slices_____________")
                    for idx in range(1, X):
                        pd.generate_temporal_random_patches_all_axis(lr_file, hr_file, output_filename,a, idx,  n_patch, binary_mask, spatial_patch_size, minimum_coverage, n_empty_patch_allowed, reverse_1, step_t)
                elif a == 1:
                    print("______Create patches for (t, x, z) slices_____________")
                    for idx in range(1, Y):
                        pd.generate_temporal_random_patches_all_axis(lr_file, hr_file, output_filename,a, idx,  n_patch, binary_mask, spatial_patch_size, minimum_coverage, n_empty_patch_allowed, reverse_1, step_t)
                elif a == 2:
                    print("______Create patches for (t, x, y) slices_____________")
                    for idx in range(1, Z):
                        pd.generate_temporal_random_patches_all_axis(lr_file, hr_file, output_filename,a, idx,  n_patch, binary_mask, spatial_patch_size, minimum_coverage, n_empty_patch_allowed, reverse_1, step_t)
    else:
        # Generate random patches for all time frames
        for index in range(0, frames):
            print('Generating patches for row', index)
            pd.generate_random_patches(lr_file, hr_file, output_filename, index, n_patch, binary_mask, spatial_patch_size, minimum_coverage, n_empty_patch_allowed, all_rotation)

    
    print(f'Done. File saved in {output_filename}')
    # also save this py file in the same directory
    