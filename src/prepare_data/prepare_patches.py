import numpy as np
import h5py
import PatchData as pd

# def load_data(input_filepath):
#     with h5py.File(input_filepath, mode = 'r' ) as hdf5:
#         data_nr = len(hdf5['u'])

#     indexes = np.arange(data_nr)
#     print("Dataset: {} rows".format(len(indexes)))
#     return indexes

def load_temporal_data(input_filepath):
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        data_nr = hdf5['u'].shape[1]

    indexes = np.arange(data_nr)
    print("Dataset: {} rows".format(len(indexes)))
    return indexes


if __name__ == "__main__": 
    temporal_preparation = True
    patch_size = 10 # Patch size, this will be checked to make sure the generated patches do not go out of bounds
    n_patch = 8    # number of patch per time frame
    n_empty_patch_allowed = 0 # max number of empty patch per frame
    all_rotation = False # When true, include 90,180, and 270 rotation for each patch. When False, only include 1 random rotation.
    mask_threshold = 0.4 # Threshold for non-binary mask 
    minimum_coverage = 0.2 # Minimum fluid region within a patch. Any patch with less than this coverage will not be taken. Range 0-1
    
    
    base_path = 'Temporal4DFlowNet/data/CARDIAC'
    lr_file = 'M1_2mm_step5_static_TLR_no_noise.h5' #LowRes velocity data
    hr_file = 'M1_2mm_step5_static.h5' #HiRes velocity data
    output_filename = f'{base_path}/Temporal{patch_size}MODEL1_2_no_noise.csv'

    
    # Load the data
    input_filepath = f'{base_path}/{lr_file}'
    file_indexes = load_temporal_data(input_filepath)
    
    # Prepare the CSV output
    pd.write_header(output_filename)

    # because the data is homogenous in 1 table, we only need the first data
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        mask = np.asarray(hdf5['mask'][0])
        frames = hdf5["u"].shape[0]

    # We basically need the mask on the lowres data, the patches index are retrieved based on the LR data.
    print("Overall shape", mask.shape)

    # Do the thresholding
    binary_mask = (mask >= mask_threshold) * 1
    if temporal_preparation: binary_mask = pd.create_temporal_mask(binary_mask, frames)

    # Generate random patches for all time frames
    for index in file_indexes:
        print('Generating patches for row', index)
        if temporal_preparation:
            pd.generate_temporal_random_patches(lr_file, hr_file, output_filename, index, frames, n_patch, binary_mask, patch_size, minimum_coverage, n_empty_patch_allowed, all_rotation)
        else:
            pd.generate_random_patches(lr_file, hr_file, output_filename, index, n_patch, binary_mask, patch_size, minimum_coverage, n_empty_patch_allowed, all_rotation)
    print(f'Done. File saved in {output_filename}')