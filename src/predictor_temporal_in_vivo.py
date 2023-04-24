import tensorflow as tf
import numpy as np
import time
import os
from Network.STR4DFlowNet_adapted import STR4DFlowNet
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset_temporal import ImageDataset_temporal
from matplotlib import pyplot as plt
import h5py
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def prepare_temporal_network(patch_size, res_increase, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block):
    # Prepare input
    input_shape = (patch_size,patch_size,patch_size,1)
    u = tf.keras.layers.Input(shape=input_shape, name='u')
    v = tf.keras.layers.Input(shape=input_shape, name='v')
    w = tf.keras.layers.Input(shape=input_shape, name='w')

    u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
    v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
    w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

    input_layer = [u,v,w,u_mag, v_mag, w_mag]

    # network & output
    net = STR4DFlowNet(res_increase,low_res_block=low_res_block, high_res_block=high_res_block,  upsampling_block=upsampling_block )
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, n_low_resblock, n_hi_resblock)
    model = tf.keras.Model(input_layer, prediction)

    return model


if __name__ == '__main__':
    # Define directories and filenames
    model_name = '20230405-1417'

    #/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/PIA/THORAX/P01/h5/P01.h5
    # set filenamaes and directories
    data_dir = '/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/PIA/THORAX/P01/h5'
    filename = 'P01.h5' #TODO double check this if the right mask is used

    output_dir = f'Temporal4DFlowNet/results/in_vivo/THORAX'
    output_filename = f'{model_name}_temporal.h5'
    
    model_path = f'Temporal4DFlowNet/models/Temporal4DFlowNet_{model_name}/Temporal4DFlowNet-best.h5'

    # Params
    patch_size = 12
    res_increase = 2
    batch_size = 16
    round_small_values = True

    # Network - default 8-4
    n_low_resblock = 4
    n_hi_resblock = 8
    low_res_block  = 'resnet_block'     # 'resnet_block' 'dense_block' csp_block
    high_res_block = 'resnet_block'       #'resnet_block'
    upsampling_block = 'linear'#'Conv3DTranspose'#'nearest_neigbor'#'linear'         #' 'linear'  'nearest_neigbor' 'Conv3DTranspose'

    # Setting up
    input_filepath = '{}/{}'.format(data_dir, filename)
    output_filepath = '{}/{}'.format(output_dir, output_filename)

    venc_colnames = ['u_max', 'v_max', 'w_max']

    assert(not os.path.exists(output_filepath)) #STOP if output file is already created

    pgen = PatchGenerator(patch_size, res_increase,include_all_axis = True)
    dataset = ImageDataset_temporal(venc_colnames=['u_max', 'v_max', 'w_max'])
    

    print("Path exists:", os.path.exists(input_filepath), os.path.exists(model_path))
    print("Outputfile exists already: ", os.path.exists(output_filename))

    if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    axis = [0, 1, 2]

    with h5py.File(input_filepath, mode = 'r' ) as h5:
        lr_shape = np.asarray(h5.get("u")).squeeze().shape
        print("Shape of in-vivo data", lr_shape)
        N_frames, _, _, _ = lr_shape

    #TODO make this nicer
    u_combined = np.zeros(lr_shape)
    v_combined = np.zeros(lr_shape)
    w_combined = np.zeros(lr_shape)

    # Loop over all axis
    for a in axis:
        print("________________________________Predict patches with axis: ", a, " ____________________________________________________")
        
        # Check the number of rows in the file
        nr_rows = dataset.get_dataset_len(input_filepath, a)
        print(f"Number of rows in dataset: {nr_rows}")
        
        print(f"Loading 4DFlowNet: {res_increase}x upsample")
        # Load the network
        network = prepare_temporal_network(patch_size, res_increase, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block)
        #low_res_block, high_res_block
        #res_increase,low_res_block=low_res_block, high_res_block=high_res_block,  upsampling_block=upsampling_block 
        network.load_weights(model_path)

        volume = np.zeros((3, u_combined.shape[0],  u_combined.shape[1], u_combined.shape[2],  u_combined.shape[3] ))
        # loop through all the rows in the input file
        for nrow in range(nr_rows):
            print("\n--------------------------")
            print(f"\nProcessed ({nrow+1}/{nr_rows}) - {time.ctime()}")

            # Load data file and indexes
            dataset.load_vectorfield(input_filepath, nrow, axis = a)
            print(f"Original image shape: {dataset.u.shape}")
            
            velocities, magnitudes = pgen.patchify(dataset)
            data_size = len(velocities[0])
            print(f"Patchified. Nr of patches: {data_size} - {velocities[0].shape}")

            # Predict the patches
            results = np.zeros((0,patch_size*res_increase, patch_size, patch_size, 3))
            start_time = time.time()

            for current_idx in range(0, data_size, batch_size):
                time_taken = time.time() - start_time
                print(f"\rProcessed {current_idx}/{data_size} Elapsed: {time_taken:.2f} secs.", end='\r')
                # Prepare the batch to predict
                patch_index = np.index_exp[current_idx:current_idx+batch_size]
                sr_images = network.predict([velocities[0][patch_index],
                                        velocities[1][patch_index],
                                        velocities[2][patch_index],
                                        magnitudes[0][patch_index],
                                        magnitudes[1][patch_index],
                                        magnitudes[2][patch_index]])

                results = np.append(results, sr_images, axis=0)
            # End of batch loop    
            print("results:", results.shape)
        
            time_taken = time.time() - start_time
            print(f"\rProcessed {data_size}/{data_size} Elapsed: {time_taken:.2f} secs.")

            
            for i in range (0,3):
                v = pgen._patchup_with_overlap(results[:,:,:,:,i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
                
                # Denormalized
                v = v * dataset.venc 
                if round_small_values:
                    print(f"Zero out velocity component less than {dataset.velocity_per_px}")
                    # remove small velocity values
                    v[np.abs(v) < dataset.velocity_per_px] = 0
                
                v = np.expand_dims(v, axis=0)
                # prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', f'{dataset.velocity_colnames[i]}__axis{a}', v, compression='gzip')
                print('Original volume: ', volume.shape, 'shape of predicition', v.shape)
                if v.shape[1] != N_frames:
                    print('reshaped v from: ', v.shape)
                    if v.shape[1] < N_frames:
                        v = np.pad(v, (0, 0), (0, N_frames - v.shape[1]), (0, 0), (0, 0))
                    else:
                        v = v[:, :N_frames, :, :]
                    print(v.shape)
                #volume u/v/w, T, X, Y, Z
                if a == 0:      volume[i, :, nrow,  :,      :] = v
                elif a == 1:    volume[i, :, :,     nrow,   :] = v
                elif a == 2:    volume[i, :, :,     :,   nrow] = v


            if dataset.dx is not None:
                new_spacing = dataset.dx / res_increase
                new_spacing = np.expand_dims(new_spacing, axis=0) 
                #prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.dx_colname, new_spacing, compression='gzip')

        # prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', f'u_axis{a}', volume[0, :, :, :], compression='gzip')
        # prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', f'v_axis{a}', volume[1, :, :, :], compression='gzip')
        # prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', f'w_axis{a}', volume[2, :, :, :], compression='gzip')
        u_combined += volume[0, :, :, :] 
        v_combined += volume[1, :, :, :] 
        w_combined += volume[2, :, :, :] 

    print("save combined predictions")
    # save and divide by 3 to get average
    prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', "u_combined", u_combined/len(axis), compression='gzip')
    prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', "v_combined", v_combined/len(axis), compression='gzip')
    prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', "w_combined", w_combined/len(axis), compression='gzip')

    print("Done!")