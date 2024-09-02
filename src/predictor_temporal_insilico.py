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
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def prepare_temporal_network(patch_size, res_increase, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block, post_processing_block):
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
    net = STR4DFlowNet(res_increase,low_res_block=low_res_block, high_res_block=high_res_block,  upsampling_block=upsampling_block , post_processing_block=post_processing_block)
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, n_low_resblock, n_hi_resblock)
    model = tf.keras.Model(input_layer, prediction)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--model", type=str, help="Optional argument to pass the name of the model")
    args = parser.parse_args()

    # Define directories and filenames
    if args.model is not None:
        model_name = args.model
    else:
        model_name = '20240709-2057' # this model: training 2, 3, validation: 1, test:4 

    print("Model name: ", model_name)
    # set_names = ['Test', 'Validation', 'Training', 'Training', 'Training', 'Training']
    # data_models= ['4', '1', '2', '3', '5', '6',]
    # steps = [ 2, 2, 2, 2, 2, 2]
    # file_names = ['M4_2mm_step2_cs_invivoP02_lr.h5', 'M1_2mm_step2_cs_invivoP01_lr.h5', 
                #   'M2_2mm_step2_cs_invivoP04_lr.h5', 'M3_2mm_step2_cs_invivoP03_lr.h5', 
                #   'M5_2mm_step2_cs_invivoP05_lr.h5', 'M6_2mm_step2_cs_invivoP03_lr.h5',]
    # file_names = [ 'M1_2mm_step2_cs_invivoP01_lr_50frames.h5', 
    #                'M2_2mm_step2_cs_invivoP04_lr_50frames.h5', ]
    #TODO
    set_names = ['Validation', 'Training', 'Training', 'Training', 'Training']
    data_models= ['1', '2', '3', '5', '6',]
    steps = [ 2, 2, 2, 2, 2]
    file_names = ['M1_2mm_step2_cs_invivoP01_lr.h5', 
                  'M2_2mm_step2_cs_invivoP04_lr.h5', 'M3_2mm_step2_cs_invivoP03_lr.h5', 
                  'M5_2mm_step2_cs_invivoP05_lr.h5', 'M6_2mm_step2_cs_invivoP03_lr.h5']

    #file_names = ['M4_2mm_step2_static_dynamic_noise.h5', 'M1_2mm_step2_static_dynamic_noise.h5'] #'M2_2mm_step2_static_dynamic_noise.h5', 'M3_2mm_step2_static_dynamic_noise.h5', 
    # file_names = ['M4_2mm_step2_temporalsmoothing_toeger_periodic_LRfct_noise.h5', 'M1_2mm_step2_temporalsmoothing_toeger_periodic_LRfct_noise.h5'] #'M2_2mm_step2_static_dynamic_noise.h5', 'M3_2mm_step2_static_dynamic_noise.h5', 
    # set filenamaes and directories
    data_dir = 'data/CARDIAC'
    output_dir = f'results/Temporal4DFlowNet_{model_name}'
    model_dir = f'models/Temporal4DFlowNet_{model_name}'

    for set_name, data_model, step, filename in zip(set_names, data_models, steps, file_names):
        print('Start predicition of:', set_name, data_model, filename)

        # set filenamaes and directories
        output_dir = f'Temporal4DFlowNet/results/Temporal4DFlowNet_{model_name}'
        output_filename = f'{set_name}set_result_model{data_model}_2mm_step{step}_{model_name[-4::]}_temporal.h5'
        
        output_filename = f'{set_name}set_result_model{data_model}_2mm_step{step}_{model_name[-4::]}_temporal.h5'
        
        model_path = f'{model_dir}/Temporal4DFlowNet-best.h5'

        # Params
        patch_size = 16
        res_increase = 2
        batch_size = 16
        round_small_values = False
        downsample_input_first = False # For LR data, which is downsampled on the fly this should be set to True
        upsampling_factor = 2

        # Network - default 8-4
        n_low_resblock = 8
        n_hi_resblock = 4
        low_res_block  = 'resnet_block'      # 'resnet_block' 'dense_block' csp_block
        high_res_block = 'resnet_block'     #'resnet_block'
        upsampling_block = 'linear'         #'nearest_neigbor'#'linear'#'Conv3DTranspose'#'nearest_neigbor'#'linear''nearest_neigbor' 'Conv3DTranspose'
        post_processing_block = None        #'unet_block'

        # Setting up
        input_filepath = '{}/{}'.format(data_dir, filename)
        output_filepath = '{}/{}'.format(output_dir, output_filename)
        print("Output file path: ", output_filepath)
        assert(not os.path.exists(output_filepath))  #STOP if output file is already created

        pgen = PatchGenerator(patch_size, res_increase,include_all_axis = True, downsample_input_first=downsample_input_first)
        dataset = ImageDataset_temporal(venc_colnames=['u_max', 'v_max', 'w_max'])#['venc_u', 'venc_v', 'venc_w'])

        print("Path exists:", os.path.exists(input_filepath), os.path.exists(model_path))
        print("Outputfile exists already: ", os.path.exists(output_filename))

        if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

        with h5py.File(input_filepath, mode = 'r' ) as h5:
            lr_shape = h5.get("u").shape

        if downsample_input_first:
            u_combined = np.zeros(lr_shape)
            v_combined = np.zeros(lr_shape)
            w_combined = np.zeros(lr_shape)
        else:
            u_combined = np.zeros((upsampling_factor*lr_shape[0], *lr_shape[1::]))
            v_combined = np.zeros((upsampling_factor*lr_shape[0], *lr_shape[1::]))
            w_combined = np.zeros((upsampling_factor*lr_shape[0], *lr_shape[1::]))

        # Loop over all axis
        for a in [0, 1, 2]:
            print("________________________________Predict patches with axis: ", a, " ____________________________________________________")
            
            # Check the number of rows in the file
            nr_rows = dataset.get_dataset_len(input_filepath, a)
            
            print(f"Number of rows in dataset: {nr_rows}")
            print(f"Loading 4DFlowNet: {res_increase}x upsample")

            # Load the network
            network = prepare_temporal_network(patch_size, res_increase, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block, post_processing_block)
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
        prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', "u_combined", u_combined/3, compression='gzip')
        prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', "v_combined", v_combined/3, compression='gzip')
        prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', "w_combined", w_combined/3, compression='gzip')

        print('-----------------------------Done with: ', set_name, data_model, '----------------------------------------------')
    print("Done!")