import os
import sys
import numpy as np
import h5py
import random
import fft_downsampling as fft
from h5functions import save_to_h5
from temporal_downsampling import temporal_box_averaging_and_downsampling, temporal_smoothing_box_function_toeger                       

def choose_venc():
    '''
        Give a 68% that data will have a same venc on all 3 velocity components.
    '''
    my_list = ['same'] * 68 + ['diff'] * 32
    return random.choice(my_list)

# Crop mask to match desired shape * downsample
def crop_mask(mask, desired_shape, downsample):
    crop = (np.array(mask.shape) - np.array(desired_shape.shape)*downsample)/2
    if crop[0]:
        mask = mask[1:-1,:,:]
    if crop[1]:
        mask = mask[:,1:-1,:]
    if crop[2]:
        mask = mask[:,:,1:-1]
        
    return mask

def simple_temporal_downsampling(hr_data, downsample =2, offset = 0):
    assert(len(hr_data.shape) == 4) # assume that data is of form t, h, w, d
    assert(offset < downsample) # offset should be less than downsamplerate

    lr_data = hr_data[offset::downsample, : , :, :]
    print("Temporal downsampling from ", hr_data.shape[0], " frames to ", lr_data.shape[0], " frames." )
    return lr_data


if __name__ == '__main__':

    # Config
    base_path = 'data/CARDIAC'
    # Put your path to Hires Dataset
    input_filepath  =  f'{base_path}/M4_2mm_step2_invivoP02_magnitude.h5'
    output_filename =  f'{base_path}/M4_2mm_step2_invivoP02_magnitude_noise.h5' 

    # Downsample rate, set to 1 to keep the same resolution 
    spatial_downsample = 1.0 #1: no downsampling, 2: half the resolution, 4: quarter the resolution
    temporal_downsample = 1.0
    keep_framerate = True # if true, the framerate is kept (e.g. creating downsampling in training pipeline), otherwise the number of frames is downsampled by 1/temporal_downsample rate
    t_downsample_method = 'smooth' # options: 'radial', 'cartesian', 'box', 'smooth'
    
    # add noise to the data
    add_noise = True

    # create magnitude image from mask in freq. domain, otherwise copy magnitude from original data
    create_magnitude = False

    # Check if file already exists
    if os.path.exists(output_filename): print("___ WARNING: overwriting already existing .h5 file!!____ ")
    assert(not os.path.exists(output_filename))    # if file already exists: STOP, since it just adds to the current file

    # --- Ready to do downsampling ---
    # setting the seeds for both random and np random, if we need to get the same random order on dataset everytime
    # np.random.seed(10)
    crop_ratio = 1 / spatial_downsample
    base_venc_multiplier = 1.1 # Default venc is set to 10% above vmax

    # Possible magnitude and venc values
    mag_values  =  np.asarray([60, 80, 120, 180, 240]) # in px values [0-4095]
    venc_values =  np.asarray([0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4, 4.5, 5.0, 6.0]) # in m/s

    # Load the mask once
    with h5py.File(input_filepath, mode = 'r' ) as hf:
        # Some h5 files have 4D mask with 1 in the temporal dimension while others are already 3D
        
        mask = np.asarray(hf['mask']).squeeze()
        if len(mask.shape) == 3: # create dynamic mask, either already loaded or with static temporal mask
            mask = np.repeat(np.expand_dims(mask, 0), hf['u'].shape[0], axis=0)

        N_frames = hf.get("u").shape[0]
                
        hr_u = np.asarray(hf['u'])
        hr_v = np.asarray(hf['v'])
        hr_w = np.asarray(hf['w'])
        mag_u = np.asarray(hf['mag_u'])


    
    print("Number of frames:", N_frames, " mask shape ", mask.shape)
    
    #--------------temporal smoothing---------------
    # temporal downsampling/smoothing here before adding noise for each frame
    if temporal_downsample > 1:
        print("Temporal downsampling/smoothing with method: ", t_downsample_method)
        if t_downsample_method == 'box':
            # overwrite the original data with smoothed data (created two "series" of temporal box averaging)
            hr_u[::2],hr_u[1::2]  = temporal_box_averaging_and_downsampling(hr_u, temporal_downsample)
            hr_v[::2],hr_v[1::2]  = temporal_box_averaging_and_downsampling(hr_v, temporal_downsample)
            hr_w[::2],hr_w[1::2]  = temporal_box_averaging_and_downsampling(hr_w, temporal_downsample)

            
        elif t_downsample_method == 'smooth':
            #parameters for toeger smoothing
            t_range = np.linspace(0, 1, hr_u.shape[0])
            smoothing = 0.004
            hr_u[::2] = temporal_smoothing_box_function_toeger(hr_u[::2], t_range[::2], smoothing)
            hr_v[::2] = temporal_smoothing_box_function_toeger(hr_v[::2], t_range[::2], smoothing)
            hr_w[::2] = temporal_smoothing_box_function_toeger(hr_w[::2], t_range[::2], smoothing)

            hr_u[1::2] = temporal_smoothing_box_function_toeger(hr_u[1::2], t_range[1::2], smoothing)
            hr_v[1::2] = temporal_smoothing_box_function_toeger(hr_v[1::2], t_range[1::2], smoothing)
            hr_w[1::2] = temporal_smoothing_box_function_toeger(hr_w[1::2], t_range[1::2], smoothing)

    #TODO extend if needed

    #--------------temporal downsampling---------------
    if not keep_framerate and N_frames == hr_u.shape[0]:
        hr_u = simple_temporal_downsampling(hr_u, temporal_downsample, offset=0)
        hr_v = simple_temporal_downsampling(hr_v, temporal_downsample, offset=0)
        hr_w = simple_temporal_downsampling(hr_w, temporal_downsample, offset=0)
        mask = simple_temporal_downsampling(mask, temporal_downsample, offset=0)
        N_frames_lr = hr_u.shape[0]
    else:
        N_frames_lr = N_frames
        
    lr_u =     np.zeros_like(hr_u)
    lr_v =     np.zeros_like(hr_u)
    lr_w =     np.zeros_like(hr_u)
    lr_mag_u = np.zeros_like(hr_u)
    lr_mag_v = np.zeros_like(hr_u)
    lr_mag_w = np.zeros_like(hr_u)

    # Loop over each frame and add noise (and spatial downsample if needed)
    for idx in range(N_frames_lr):
        targetSNRdb = np.random.randint(140,170) / 10
        print("Processing data row", idx, "target SNR", targetSNRdb, "db")
        
        # Create the magnitude based on the possible values
        ## This is a part of augmentation to make sure we have varying magnitude
        mag_multiplier = mag_values[idx % len(mag_values)]
        mag_image = mask * mag_multiplier #TODO overthink this approach!
        # mag_image[:, 1::, 1::, 1::] = mag_u
        
        # get vel image for given frame index
        hr_u_frame = hr_u[idx]
        hr_v_frame = hr_v[idx]
        hr_w_frame = hr_w[idx]
        
        # ------------ venc extraction----------------
        with h5py.File(input_filepath, mode = 'r' ) as hf:
            idx_venc = idx
            if not keep_framerate: idx_venc = idx * temporal_downsample

            # Calculate the possible VENC for each direction (* 1.1 to avoid aliasing)
            max_u = np.asarray(hf['u_max'][idx_venc]) * base_venc_multiplier
            max_v = np.asarray(hf['v_max'][idx_venc]) * base_venc_multiplier
            max_w = np.asarray(hf['w_max'][idx_venc]) * base_venc_multiplier
        
        # We assume most of the time, we use venc 1.50 m/s
        all_max = np.array([max_u, max_v, max_w])
        
        venc_choice = choose_venc()
        if (venc_choice == 'same'):
            max_vel = np.max(all_max)
            if max_vel < 1.5:
                venc_u = 1.5
                venc_v = 1.5
                venc_w = 1.5
            else:
                # choose a venc up to 2 higher than current max vel
                randindx = np.random.randint(2)
                venc = venc_values[np.where(venc_values > max_vel)][randindx]
                venc_u = venc
                venc_v = venc
                venc_w = venc
        else:
            # Different venc
            randindx = np.random.randint(2)
            venc_u = venc_values[np.where(venc_values > max_u)][randindx]

            randindx = np.random.randint(2)
            venc_v = venc_values[np.where(venc_values > max_v)][randindx]

            randindx = np.random.randint(2)
            venc_w = venc_values[np.where(venc_values > max_w)][randindx]
            
            # Skew the randomness by setting main velocity component to 1.5
            main_vel = np.argmax(all_max) # check which one is main vel component
            vencs = [venc_u, venc_v, venc_w]
            if vencs[main_vel] < 1.5:
                print("Forcing venc", vencs[main_vel], " to 1.5")
                vencs[main_vel] = 1.5 # just because 1.5 is the common venc

                # set it back to the object
                venc_u = vencs[0]
                venc_v = vencs[1]
                venc_w = vencs[2]
        
        #----------------- Spatial downsample and add noise -----------------
        # Downsample the data in the frequency domain (spatial) and add white gaussian noise
        if  add_noise:
            lr_u[idx, :, :, :], lr_mag_u[idx, :, :, :] =  fft.noise_and_downsampling(hr_u_frame, mag_image[idx], venc_u,  targetSNRdb, spatial_crop_ratio=crop_ratio, add_noise=add_noise)   
            lr_v[idx, :, :, :], lr_mag_v[idx, :, :, :] =  fft.noise_and_downsampling(hr_v_frame, mag_image[idx], venc_v,  targetSNRdb, spatial_crop_ratio=crop_ratio, add_noise=add_noise)   
            lr_w[idx, :, :, :], lr_mag_w[idx, :, :, :] =  fft.noise_and_downsampling(hr_w_frame, mag_image[idx], venc_w,  targetSNRdb, spatial_crop_ratio=crop_ratio, add_noise=add_noise)  
        else:
            lr_u[idx, :, :, :], lr_mag_u[idx, :, :] = hr_u_frame, mag_image[idx]
            lr_v[idx, :, :, :], lr_mag_v[idx, :, :] = hr_v_frame, mag_image[idx]
            lr_w[idx, :, :, :], lr_mag_w[idx, :, :] = hr_w_frame, mag_image[idx]

        #TODO compute SNR here
        # print("Peak signal to noise ratio:", peak_signal_to_noise_ratio(hr_u_frame, hr_u[idx, :, :, :]), " db")

        # print('Signal to noise ratio inside fluid region on noisy data: ', signaltonoise_fluid_region(hr_u[idx]))

        # Save the data to the file
        if not keep_framerate and idx % temporal_downsample == 0:
            save_to_h5(output_filename, "venc_u", venc_u)
            save_to_h5(output_filename, "venc_v", venc_v)
            save_to_h5(output_filename, "venc_w", venc_w)
            save_to_h5(output_filename, "SNRdb", targetSNRdb)
        else:
            save_to_h5(output_filename, "venc_u", venc_u)
            save_to_h5(output_filename, "venc_v", venc_v)
            save_to_h5(output_filename, "venc_w", venc_w)
            save_to_h5(output_filename, "SNRdb", targetSNRdb)


    if create_magnitude:
        # magnitude image is created from the mask in the frequency domain
        mag_u = lr_mag_u   
        mag_v = lr_mag_v   
        mag_w = lr_mag_w
    else:
        # magnitude image is copied from the original magnitude input image
        with h5py.File(input_filepath, mode = 'r' ) as hf:
            mag_u = np.asarray(hf['mag_u'])
            mag_v = np.asarray(hf['mag_v'])
            mag_w = np.asarray(hf['mag_w'])
        
        if not keep_framerate:
            mag_u = simple_temporal_downsampling(mag_u, temporal_downsample, offset=0)
            mag_v = simple_temporal_downsampling(mag_v, temporal_downsample, offset=0)
            mag_w = simple_temporal_downsampling(mag_w, temporal_downsample, offset=0)

    # include tests to check that shape matches with HR data (if framerate keeps the same and no spatial downsampling)
    if keep_framerate and spatial_downsample == 1.0:
        assert(N_frames == N_frames_lr)
        with h5py.File(input_filepath, mode = 'r+' ) as hf:
            assert(hf['u'].shape == lr_u.shape)
            assert(hf['v'].shape == lr_v.shape)
            assert(hf['w'].shape == lr_w.shape)
            assert(hf['mag_u'].shape == mag_u.shape)
            assert(hf['mag_v'].shape == mag_v.shape)
            assert(hf['mag_w'].shape == mag_w.shape)
            assert(hf['mask'].shape == mask.shape)

    # Save the processed images
    save_to_h5(output_filename, "u", lr_u, expand_dims=False)
    save_to_h5(output_filename, "v", lr_v, expand_dims=False)
    save_to_h5(output_filename, "w", lr_w, expand_dims=False)

    save_to_h5(output_filename, "mag_u", mag_u, expand_dims=False)
    save_to_h5(output_filename, "mag_v", mag_v, expand_dims=False)
    save_to_h5(output_filename, "mag_w", mag_w, expand_dims=False)

    save_to_h5(output_filename, "mask", mask)

    print("Done!")