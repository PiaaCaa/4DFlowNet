import numpy as np
import os
import h5py
import random
import fft_downsampling as fft
import scipy.ndimage as ndimage
from h5functions import save_to_h5
from visualize_utils import generate_gif_volume
# from utils.evaluate_utils import

                       

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

def simple_temporal_downsampling(hr_data, downsample =2):
    assert(len(hr_data.shape) == 4) # assume that data is of form t, h, w, d
    if downsample ==2:
        # if hr_data.shape[0] % 2 == 0:
        #     print("Even number of frames: delete last frame")
        
        lr_frames = int(np.ceil(hr_data.shape[0]/2))

        lr_data = np.zeros((lr_frames, hr_data.shape[1], hr_data.shape[2], hr_data.shape[3]))
        print("Temporal downsampling from ", hr_data.shape[0], " frames to ", lr_frames, " frames." )
        lr_data = hr_data[::2, : , :, :]
        print(hr_data[::2, : , :, :].shape)
        return lr_data
        
    else:
        print("Only implemented for downsampling by 2, please implement if needed.")


if __name__ == '__main__':
    # Config
    base_path = 'Temporal4DFlowNet/data/CARDIAC'
    # Put your path to Hires Dataset
    input_filepath  =  f'{base_path}/M3_2mm_step2_static_dynamic.h5'
    output_filename = f'{base_path}/M3_2mm_step2_static_dynamic_noise.h5' 
    # Downsample rate 
    downsample = 2

    # Check if file already exists
    if os.path.exists(output_filename): print("___ WARNING: overwriting already existing .h5 file!!____ ")
    assert(not os.path.exists(output_filename))    # if file already exists: STOP, since it just adds to the current file

    # --- Ready to do downsampling ---
    # setting the seeds for both random and np random, if we need to get the same random order on dataset everytime
    # np.random.seed(10)
    crop_ratio = 1 / downsample
    base_venc_multiplier = 1.1 # Default venc is set to 10% above vmax

    # For radial downsampling the average of the adjacent pixels are taken and the noise is added
    use_radial_downsamling = False
    radia_downsamping_avg_pixel = 3 # number of pixels to be averaged over, should be odd

    # Possible magnitude and venc values
    mag_values  =  np.asarray([60, 80, 120, 180, 240]) # in px values [0-4095]
    venc_values =  np.asarray([0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4, 4.5]) # in m/s

    # Load the mask once
    with h5py.File(input_filepath, mode = 'r' ) as hf:
        # Some h5 files have 4D mask with 1 in the temporal dimension while others are already 3D
        #create temporal mask, either already loaded or with static temporal mask
        mask = np.asarray(hf['mask']).squeeze()
        if len(mask.shape) == 3: 
            mask = np.repeat(np.expand_dims(mask, 0), hf['u'].shape[0], axis=0)

        data_count = len(hf.get("u"))
        
        hr_u =     np.zeros_like(hf["u"])
        hr_v =     np.zeros_like(hf["u"])
        hr_w =     np.zeros_like(hf["u"])
        hr_mag_u = np.zeros_like(hf["u"])
        hr_mag_v = np.zeros_like(hf["u"])
        hr_mag_w = np.zeros_like(hf["u"])
    
    print("Datacount:", data_count, " mask shape ", mask.shape)
    
    is_mask_saved = False # just to mark if the mask already saved or not
    for idx in range(data_count):
        targetSNRdb = np.random.randint(140,170) / 10
        print("Processing data row", idx, "target SNR", targetSNRdb, "db")
        
        # Create the magnitude based on the possible values
        ## This is a part of augmentation to make sure we have varying magnitude
        mag_multiplier = mag_values[idx % len(mag_values)]
        mag_image = mask * mag_multiplier
        
        # Load the velocity U V W from H5
        with h5py.File(input_filepath, mode = 'r' ) as hf:
            mask = np.asarray(hf['mask']).squeeze()
            # Some h5 files have 4D mask with 1 in the temporal dimension while others are already 3D
            if len(mask.shape) == 3: 
                mask = np.repeat(np.expand_dims(mask, 0), hf['u'].shape[0], axis=0)
            
            if use_radial_downsamling:
                    print(f"_____Radial downsampling with avaging over {radia_downsamping_avg_pixel} pixels")
                    hr_u_frame = np.zeros_like(hf['u'][idx])
                    hr_v_frame = np.zeros_like(hf['v'][idx])
                    hr_w_frame = np.zeros_like(hf['w'][idx])
                    for i in range(idx -radia_downsamping_avg_pixel//2, idx+radia_downsamping_avg_pixel//2+1):
                        
                        # use periodical boundary conditions, i.e. after last frame take first frame again and vice verse
                        if i >= data_count :
                            i = i%(data_count)
                        # sum up all the 3D data 
                        hr_u_frame += np.asarray(hf['u'][i])
                        hr_v_frame += np.asarray(hf['v'][i])
                        hr_w_frame += np.asarray(hf['w'][i])
                    
                    # divide by number of frames to take the average
                    hr_u_frame /= radia_downsamping_avg_pixel
                    hr_v_frame /= radia_downsamping_avg_pixel
                    hr_w_frame /= radia_downsamping_avg_pixel
                        
            else:
                hr_u_frame = np.asarray(hf['u'][idx])
                hr_v_frame = np.asarray(hf['v'][idx])
                hr_w_frame = np.asarray(hf['w'][idx])
            
            # Calculate the possible VENC for each direction (* 1.1 to avoid aliasing)
            max_u = np.asarray(hf['u_max'][idx]) * base_venc_multiplier
            max_v = np.asarray(hf['v_max'][idx]) * base_venc_multiplier
            max_w = np.asarray(hf['w_max'][idx]) * base_venc_multiplier
        
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
        
        
        # attention: is just adding noise NOT downsampling image
        hr_u[idx, :, :, :], hr_mag_u[idx, :, :, :] =  fft.downsample_phase_img(hr_u_frame, mag_image[idx], venc_u, crop_ratio, targetSNRdb, temporal_downsampling=True)   
        hr_v[idx, :, :, :], hr_mag_v[idx, :, :, :] =  fft.downsample_phase_img(hr_v_frame, mag_image[idx], venc_v, crop_ratio, targetSNRdb, temporal_downsampling=True)   
        hr_w[idx, :, :, :], hr_mag_w[idx, :, :, :] =  fft.downsample_phase_img(hr_w_frame, mag_image[idx], venc_w, crop_ratio, targetSNRdb, temporal_downsampling=True)   

        print("Peak signal to noise ratio:", peak_signal_to_noise_ratio(hr_u_frame, hr_u[idx, :, :, :]), " db")
        # hr_u[idx, :, :, :], mag_u = hr_u_frame, mag_image
        # hr_v[idx, :, :, :], mag_v = hr_v_frame, mag_image
        # hr_w[idx, :, :, :], mag_w = hr_w_frame, mag_image
        # hr_mag_u[idx, :, :, :] = mag_u
        # hr_mag_v[idx, :, :, :] = mag_v
        # hr_mag_w[idx, :, :, :] = mag_w

        # only every second (even) needed for downsampling 
        if idx % 2 == 0: 
            save_to_h5(output_filename, "venc_u", venc_u)
            save_to_h5(output_filename, "venc_v", venc_v)
            save_to_h5(output_filename, "venc_w", venc_w)
            save_to_h5(output_filename, "SNRdb", targetSNRdb)
        

    # DO the downsampling
    #TODO this can be done more fancy
    # lr_u = simple_temporal_downsampling(hr_u, downsample)
    # lr_v = simple_temporal_downsampling(hr_v, downsample)
    # lr_w = simple_temporal_downsampling(hr_w, downsample)
    lr_u = hr_u
    lr_v = hr_v
    lr_w = hr_w

    mag_u = hr_mag_u   #simple_temporal_downsampling(hr_mag_u)
    mag_v = hr_mag_v   #simple_temporal_downsampling(hr_mag_v)
    mag_w = hr_mag_w   #simple_temporal_downsampling(hr_mag_w)

    
    # Save the downsampled images
    
    save_to_h5(output_filename, "u", lr_u, expand_dims=False)
    save_to_h5(output_filename, "v", lr_v, expand_dims=False)
    save_to_h5(output_filename, "w", lr_w, expand_dims=False)

    save_to_h5(output_filename, "mag_u", mag_u, expand_dims=False)
    save_to_h5(output_filename, "mag_v", mag_v, expand_dims=False)
    save_to_h5(output_filename, "mag_w", mag_w, expand_dims=False)

    # Only save mask once

    if not is_mask_saved:
        # Use the downsampled shape, lr_u, to crop mask before applying zoom.
        # Otherwise the resulting mask and vector fields may mismatch in shape.
        #mask_crop = crop_mask(mask, np.array(lr_u), downsample)
        #new_mask = ndimage.zoom(mask_crop, crop_ratio, order=1)
        print("Saving downsampled mask...")
        save_to_h5(output_filename, "mask", mask)

        is_mask_saved = True

    print("Done!")