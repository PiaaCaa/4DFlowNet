import numpy as np
import os
import h5py
from scipy.integrate import trapz
from matplotlib import pyplot as plt
# from utils import h5functions
# from prepare_data import h5functions
import h5functions
#from prepare_data 
"""
This file contains functions for temporal downsampling with the aim to mimic the temporal sampling of MRI machine
"""


def cartesian_temporal_downsampling(hr, sampling_factor, offset = 0):
    '''
    Downsample by simply taking every n-th frame (sampling factor). Start at offset
    '''
    assert len(hr.shape) == 4, "Input should be 4D"
    assert sampling_factor >= 1, "Sampling factor should be >= 1"

    return hr[offset::sampling_factor, : , :, :]



def temporal_box_averaging(hr, stepsize):
    """
    Average the temporal dimension with a given radius
    This can be used for temporal downsampling as this is simple average smoothing
    """
    assert len(hr.shape) == 4, "Input should be 4D"
    assert stepsize >= 1, "Radius should be >= 1"

    T = hr.shape[0]
    hr_avg = np.zeros_like(hr)

    # loop through all the frames and save the average in hr_avg                        
    for t in range(T):
        for i in range(t, t+stepsize, stepsize):
            j = i.copy()
            # use periodical boundary conditions, i.e. after last frame take first frame again and vice verse
            if i >= T :
                j = j%(T)
            
            # sum up all the 3D data 
            hr_avg += np.asarray(hr_avg[j])
    
    # divide by number of frames to take the average
    hr_avg  /= stepsize

    return hr_avg

def temporal_box_averaging_and_downsampling(hr, downsampling_factor):
    """
    create a box averaging function for temporal downsampling
    """
    assert len(hr.shape) == 4, "Input should be 4D"
    assert downsampling_factor >= 1, "Downsampling factor should be >= 1"

    N_frames, x, y, z = hr.shape

    N_frames_lr = int(np.ceil(N_frames/downsampling_factor))

    sampling_1 = np.zeros((N_frames_lr, x, y, z))
    sampling_2 = np.zeros((N_frames_lr, x, y, z))
    
    offset_sampling = downsampling_factor//2

    # append the first 4 frames to the end to include periodic boundaries
    data_vel = np.concatenate((hr,hr[0:downsampling_factor]), axis=0)
    for i in range(N_frames_lr):
        sampling_1[i, :, :, :] = np.average(data_vel[downsampling_factor*i                :downsampling_factor*i+downsampling_factor]                , axis=0)
        sampling_2[i, :, :, :] = np.average(data_vel[downsampling_factor*i+offset_sampling:downsampling_factor*i+downsampling_factor+offset_sampling], axis=0)
    
    # TODO check if this is alright!!
    # sampling_2 = np.roll(sampling_2, axis= 0, shift=1)
    return sampling_1, sampling_2



def temporal_smoothing_box_function_toeger(hr,t_range, sigma):
    """
    This is the implementation of the smoothing box function for temporal averaging and is based on the paper:  "Blood flow imaging by optimal matching of computational fluid dynamics to 4D-flow data" by Töger et al. 2020
    Note that temporal boundaries are handled periodically.
    Also, different to suggested in the paper, not the area under the curve is normalized to 1, but the sum of the discrete weights are normalized to 1. 
    t_range: range of the heart cycle, going from 0 to (often) 1000 ms (1s), which is then divided into the number of frames
    Returns the smoothed data.
    """
    assert len(hr.shape) == 4, "Input should be 4D"

    hr_avg = np.zeros_like(hr)
    start_t = t_range[0]
    end_t = t_range[-1]
    len_t = end_t - start_t
    dt = t_range[1] - t_range[0] # temporal resolution

    # extend range to handle periodic boundary conditions
    # extend the boundaries by a quarter fo total time to cover temporal cycle
    extended_boundaries_left = np.arange(start_t, start_t - (len_t / 4), -dt)[::-1] #reverse
    extended_boundaries_left = extended_boundaries_left[:-1] # remove last element as it is already included in t_range
    extended_boundaries_right = np.arange(end_t+dt, end_t+(len_t/4), dt)

    # boundary extension length to be the same
    if len(extended_boundaries_right) > len(extended_boundaries_left):
        extended_boundaries_right = extended_boundaries_right[:-1]
    elif len(extended_boundaries_right) < len(extended_boundaries_left):
        extended_boundaries_left = extended_boundaries_left[1:]

    t_range_extended = np.append(np.append(extended_boundaries_left, t_range), extended_boundaries_right)

    def smoothed_box_fct(t, t0, w, sigma):
            """
            Smoothed box function. With alpha = 1 this is not normalized to 1
            """
            non_normalized = (1/(1+np.exp(-(t-(t0-w/2))/sigma)) - 1/(1+np.exp(-(t-(t0+w/2))/sigma)))
            alpha = 1
            # alpha = 1/integral_trapez(non_normalized, t)
            return alpha * non_normalized

    def integral_trapez(fct, t):
        """
        Calculate the integral of the smoothed box function with trapey formula   
        """
        return trapz(fct, t)

    
    # loop through all the frames and return the smoothed result hr_avg
    for i, t0 in enumerate(t_range):
        weighting =  smoothed_box_fct(t_range_extended, t0, dt, sigma)
        
        # normalize the weighting # note: this is not included in the paper 
        weighting /= np.sum(weighting)

        # add the weighting to the periodic boundaries
        periodic_weighting = np.zeros_like(t_range)
        periodic_weighting = weighting[len(extended_boundaries_left):len(extended_boundaries_left)+len(t_range)] # middle
        periodic_weighting[:len(extended_boundaries_right)] += weighting[-len(extended_boundaries_right):] 
        periodic_weighting[-len(extended_boundaries_left):] += weighting[:len(extended_boundaries_left)]


        # weight input by the periodic weighting
        hr_avg[i, :, :, :] = np.sum(hr*periodic_weighting[:, None, None, None], axis = 0)
    
    print(f"Created temporally smoothed data with sigma = {sigma} in range {start_t:.4f} to {end_t:.4f} and resolution {dt:.4f}")
    #4f
    
    return hr_avg



if __name__ == '__main__':
    # load data
    hr_file = 'data/CARDIAC/M6_2mm_step2_static_dynamic.h5'

    # save data
    smooth_file_lr  = 'data/CARDIAC/M6_2mm_step2_temporalsmoothing_toeger_periodic_LRfct.h5'
    smooth_file_hr =  'data/CARDIAC/M6_2mm_step2_temporalsmoothing_toeger_periodic_HRfct.h5'

    keys =  ["mask", ] 
    # delete_data_from_h5(smooth_file_lr, keys)
    # # delete_data_from_h5(smooth_file_hr, keys)
    # merge_data_to_h5(smooth_file_lr, hr_file)
    add_keys = ["dx", "u_max", "v_max", "w_max",  "mag_u", "mag_v", "mag_w", ]

    with h5py.File(hr_file, mode = 'r' ) as p1:
          hr_u = np.asarray(p1['u']) 
          hr_v = np.asarray(p1['v'])
          hr_w = np.asarray(p1['w'])
          mask = np.asarray(p1['mask']).squeeze()
          print(p1.keys(), 'shapes', hr_u.shape, hr_v.shape, hr_w.shape, mask.shape)

   
    t_range = np.linspace(0, 1, hr_u.shape[0])
    smoothing = 0.004

    #-------LR function smoothing-------
    # Note the output will be in same dimension as HR; but the smoothing function is applied on the downsampled data.
    if True: 
        if True: 
            # downsample and then apply smoothing
            hr_u0 = hr_u[::2]
            hr_v0 = hr_v[::2]
            hr_w0 = hr_w[::2]
            hr_u1 = hr_u[1::2]
            hr_v1 = hr_v[1::2]
            hr_w1 = hr_w[1::2]

            t_range0 = t_range[::2]
            t_range1 = t_range[1::2]
            

            hr_u_temporal_smoothing0 = temporal_smoothing_box_function_toeger(hr_u0, t_range0, smoothing)
            hr_v_temporal_smoothing0 = temporal_smoothing_box_function_toeger(hr_v0, t_range0, smoothing)
            hr_w_temporal_smoothing0 = temporal_smoothing_box_function_toeger(hr_w0, t_range0, smoothing)

            hr_u_temporal_smoothing1 = temporal_smoothing_box_function_toeger(hr_u1, t_range1, smoothing)
            hr_v_temporal_smoothing1 = temporal_smoothing_box_function_toeger(hr_v1, t_range1, smoothing)
            hr_w_temporal_smoothing1 = temporal_smoothing_box_function_toeger(hr_w1, t_range1, smoothing)

            print('shapes', hr_u_temporal_smoothing0.shape, hr_u_temporal_smoothing1.shape, hr_u.shape)
            if os.path.exists(smooth_file_lr):
                    print("STOP - File already exists!")
            
            u_temp_smoothing = np.zeros_like(hr_u)
            v_temp_smoothing = np.zeros_like(hr_v)
            w_temp_smoothing = np.zeros_like(hr_w)

            u_temp_smoothing[::2] = hr_u_temporal_smoothing0
            u_temp_smoothing[1::2] = hr_u_temporal_smoothing1
            v_temp_smoothing[::2] = hr_v_temporal_smoothing0
            v_temp_smoothing[1::2] = hr_v_temporal_smoothing1
            w_temp_smoothing[::2] = hr_w_temporal_smoothing0
            w_temp_smoothing[1::2] = hr_w_temporal_smoothing1

            
            
            print(f'saving to {smooth_file_lr}')
            h5functions.save_to_h5(smooth_file_lr, 'u', u_temp_smoothing, expand_dims=False)
            h5functions.save_to_h5(smooth_file_lr, 'v', v_temp_smoothing, expand_dims=False)
            h5functions.save_to_h5(smooth_file_lr, 'w', w_temp_smoothing, expand_dims=False)
        else:
            # apply smoothing on the downsampled data
            print("---------downsampling to 100 to 50------------")
            hr_u0, hr_u1 = temporal_box_averaging_and_downsampling(hr_u, 2)
            hr_v0, hr_v1 = temporal_box_averaging_and_downsampling(hr_v, 2)
            hr_w0, hr_w1 = temporal_box_averaging_and_downsampling(hr_w, 2)
            # print("---------downsampling to 50 to 25------------")
            # test_hr_u1, test_hr_u2 = temporal_box_averaging_and_downsampling(hr_u0, 2)
            # # test_hr_u3, test_hr_u4 = temporal_box_averaging_and_downsampling(hr_u1, 2)

            print("---------downsampling to 100 to 25 ------------")
            lr_u2, lr_u3 = temporal_box_averaging_and_downsampling(hr_u, 4)
            lr_v2, lr_v3 = temporal_box_averaging_and_downsampling(hr_v, 4)
            lr_w2, lr_w3 = temporal_box_averaging_and_downsampling(hr_w, 4)


            lr_u = np.zeros_like(hr_u0)
            lr_v = np.zeros_like(hr_v0)
            lr_w = np.zeros_like(hr_w0)

            lr_u[::2] = lr_u2
            lr_u[1::2] = lr_u3
            lr_v[::2] = lr_v2
            lr_v[1::2] = lr_v3
            lr_w[::2] = lr_w2
            lr_w[1::2] = lr_w3

            print('shapes', lr_u.shape, lr_v.shape, lr_w.shape,  mask[::2].shape, ' hr ', hr_u0.shape)

            mean_speed_lr = np.average(np.sqrt(lr_u**2 + lr_v**2 + lr_w**2), axis = (1, 2, 3), weights=  mask[::2])
            mean_speed_hr = np.average(np.sqrt(hr_u0**2 + hr_v0**2 + hr_w0**2), axis = (1, 2, 3), weights=  mask[::2])
            t_range = np.arange(mean_speed_hr.shape[0])
            plt.subplot(1, 2, 1)
            plt.plot(mean_speed_hr, label = 'hr')
            plt.plot(t_range[::2], mean_speed_lr[::2], label = 'lr')
            plt.subplot(1, 2, 2)
            plt.plot(mean_speed_hr, label = 'hr')
            plt.plot(t_range[1::2], mean_speed_lr[1::2], label = 'lr')
            
            plt.legend()

            plt.show()



            if os.path.exists(smooth_file_hr0):
                print(f"STOP - File {smooth_file_hr0} already exists!")
            else:
            
                # save hr first
                print(f'saving to {smooth_file_hr0}')
                h5functions.save_to_h5(smooth_file_hr0, 'u', hr_u0)
                h5functions.save_to_h5(smooth_file_hr0, 'v', hr_v0)
                h5functions.save_to_h5(smooth_file_hr0, 'w', hr_w0)
                h5functions.save_to_h5(smooth_file_hr0, 'mask',  mask [::2])
                with h5py.File(hr_file, mode = 'r' ) as p1:
                    for key in add_keys:
                        if key != 'dx':
                            h5functions.save_to_h5(smooth_file_hr0, key, np.array(p1[key])[::2], expand_dims=False)
                        else:
                            h5functions.save_to_h5(smooth_file_hr0, key, np.array(p1[key]), expand_dims=False)
                        print(np.array(p1[key]).shape)

            # if os.path.exists(smooth_file_hr1):
            #     print(f"STOP - File {smooth_file_hr1} already exists!")
            # else:
            #             # save hr first
            #     print(f'saving to {smooth_file_hr1}')
            #     h5functions.save_to_h5(smooth_file_hr1, 'u', hr_u0)
            #     h5functions.save_to_h5(smooth_file_hr1, 'v', hr_v0)
            #     h5functions.save_to_h5(smooth_file_hr1, 'w', hr_w0)
            #     h5functions.save_to_h5(smooth_file_hr1, 'mask',  mask [::2])
            #     with h5py.File(hr_file, mode = 'r' ) as p1:
            #         for key in add_keys:
            #             if key != 'dx':
            #                 h5functions.save_to_h5(smooth_file_hr1, key, np.array(p1[key])[::2])
            #             else:
            #                 h5functions.save_to_h5(smooth_file_hr1, key, np.array(p1[key]))
            #             print(np.array(p1[key]).shape)

            if os.path.exists(smooth_file_lr):
                print(f"STOP - File {smooth_file_lr} already exists!")
            else:
                print(f'saving to {smooth_file_lr}')
                h5functions.save_to_h5(smooth_file_lr, 'u', lr_u, expand_dims=False)
                h5functions.save_to_h5(smooth_file_lr, 'v', lr_v, expand_dims=False)
                h5functions.save_to_h5(smooth_file_lr, 'w', lr_w, expand_dims=False)
                with h5py.File(smooth_file_hr0, mode = 'r' ) as p1:
                    for key in add_keys:
                            h5functions.save_to_h5(smooth_file_lr, key, np.array(p1[key]), expand_dims=False)
                            h5functions.save_to_h5(smooth_file_lr, key, np.array(p1[key]), expand_dims=False)
                            print(np.array(p1[key]).shape)

            # also save magnitude etc.
    

            # add to file orginial data such as mask, venc etc.
            # merge_data_to_h5(smooth_file_lr, hr_file)

    #-----------HR smoothing------------ 
    if True: 
        u_temp_smoothing = temporal_smoothing_box_function_toeger(hr_u, t_range, smoothing)
        v_temp_smoothing = temporal_smoothing_box_function_toeger(hr_v, t_range, smoothing)
        w_temp_smoothing = temporal_smoothing_box_function_toeger(hr_w, t_range, smoothing)

        if os.path.exists(smooth_file_hr):
            print("STOP - File already exists!")
            exit()
        print(f'saving to {smooth_file_hr}')
        h5functions.save_to_h5(smooth_file_hr, 'u', u_temp_smoothing,  expand_dims=False)
        h5functions.save_to_h5(smooth_file_hr, 'v', v_temp_smoothing,  expand_dims=False)
        h5functions.save_to_h5(smooth_file_hr, 'w', w_temp_smoothing,  expand_dims=False)

        h5functions.merge_data_to_h5(smooth_file_hr, hr_file)



      