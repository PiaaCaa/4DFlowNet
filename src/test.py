# from tvtk.api import tvtk, write_data
import numpy as np
from utils.evaluate_utils import get_boundaries, calculate_mean_speed, temporal_linear_interpolation_np, temporal_NN_interpolation
import matplotlib.pyplot as plt
import h5py
import os
from utils import prediction_utils
from prepare_data import temporal_downsampling
from scipy.integrate import trapz


if __name__ == '__main__':
    in_vivo_path = '/home/pcallmer/Temporal4DFlowNet/data/CARDIAC/M1_2mm_step2_static_dynamic_noise.h5'
    data_original = {}
    vel_colnames = ['u', 'v','w']
    venc_colnames = [ 'u_max', 'v_max', 'w_max']
    mag_colnames = [ 'mag_u', 'mag_v', 'mag_w']
    vencs = {}

    
    hr_file = 'data/CARDIAC/M4_2mm_step2_static_dynamic.h5'

    with h5py.File(hr_file, mode = 'r' ) as p1:
          hr_u = np.asarray(p1['u']) 
          hr_v = np.asarray(p1['v'])
          hr_w = np.asarray(p1['w'])

    t_range = np.linspace(0, 1, hr_u.shape[0])
    smoothing = 0.004

    
    
    hr_u_temporal_smoothing = temporal_downsampling.temporal_smoothing_box_function_toeger(hr_u, t_range, smoothing)
    # hr_v_temporal_smoothing = temporal_downsampling.temporal_smoothing_box_function_toeger(hr_v, t_range, smoothing)
    # hr_w_temporal_smoothing = temporal_downsampling.temporal_smoothing_box_function_toeger(hr_w, t_range, smoothing)

    # prediction_utils.save_to_h5('results/interpolation/M4_2mm_step2_temporalsmooting_toeger_normalized.h5', 'u_temporal_smoothing', hr_u_temporal_smoothing, compression='gzip')
    # prediction_utils.save_to_h5('results/interpolation/M4_2mm_step2_temporalsmooting_toeger_normalized.h5', 'v_temporal_smoothing', hr_v_temporal_smoothing, compression='gzip')
    # prediction_utils.save_to_h5('results/interpolation/M4_2mm_step2_temporalsmooting_toeger_normalized.h5', 'w_temporal_smoothing', hr_w_temporal_smoothing, compression='gzip')
    # B = np.arange(0, 30, 1).reshape((3, 5, 2))
    # A = np.array([1, 2, 3])

    # print(B * A[:, None, None])	

    # create interpolation file from hr (no noisy data!)
    # interpolation_path = 'results/interpolation/M4_2mm_step2_static_dynamic_interpolate_no_noise.h5'
    # for vel in vel_colnames:
    #     with h5py.File(hr_file, mode = 'r' ) as p1: 
    #         data_original[vel] = np.asarray(p1[vel])

    #         # downsample and create linear interpolation model
    #         interpolation_linear = temporal_linear_interpolation_np(data_original[vel][::2], data_original[vel].shape)
    #         interpolation_NN = temporal_NN_interpolation(data_original[vel][::2], data_original[vel].shape)

    #     print("dir exists", os.path.isdir('results/interpolation/'))
    #     prediction_utils.save_to_h5(interpolation_path, f'linear_{vel}' , interpolation_linear, compression='gzip')
    #     prediction_utils.save_to_h5(interpolation_path, f'NN_{vel}' , interpolation_NN, compression='gzip')

    
    # sigmas = np.linspace(0.1, 5, 10)
    # t = np.arange(0, 10, w)
    # t0s = np.arange(0, 10, 0.1)
    # dt = t_range[1] - t_range[0]
    # for t0 in t_range:
    #     plt.plot(t_range, smoothed_box_fct(t_range, t0, dt, smoothing), label = f't0 = {t0}')
    
    # plt.title(f"Smoothed box function with sigma = {smoothing}")
    # plt.xlabel('time')
    # plt.ylabel('f(x)')
    # # plt.legend()
    # plt.show()



    if False:
        for m in ['1', '2', '3', '4']:
            in_vivo_path = f'/home/pcallmer/Temporal4DFlowNet/data/CARDIAC/M{m}_2mm_step2_static_dynamic.h5'
            with h5py.File(in_vivo_path, mode = 'r' ) as p1:
                        # print(p1.keys())
                        print('M', m)
                        print('shape', p1['u'].shape)
                        print(np.array(p1['dx']))
                        mask =  np.asarray(p1['mask'])
                        temporal_mask = mask.copy()

                        # data_original['mask'] = temporal_mask
                        for vel, venc, mag in zip(vel_colnames, venc_colnames, mag_colnames):
                            data_original[vel] = np.asarray(p1[vel])
                            data_original[f'{vel}_fluid'] = np.multiply(data_original[vel], temporal_mask)
                        #     data_original[mag] = np.asarray(p1[mag])

                        speed = np.sqrt(np.square(data_original['u']) + np.square(data_original['v']) + np.square(data_original['w']))
                        mean_speed = calculate_mean_speed(data_original['u'], data_original['v'], data_original['w'], temporal_mask)
                        plt.clf()
                        plt.figure(figsize=(10,3))
                        plt.plot(mean_speed, '.-', label = 'High resolution', color = 'black')
                        plt.title('Mean speed')
                        plt.xlabel('frame')
                        plt.ylabel(' mean speed (cm/s)')
                        plt.legend()
                        plt.savefig(f'/home/pcallmer/Temporal4DFlowNet/results/data/mean_speed_M{m}.svg')

                        # print('mean speed ', mean_speed)
                        # print('mean speed ', np.average(mean_speed))
                        # print('mean speed ', np.average(speed, weights=temporal_mask, axis = (1,2,3))*100)
                        # print('mean speed diff', np.average(speed, weights=temporal_mask, axis = (1,2,3))*100 - mean_speed)
                        print('mean speed min', np.min(mean_speed))
                        print('mean speed max', np.max(mean_speed))
                        print('mean speed mean', np.mean(mean_speed))
                        print('mean speed std', np.std(mean_speed))
                        print('mean speed median', np.median(mean_speed))

                        print('max speed ', np.max(speed))
                        print('min speed ', np.min(speed))
                        print('mean speed ', np.mean(speed))

   