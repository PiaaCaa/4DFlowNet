# from tvtk.api import tvtk, write_data
import numpy as np
from utils.evaluate_utils import get_boundaries, calculate_mean_speed, temporal_linear_interpolation_np, temporal_NN_interpolation
import matplotlib.pyplot as plt
import h5py
import sys
import os
import pyvista as pv
# from utils import prediction_utils
# from prepare_data import temporal_downsampling
from scipy.integrate import trapz
import pandas as pd

def calculate_RST(u, v, w, temporal_window, t_range):
    """
    Calculate the mean velocity (time averaged) for the given data
    Temporal window is the number of ms of which is averaged
    """
    # question: increase temporal resolution using linear interpolation ? 

    N_frames = u.shape[0]
    dt = t_range[1] - t_range[0]
    N_dtwindow = int(np.ceil(temporal_window/dt)//2) # number of frames in the temporal window in one direction
    print(dt, np.ceil(temporal_window/dt), N_dtwindow)
    
    mean_u = np.zeros_like(u)
    mean_v = np.zeros_like(v)
    mean_w = np.zeros_like(w)

    xx = np.zeros_like(u)
    xy = np.zeros_like(u)
    xz = np.zeros_like(u)
    yy = np.zeros_like(u)
    yz = np.zeros_like(u)
    zz = np.zeros_like(u)

    if N_dtwindow >0 : 
        print(f"Calculate RST with a temporal window of {temporal_window} ms ({N_dtwindow} frames)")

        # calculate the mean velocity for each frame (periodical windows)
        for i in range(N_frames):
            idx = np.index_exp[i-N_dtwindow:i+N_dtwindow]
            if i-N_dtwindow < 0:
                idx = np.index_exp[0:i+N_dtwindow*2]
                mean_u[i] = np.average(np.roll(u, N_dtwindow -i, axis = 0)[idx], axis = 0)
                mean_v[i] = np.average(np.roll(v, N_dtwindow -i, axis = 0)[idx], axis = 0)
                mean_w[i] = np.average(np.roll(w, N_dtwindow -i, axis = 0)[idx], axis = 0)
            elif i+N_dtwindow >= N_frames:
                idx = np.index_exp[i-N_dtwindow*2:i]
                mean_u[i] = np.average(np.roll(u,-N_dtwindow, axis=0)[idx], axis = 0)
                mean_v[i] = np.average(np.roll(v,-N_dtwindow, axis=0)[idx], axis = 0)
                mean_w[i] = np.average(np.roll(w,-N_dtwindow, axis=0)[idx], axis = 0)
            else:
                mean_u[i] = np.average(u[idx], axis = 0)
                mean_v[i] = np.average(v[idx], axis = 0)
                mean_w[i] = np.average(w[idx], axis = 0)


            # assume N = 1 measurements as we just have one cycle
                
            # calculate the fluctuation
            fluct_u = u[i] - mean_u[i]
            fluct_v = v[i] - mean_v[i]
            fluct_w = w[i] - mean_w[i]

            # calculate Raynolds shear stress tensor (3 x 3) which has 6 unique component due ot symmetry
            xx[i] = np.multiply(fluct_u, fluct_u)
            xy[i] = np.multiply(fluct_u, fluct_v)
            xz[i] = np.multiply(fluct_u, fluct_w)
            yy[i] = np.multiply(fluct_v, fluct_v)
            yz[i] = np.multiply(fluct_v, fluct_w)
            zz[i] = np.multiply(fluct_w, fluct_w)
    else:
        mean_u = u
        mean_v = v
        mean_w = w

    return mean_u, mean_v, mean_w, xx, xy, xz, yy, yz, zz

def save_meanu_RST_to_vti(dt_window, mean_u, mean_v, mean_w, xx, xy, xz, yy, yz, zz, save_as):
    """Save vti file for each time frame including the mean velocity and the Reynolds shear stress tensor"""
    
    N_frames = mean_u.shape[0]

    for frame in range(N_frames):

        mean_vel    = np.column_stack((mean_u[frame].ravel(order="F"), mean_v[frame].ravel(order="F"), mean_w[frame].ravel(order="F")))
        RST         = np.column_stack((xx[frame].ravel(order="F"), xy[frame].ravel(order = "F"), xz[frame].ravel(order = "F"),yy[frame].ravel(order = "F"),  yz[frame].ravel(order = "F"),  zz[frame].ravel(order = "F")))

        # create with pyvista
        grid = pv.UniformGrid()

        grid.dimensions = np.array(mean_u[frame].shape) + 1
        grid.origin = (0, 0, 0)  # The bottom left corner of the data set
        grid.spacing = (2, 2, 2)

        grid.cell_data[f"U{dt_window}"] = mean_vel
        grid.cell_data[f"RST{dt_window}"] = RST

        grid.save(f"{save_as}_frame{frame}.vti")

        
    print(f"Done! - Saved {N_frames} vti files to {save_as}")



def concatenate_csv_files(lst_files, output_filename):
    # Check if the list is empty
    if not lst_files:
        print("No files to concatenate.")
        return

    # Load the first file to get the header
    df_concat = pd.read_csv(lst_files[0])

    # Iterate over the remaining files
    for file in lst_files[1:]:
        # Load the file
        df = pd.read_csv(file)

        # Check if the headers match
        if not df.columns.equals(df_concat.columns):
            print(f"Header mismatch in file: {file}")
            continue

        # Concatenate the dataframes
        df_concat = pd.concat([df_concat, df], ignore_index=True)

    # Save the concatenated dataframe to a new CSV file
    df_concat.to_csv(output_filename, index=False)

    print(f"Concatenated {len(lst_files)} files into {output_filename}.")

        





if __name__ == '__main__':
       # Get the parent directory
    # parent_dir = os.path.dirname(os.path.realpath(__file__))
    # print("Parent dir:", parent_dir)
    # Add the parent directory to sys.path
    # sys.path.append("c:/Users/piacal/Code/SuperResolution4DFlowMRI/Temporal4DFlowNet/")
    # exit()
    
    # A = np.arange(20)
    # dt = 3
    # for i in range(len(A)):
    #     idx = np.index_exp[i-dt:i+dt]
    #     if i-dt < 0:
    #         idx = np.index_exp[0:i+dt*2]
    #         print('idx', idx, np.roll(A, dt-i)[idx])
    #     elif i+dt >= len(A):
    #          idx = np.index_exp[i-dt*2:i]
    #          print('idx', idx, np.roll(A, -dt)[idx])
    #     else:
    #         print('idx', idx, A[idx])

    # csv_files = ['data/CARDIAC/Temporal16MODEL2_2mm_step2_invivoP04_magn_tempsmooth_toeger.csv', 'data/CARDIAC/Temporal16MODEL3_2mm_step2_invivoP03_magn_tempsmooth_toeger.csv', ]
    # name = 'data/CARDIAC/Temporal16MODEL23_2mm_step2_invivoP04P03_magn_tempsmooth_toeger.csv'

    # concatenate_csv_files(csv_files, name)
    # data_original = {}
    # vel_colnames = ['u', 'v','w']
    # venc_colnames = [ 'u_max', 'v_max', 'w_max']
    # mag_colnames = [ 'mag_u', 'mag_v', 'mag_w']
    # vencs = {}

    
    hr_file = 'data/CARDIAC/M1_2mm_step2_static_dynamic.h5'

    with h5py.File(hr_file, mode = 'r' ) as p1:
          hr_u = np.asarray(p1['u']) 
          hr_v = np.asarray(p1['v'])
          hr_w = np.asarray(p1['w'])

    t_range = (0, 1, hr_u.shape[0])
    dt_window = 0.00
    mean_u, mean_v, mean_w, xx, xy, xz, yy, yz, zz = calculate_RST(hr_u, hr_v, hr_w, dt_window, np.linspace(0, 1, hr_u.shape[0]))
    save_meanu_RST_to_vti("0", mean_u, mean_v, mean_w, xx, xy, xz, yy, yz, zz, 'results/data/RST_M1_2mm_step2_static_dynamic')

    

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

   