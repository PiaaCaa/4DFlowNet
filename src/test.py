# from tvtk.api import tvtk, write_data
import numpy as np
from utils.evaluate_utils import get_boundaries, calculate_mean_speed, temporal_linear_interpolation_np, temporal_NN_interpolation
import matplotlib.pyplot as plt
import h5py
import sys
import os
import pyvista as pv
# from utils import prediction_utils
from prepare_data import temporal_downsampling
from prepare_data import fft_downsampling as fft_fcts
from scipy.integrate import trapz
import scipy.io as sio
import pandas as pd
import glob, os
from prepare_data import h5functions
from tqdm import tqdm

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
    """
    Concatenate the CSV files into a single CSV file
    """
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

        



#TODO delete this since it is in testing foler
def check_csv_patch_compatibility(csv_file, data_dir = 'data/CARDIAC'):
    """
    Check if the the pathes (and shapes) of the .h5 files loaded in the csv file are compatible 
    """
    
    print(f"--------- Checking csv file {os.path.basename(csv_file)} ---------")
    df = pd.read_csv(csv_file)
    lr_files = df['source'].unique()
    hr_files = df['target'].unique()

    vel_colnames = ['u', 'v','w']
    mag_colnames = [ 'mag_u', 'mag_v', 'mag_w']


    lst_missmatch = []

    assert len(lr_files) == len(hr_files), "Number of low resolution and high resolution files do not match"
    for lr, hr in zip(lr_files, hr_files):
        with h5py.File(f'{data_dir}/{lr}', mode = 'r' ) as p1:
            with h5py.File(f'{data_dir}/{hr}', mode = 'r' ) as p2:

                reference_shape = p1['u'].shape
                for vel, mag in zip(vel_colnames, mag_colnames):
                    if p1[vel].shape != reference_shape:
                        print("Shape of", lr, "do not match for", vel, p1[vel].shape,  "vs", reference_shape)
                        if lr not in lst_missmatch:
                            lst_missmatch.append(lr)
                    if p1[mag].shape != reference_shape:
                        print("Shape of", lr, "do not match for", mag, p1[mag].shape,  "vs", reference_shape)
                        if lr not in lst_missmatch:
                            lst_missmatch.append(lr)
                    if p2[vel].shape != reference_shape:
                        print("Shape of", hr, "do not match for", vel, p1[vel].shape, "vs", reference_shape)
                        if hr not in lst_missmatch:
                            lst_missmatch.append(hr)
                    if p2[mag].shape != reference_shape:
                        print("Shape of", hr, "do not match for", mag, p1[mag].shape, "vs", reference_shape)
                        if hr not in lst_missmatch:
                            lst_missmatch.append(hr)
                
                lr_mask = np.array(p1['mask']).squeeze()
                hr_mask = np.array(p2['mask']).squeeze()

                if lr_mask.shape != reference_shape:
                    print("Shape of", lr, "do not match for mask", p1['mask'].shape, "vs", reference_shape)
                    if lr not in lst_missmatch:
                        lst_missmatch.append(lr)
                if hr_mask.shape != reference_shape:
                    print("Shape of", hr, "do not match for mask", p2['mask'].shape, "vs", reference_shape)
                    if hr not in lst_missmatch:
                        lst_missmatch.append(hr)

    return lst_missmatch


def temporal_sinc_interpolation(data, s_range, e_range):
    """
    Interpolate the data in the temporal direction using sinc interpolation
    data is nd array, 
    s_range is range of the sample points
    e_range is evaluation range
    """
    dt = s_range[1] - s_range[0]
    T = s_range[-1] - s_range[0]
    print('dt', dt, '1/T', 1/T)
    print("data shape", data.shape)

    def sinc_t(data, t):
        return np.dot(data ,  np.sinc((t - s_range)/dt))

    sinc_interp = [sinc_t(data, t_eval) for t_eval in e_range]

    return np.asarray(sinc_interp)

def temporal_sinc_interpolation_ndarray(data, s_range, e_range):
    """
    Interpolate the data in the temporal direction using sinc interpolation
    data is nd array: expect T x X x Y x Z
    s_range is range of the sample points
    e_range is evaluation range
    """
    
    dt = s_range[1] - s_range[0]

    sinc_matrix =  np.sinc((e_range - s_range[:, None])/dt).transpose()

    #tensordot product
    sinc_interp = np.tensordot(sinc_matrix, data, axes = ([1], [0]))
    
    return np.asarray(sinc_interp)




def sinc_interpolation_fft(x: np.ndarray, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Fast Fourier Transform (FFT) based sinc or bandlimited interpolation.
    
    Args:
        x (np.ndarray): signal to be interpolated, can be 1D or 2D
        s (np.ndarray): time points of x (*s* for *samples*) 
        u (np.ndarray): time points of y (*u* for *upsampled*)
        
    Returns:
        np.ndarray: interpolated signal at time points *u*
    """
    num_output = len(u)

    # Compute the FFT of the input signal
    X = np.fft.rfft(x)

    # Create a new array for the zero-padded frequency spectrum
    X_padded = np.zeros(num_output // 2 + 1, dtype=complex)

    # Copy the original frequency spectrum into the zero-padded array
    X_padded[:X.shape[0]] = X

    # Compute the inverse FFT of the zero-padded frequency spectrum
    x_interpolated = np.fft.irfft(X_padded, n=num_output)

    return x_interpolated * (num_output / len(s))


def adjust_image_size(image, new_shape):
    """
    Adjust the size of the image to the new shape, assumes 4D image
    """
    old_shape = image.shape
    
    padding = []

    # pad the image
    for i in range(len(new_shape)):
        # diff positive for padding and negative for cropping
        diff = new_shape[i] - old_shape[i]
        
        if diff > 0:
            # pad the image
            pad_before = diff // 2
            pad_after = diff - pad_before
            padding.append((pad_before, pad_after))
        else:
            # no adjustment needed
            padding.append((0, 0))

        #cropping
        if diff < 0:
            t_mid = int(old_shape[i] // 2)
            cropl = int(np.floor(abs(new_shape[i]) / 2))
            cropr = int(np.ceil(abs(new_shape[i]) / 2))
            if i == 0:
                image = image[t_mid - cropl:t_mid + cropr, :, :, :]
            elif i == 1:
                image = image[:, t_mid - cropl:t_mid + cropr, :, :]
            elif i == 2:
                image = image[:, :, t_mid - cropl:t_mid + cropr, :]
            elif i == 3:
                image = image[:, :, :, t_mid - cropl:t_mid + cropr]

    # pad the image
    new_image = np.pad(image, padding, mode='constant', constant_values=0)

    print(f"Adjusted image size from {old_shape} to {new_image.shape}")
    return new_image


def k_space_sampling_timeseries():
    path_kmask = 'data/kspacemask.h5'
    path_order = 'data/order.mat'
    path_datamodel = 'data/CARDIAC/M2_2mm_step1_static_dynamic.h5'
    save_as = 'results/interpolation/M2_2mm_step1_static_dynamic_kspace_sampled.h5'

    # make batches to process data
    batchsize = 5000
    
    order = sio.loadmat(path_order)
    phs = order['phs'].squeeze()
    phs_max = np.max(phs)
    set_sampling = order['set'].squeeze()
    print( phs.shape)

    # load CFD data 
    with h5py.File(path_datamodel, mode = 'r' ) as p1:
            data = {}
            data['mask'] = np.asarray(p1['mask']).squeeze()
            for vel in ['u', 'v', 'w']:
                data[vel] = np.asarray(p1[vel])
                data[f'venc_{vel}'] = np.asarray(p1[f'{vel}_max']).squeeze()
    data['magnitude'] = np.ones_like(data['u'])

    # get shape of kspacemask
    with h5py.File(path_kmask, mode = 'r' ) as p1:
        _, xk, yk, zk = np.asarray(p1['mask'][1:3, :, :, :], dtype = np.int8).shape
    
    plt.imshow(data['u'][10, :, :, 30])
    plt.title('Original velocity image')
    plt.show()

    # crop data to the same size as kspacemask
    for vel in ['u', 'v', 'w', 'mask', 'magnitude']:
        data[vel] = adjust_image_size(data[vel], (data[vel].shape[0], xk, yk, zk))
    
    data['magnitude'] = np.ones_like(data['u'])

    print(data['u'].shape, data['mask'].shape)
    plt.imshow(data['u'][10, :, :, 60])
    plt.title('Cropped velocity image')
    plt.show()

    # interpolate data to maximum number of frames of phds
    t_range_orig = np.linspace(0, 1, data[vel].shape[0])
    t_range_interp = np.linspace(0, 1, phs_max)

    interpolated_data = {}
    for vel in ['u', 'v', 'w']:
        # use sinc interpolation
        interpolated_data[vel] = temporal_sinc_interpolation_ndarray(data[vel], t_range_orig, t_range_interp)
        interpolated_data[f'venc_{vel}'] = temporal_sinc_interpolation(data[f'venc_{vel}'], t_range_orig, t_range_interp)
        
    interpolated_data['mask'] = np.ceil(temporal_sinc_interpolation_ndarray(data['mask'], t_range_orig, t_range_interp))
    interpolated_data['magnitude'] = temporal_sinc_interpolation_ndarray(data['magnitude'], t_range_orig, t_range_interp)

    plt.imshow(data['u'][10, :, :, 60])
    plt.imshow(interpolated_data['u'][10, :, :, 60])
    plt.title('Interpolated velocity image with sinc')
    plt.show()

    # convert data to k-space
    kspace_data = {}
    for vel in ['u', 'v', 'w']:
        kspace_data[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.complex64)
        for t in range(interpolated_data[vel].shape[0]):
            kspace_data[vel][t] = fft_fcts.velocity_img_to_centered_kspace(interpolated_data[vel][t], interpolated_data['magnitude'][t], venc = interpolated_data[f'venc_{vel}'][t])

    #example
    #reconstruct example
    ex_reconstructed, _ = fft_fcts.centered_kspace_to_velocity_img(kspace_data['u'][10], interpolated_data['magnitude'][10], venc = interpolated_data[f'venc_u'][10])
    plt.imshow(ex_reconstructed[ :, :, 60])
    plt.title('Reconstructed velocity image from fft')
    plt.show()

    # now sample as given in phs
    kspace_data_sampled = {}
    for vel in ['u', 'v', 'w']:
        kspace_data_sampled[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.complex64)

    # total size
    total_size = order['phs'].squeeze().shape[0] 

    #--------------------rewrite this-------------------
    # for b in tqdm(range(0, total_size, batchsize)):

    #     indices = np.where(phs[b:b+batchsize] == 1 & set_sampling[b:b+batchsize] == 1)
    #     # load only batches for computational purposes
    #     with h5py.File(path_kmask, mode = 'r' ) as p1:
    #         print(b, b+batchsize)
    #         kspacemask = np.asarray(p1['mask'][b:b+batchsize:4, :, :, :], dtype = np.int8)
        
    #     phs_batch = phs[b:b+batchsize:4]
        
    #     print(kspacemask.shape, phs_batch.shape)


    #     # now sample as given in phs
    #     for vel in ['u', 'v', 'w']:
    #         print(kspace_data_sampled[vel].shape, kspacemask.shape, kspace_data[vel].shape, phs.shape)
    #         for i, ph in enumerate(phs_batch):
    #             # kspace_data_sampled[vel][ph -1, :, :, :] += np.multiply(kspace_data[vel][ph -1, :, :, :], kspacemask[i, :, :, :])

    #             masked_kspace = np.zeros_like(kspace_data[vel][ph -1, :, :, :]) 
    #             masked_kspace[np.where(kspacemask[i, :, :, :] != 0)] = kspace_data[vel][ph -1, :, :, :][np.where(kspacemask[i, :, :, :] != 0)]
    #             kspace_data_sampled[vel][ph -1, :, :, :] += masked_kspace

    #---------------------------------------------------
    for b in tqdm(range(0, total_size, batchsize)):
        # iterate over number of max number of phs

        phs_batch = phs[b:b+batchsize]

        # iterate over total number of phs
        for segm in range(phs_max):
            indices = np.where(np.logical_and(set_sampling[b:b+batchsize] == 1, phs_batch == segm+1))
            
            # reduce phsbatch
            phs_batch_red = phs_batch[indices]

            with h5py.File(path_kmask, mode = 'r' ) as p1:
                kspacemask = np.asarray(p1['mask'][b:b+batchsize, :, :, :], dtype = np.int8)
                kspacemask = kspacemask[indices[0], :, :, :] # reduce

            print("n segm", segm, kspacemask.shape, phs_batch_red.shape)
            k_space_redsum = np.sum(kspacemask, axis = 0)

            # now sample as given in phs
            for vel in ['u', 'v', 'w']:
                kspace_data_sampled[vel][segm] += np.multiply(kspace_data[vel][segm], k_space_redsum)

    
    # convert back to velocity image
    velocity_data_sampled = {}
    for vel in ['u', 'v', 'w']:
        velocity_data_sampled[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.float32)
        for t in range(phs_max):
            velocity_data_sampled[vel][t], _ = fft_fcts.centered_kspace_to_velocity_img(kspace_data_sampled[vel][t], interpolated_data['magnitude'][t], venc = interpolated_data[f'venc_{vel}'][t])


    plt.imshow(velocity_data_sampled['u'][10, :, :, 60])
    plt.show()

    # save to h5

    for vel in ['u', 'v', 'w']:
        h5functions.save_to_h5(save_as, vel, velocity_data_sampled[vel], expand_dims=False)
        h5functions.save_to_h5(save_as, f'venc_{vel}', interpolated_data[f'venc_{vel}'], expand_dims=False)


def k_space_static_test2():
    path_kmask = 'data/kspacemask.h5'
    path_order = 'data/order.mat'
    path_datamodel = 'data/CARDIAC/M1_2mm_step1_static_dynamic.h5'

    # make batches to process data
    batchsize = 4*100
    
    order = sio.loadmat(path_order)
    phs = order['phs'].squeeze()
    phs_max = np.max(phs)
    set_sampling = order['set'].squeeze()
    print( phs.shape)

    # load CFD data 
    with h5py.File(path_datamodel, mode = 'r' ) as p1:
            data = {}
            data['mask'] = np.asarray(p1['mask']).squeeze()
            for vel in ['u', 'v', 'w']:
                data[vel] = np.asarray(p1[vel])
                data[f'venc_{vel}'] = np.asarray(p1[f'{vel}_max']).squeeze()
    data['magnitude'] = np.ones_like(data['u'])

    with h5py.File(path_kmask, mode = 'r' ) as p1:
            _, xk, yk, zk = np.asarray(p1['mask'][1:3, :, :, :], dtype = np.int8).shape
    
    plt.imshow(data['u'][10, :, :, 30])
    plt.title('Original velocity image')
    # plt.show()

    # crop data to the same size as kspacemask
    for vel in ['u', 'v', 'w', 'mask', 'magnitude']:
        data[vel] = adjust_image_size(data[vel], (data[vel].shape[0], xk, yk, zk))

    midx = int(data['u'].shape[1]//2)
    #TODO delete this when croping works!!
    for vel in ['u', 'v', 'w', 'mask']:
        data[vel] = data[vel][:, midx-int(np.floor(xk/2)):midx+int(np.ceil(xk/2)), :, :]
        # print(data['u'].shape, data['mask'].shape)
    
    data['magnitude'] = np.ones_like(data['u'])
        # data[vel] = np.pad(data[vel], ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    print(data['u'].shape, data['mask'].shape)
    plt.imshow(data['u'][10, :, :, 60])
    plt.title('Cropped velocity image')
    # plt.show()

    # interpolate data to maximum number of frames of phds
    t_range_orig = np.linspace(0, 1, data[vel].shape[0])
    t_range_interp = np.linspace(0, 1, phs_max)

    interpolated_data = {}
    for vel in ['u', 'v', 'w']:
        # use sinc interpolation
        interpolated_data[vel] = temporal_sinc_interpolation_ndarray(data[vel], t_range_orig, t_range_interp)
        interpolated_data[f'venc_{vel}'] = temporal_sinc_interpolation(data[f'venc_{vel}'], t_range_orig, t_range_interp)
        
    interpolated_data['mask'] = np.ceil(temporal_sinc_interpolation_ndarray(data['mask'], t_range_orig, t_range_interp))
    interpolated_data['magnitude'] = temporal_sinc_interpolation_ndarray(data['magnitude'], t_range_orig, t_range_interp)

    plt.imshow(data['u'][10, :, :, 60])

    plt.imshow(interpolated_data['u'][10, :, :, 60])
    plt.title('Interpolated velocity image with sinc')
    # plt.show()

    # convert data to k-space
    # kspace_data = {}
    # for vel in ['u', 'v', 'w']:
    #     kspace_data[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.complex64)
    #     for t in range(interpolated_data[vel].shape[0]):
    #         kspace_data[vel][t] = fft_fcts.velocity_img_to_kspace(interpolated_data[vel][t], interpolated_data['magnitude'][t], venc = interpolated_data[f'venc_{vel}'][t])

    #-----------------now use only one k-space set----------------------------
    print('shapes: ', interpolated_data['u'].shape, interpolated_data['magnitude'].shape, interpolated_data[f'venc_u'].shape)
    kspace_data = {}
    kspace_data['u'] = fft_fcts.velocity_img_to_centered_kspace(interpolated_data['u'][0, :, :, :], interpolated_data['magnitude'][0, :, :, :], venc = interpolated_data[f'venc_u'][0])
    kspace_data['v'] = fft_fcts.velocity_img_to_centered_kspace(interpolated_data['v'][0, :, :, :], interpolated_data['magnitude'][0, :, :, :], venc = interpolated_data[f'venc_v'][0])
    kspace_data['w'] = fft_fcts.velocity_img_to_centered_kspace(interpolated_data['w'][0, :, :, :], interpolated_data['magnitude'][0, :, :, :], venc = interpolated_data[f'venc_w'][0])

    interpolated_data['u'] = interpolated_data['u'][0, :, :, :]
    interpolated_data['v'] = interpolated_data['v'][0, :, :, :]
    interpolated_data['w'] = interpolated_data['w'][0, :, :, :]
    interpolated_data['venc_u'] = interpolated_data['venc_u'][0]
    interpolated_data['venc_v'] = interpolated_data['venc_v'][0]
    interpolated_data['venc_w'] = interpolated_data['venc_w'][0]
    interpolated_data['magnitude'] = np.ones_like(interpolated_data['u'])


    #example
    #reconstruct example
    ex_reconstructed, _ = fft_fcts.centered_kspace_to_velocity_img(kspace_data['u'], interpolated_data['magnitude'], venc = interpolated_data[f'venc_u'])
    plt.imshow(ex_reconstructed[ :, :, 60])
    plt.title('Reconstructed velocity image from fft')
    plt.show()

    # now sample as given in phs
    kspace_data_sampled = {}
    for vel in ['u', 'v', 'w']:
        kspace_data_sampled[vel] = np.zeros((xk, yk, zk), dtype = np.complex64)

    # total size
    total_size = order['phs'].squeeze().shape[0] 

    # find all corresponding indices
    indices = np.where(np.logical_and(set_sampling == 1, phs == 1))

    with h5py.File(path_kmask, mode = 'r' ) as p1:
        k_space_mask_static = np.asarray(p1['mask'][indices[0]], dtype = np.int8)

    print(k_space_mask_static.shape)

    plt.imshow(np.abs(np.sum((k_space_mask_static[:, :, :, 60]), axis = 0)))
    plt.show()

    k_space_redsum = np.sum(k_space_mask_static, axis = 0)
    print(np.unique(k_space_redsum))

    for vel in ['u', 'v', 'w']:
        kspace_data_sampled[vel] = np.zeros((xk, yk, zk), dtype = np.complex64)
        kspace_data_sampled[vel] = np.multiply(kspace_data[vel], k_space_redsum)

    plt_x, plt_y, plt_z = np.where(k_space_redsum != 0)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(plt_x, plt_y, plt_z, c = 'r', marker = 'o')
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(kspace_data['u'][:, :, 60]))
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(kspace_data_sampled['u'][:, :, 60]))
    plt.show()


    # convert back to velocity image
    velocity_data_sampled = {}
    for vel in ['u', 'v', 'w']:
        velocity_data_sampled[vel], _ = fft_fcts.centered_kspace_to_velocity_img(kspace_data_sampled[vel], interpolated_data['magnitude'], venc = interpolated_data[f'venc_{vel}'])

    plt.imshow(velocity_data_sampled['u'][:, :, 60])
    plt.show()

    # save to h5
    save_as = 'results/interpolation/M1_2mm_step1_static_dynamic_kspace_sampled5_static3.h5'
    for vel in ['u', 'v', 'w']:
        h5functions.save_to_h5(save_as, vel, velocity_data_sampled[vel], expand_dims=False)
        # h5functions.save_to_h5(save_as, f'venc_{vel}', interpolated_data[f'venc_{vel}'], expand_dims=False)

    exit()
    # for b in tqdm(range(0, total_size, batchsize)):

    #     # load only batches for computational purposes
    #     with h5py.File(path_kmask, mode = 'r' ) as p1:
    #         print(b, b+batchsize)
    #         kspacemask = np.asarray(p1['mask'][b:b+batchsize:4, :, :, :], dtype = np.int8)
        
    #     phs_batch = phs[b:b+batchsize:4]
        
    #     print(kspacemask.shape, phs_batch.shape)

    #     tk, xk, yk, zk = kspacemask.shape

    #     # now sample as given in phs
    #     for vel in ['u', 'v', 'w']:
    #         print(kspace_data_sampled[vel].shape, kspacemask.shape, kspace_data[vel].shape, phs.shape)
    #         for i, ph in enumerate(phs_batch):
    #             # kspace_data_sampled[vel][ph -1, :, :, :] += np.multiply(kspace_data[vel][ph -1, :, :, :], kspacemask[i, :, :, :])

    #             masked_kspace = np.zeros_like(kspace_data[vel][ph -1, :, :, :]) 
    #             masked_kspace[np.where(kspacemask[i, :, :, :] != 0)] = kspace_data[vel][ph -1, :, :, :][np.where(kspacemask[i, :, :, :] != 0)]
    #             kspace_data_sampled[vel][ph -1, :, :, :] += masked_kspace

        
    # # convert back to velocity image
    # velocity_data_sampled = {}
    # for vel in ['u', 'v', 'w']:
    #     velocity_data_sampled[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.float32)
    #     for t in range(phs_max):
    #         velocity_data_sampled[vel][t], _ = fft_fcts.kspace_to_velocity_img(kspace_data_sampled[vel][t], interpolated_data['magnitude'][t], venc = interpolated_data[f'venc_{vel}'][t])


    plt.imshow(velocity_data_sampled['u'][10, :, :, 60])
    plt.show()

    # save to h5
    save_as = 'results/interpolation/M1_2mm_step1_static_dynamic_kspace_sampled5.h5'
    for vel in ['u', 'v', 'w']:
        h5functions.save_to_h5(save_as, vel, velocity_data_sampled[vel], expand_dims=False)
        h5functions.save_to_h5(save_as, f'venc_{vel}', interpolated_data[f'venc_{vel}'], expand_dims=False)






if __name__ == '__main__':

    csv_dir = 'data/CARDIAC'
    data_dir = 'data/CARDIAC'

    csv_files = [f for f in glob.glob(f"{csv_dir}/*.csv")]
    lst_missmatch = []
    for csv_file in csv_files:
        lst_missmatch.extend(check_csv_patch_compatibility(csv_file))
    
    lst_missmatch = list(set(lst_missmatch)) #exlcude duplicates
    print("Checked all csv files. Missmatch in the following files: ", lst_missmatch)

    mag_colnames = [ 'mag_u', 'mag_v', 'mag_w']

    # make new datasets with flower magnitude
    new_model_names = ['M1_2mm_step2_cloudmagnRot_boxavg_LRfct_noise.h5', 'M2_2mm_step2_cloudmagnRot_boxavg_LRfct_noise.h5', 'M3_2mm_step2_cloudmagnRot_boxavg_LRfct_noise.h5', 'M4_2mm_step2_cloudmagnRot_boxavg_LRfct_noise.h5']
    merge_models    = ['M1_2mm_step2_invivoP01_boxavg_LRfct_noise.h5' , 'M2_2mm_step2_invivoP04_boxavg_LRfct_noise.h5', 'M3_2mm_step2_invivoP03_boxavg_LRfct_noise.h5', 'M4_2mm_step2_invivoP02_boxavg_LRfct_noise.h5']
    flower_magn = 'data/flower_magn_data_4D_spatial_rotated_Alexcode2.h5'

    for new_model, orig_model in zip(new_model_names, merge_models):
        MX = new_model.split('_')[0]
        
        with h5py.File(flower_magn, mode = 'r' ) as h5:
            flower_magn_data = np.array(h5[MX])

        # if os.path.isfile(f'{data_dir}/{new_model}'):
        #     print(f"File {new_model} already exists. Not writing to it.")
        #     continue
        create_h5_file(data_dir, new_model)

        with h5py.File(f'{data_dir}/{new_model}', mode = 'a' ) as new_data:
            with h5py.File(f'{data_dir}/{orig_model}', mode = 'r' ) as orig_data:
                for key in orig_data.keys():
                    print(key, orig_data.get(key).shape)
                    new_data.create_dataset(key, data = orig_data.get(key))
                
                for mag in mag_colnames:
                    del new_data[mag]

                    if flower_magn_data.shape != new_data[mag[-1]].shape:
                        print('Check model', new_model, 'for', mag, 'shape', flower_magn_data.shape, 'vs', new_data[mag[-1]].shape)
                        flower_magn_data = adjust_image_size(flower_magn_data, new_data[mag[-1]].shape)
                        
                    new_data.create_dataset(mag, data = flower_magn_data)

    exit()
    k_space_sampling_timeseries()
    exit()
    path_kmask = 'data/kspacemask.h5'
    path_order = 'data/order.mat'
    path_datamodel = 'data/CARDIAC/M1_2mm_step1_static_dynamic.h5'

    # make batches to process data
    batchsize = 4*100
    
    order = sio.loadmat(path_order)
    phs = order['phs'].squeeze()
    phs_max = np.max(phs)
    set_sampling = order['set'].squeeze()
    print(order.keys())
    print( phs.shape)

    exit()
    # load CFD data 
    with h5py.File(path_datamodel, mode = 'r' ) as new_data:
            data = {}
            data['mask'] = np.asarray(new_data['mask']).squeeze()
            for vel in ['u', 'v', 'w']:
                data[vel] = np.asarray(new_data[vel])
                data[f'venc_{vel}'] = np.asarray(new_data[f'{vel}_max']).squeeze()
    data['magnitude'] = np.ones_like(data['u'])

    with h5py.File(path_kmask, mode = 'r' ) as new_data:
            _, xk, yk, zk = np.asarray(new_data['mask'][1:3, :, :, :], dtype = np.int8).shape
    
    plt.imshow(data['u'][10, :, :, 30])
    plt.title('Original velocity image')
    plt.show()

    # crop data to the same size as kspacemask
    for vel in ['u', 'v', 'w', 'mask', 'magnitude']:
        data[vel] = adjust_image_size(data[vel], (data[vel].shape[0], xk, yk, zk))

    midx = int(data['u'].shape[1]//2)
    #TODO delete this when croping works!!
    for vel in ['u', 'v', 'w', 'mask']:
        data[vel] = data[vel][:, midx-int(np.floor(xk/2)):midx+int(np.ceil(xk/2)), :, :]
        # print(data['u'].shape, data['mask'].shape)
    
    data['magnitude'] = np.ones_like(data['u'])
        # data[vel] = np.pad(data[vel], ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    print(data['u'].shape, data['mask'].shape)
    plt.imshow(data['u'][10, :, :, 60])
    plt.title('Cropped velocity image')
    plt.show()

    # interpolate data to maximum number of frames of phds
    t_range_orig = np.linspace(0, 1, data[vel].shape[0])
    t_range_interp = np.linspace(0, 1, phs_max)

    interpolated_data = {}
    for vel in ['u', 'v', 'w']:
        # use sinc interpolation
        interpolated_data[vel] = temporal_sinc_interpolation_ndarray(data[vel], t_range_orig, t_range_interp)
        interpolated_data[f'venc_{vel}'] = temporal_sinc_interpolation(data[f'venc_{vel}'], t_range_orig, t_range_interp)
        
    interpolated_data['mask'] = np.ceil(temporal_sinc_interpolation_ndarray(data['mask'], t_range_orig, t_range_interp))
    interpolated_data['magnitude'] = temporal_sinc_interpolation_ndarray(data['magnitude'], t_range_orig, t_range_interp)

    plt.imshow(data['u'][10, :, :, 60])

    plt.imshow(interpolated_data['u'][10, :, :, 60])
    plt.title('Interpolated velocity image with sinc')
    plt.show()

    # convert data to k-space
    kspace_data = {}
    for vel in ['u', 'v', 'w']:
        kspace_data[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.complex64)
        for t in range(interpolated_data[vel].shape[0]):
            kspace_data[vel][t] = fft_fcts.velocity_img_to_kspace(interpolated_data[vel][t], interpolated_data['magnitude'][t], venc = interpolated_data[f'venc_{vel}'][t])

    #example
    #reconstruct example
    ex_reconstructed, _ = fft_fcts.kspace_to_velocity_img(kspace_data['u'][10], interpolated_data['magnitude'][10], venc = interpolated_data[f'venc_u'][10])
    plt.imshow(ex_reconstructed[ :, :, 60])
    plt.title('Reconstructed velocity image from fft')
    plt.show()

    # now sample as given in phs
    kspace_data_sampled = {}
    for vel in ['u', 'v', 'w']:
        kspace_data_sampled[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.complex64)

    # total size
    total_size = order['phs'].squeeze().shape[0] 

    for b in tqdm(range(0, total_size, batchsize)):

        # load only batches for computational purposes
        with h5py.File(path_kmask, mode = 'r' ) as new_data:
            print(b, b+batchsize)
            kspacemask = np.asarray(new_data['mask'][b:b+batchsize:4, :, :, :], dtype = np.int8)
        
        phs_batch = phs[b:b+batchsize:4]
        
        print(kspacemask.shape, phs_batch.shape)

        tk, xk, yk, zk = kspacemask.shape

        # now sample as given in phs
        for vel in ['u', 'v', 'w']:
            print(kspace_data_sampled[vel].shape, kspacemask.shape, kspace_data[vel].shape, phs.shape)
            for i, ph in enumerate(phs_batch):
                # kspace_data_sampled[vel][ph -1, :, :, :] += np.multiply(kspace_data[vel][ph -1, :, :, :], kspacemask[i, :, :, :])

                masked_kspace = np.zeros_like(kspace_data[vel][ph -1, :, :, :]) 
                masked_kspace[np.where(kspacemask[i, :, :, :] != 0)] = kspace_data[vel][ph -1, :, :, :][np.where(kspacemask[i, :, :, :] != 0)]
                kspace_data_sampled[vel][ph -1, :, :, :] += masked_kspace

        
    # convert back to velocity image
    velocity_data_sampled = {}
    for vel in ['u', 'v', 'w']:
        velocity_data_sampled[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.float32)
        for t in range(phs_max):
            velocity_data_sampled[vel][t], _ = fft_fcts.kspace_to_velocity_img(kspace_data_sampled[vel][t], interpolated_data['magnitude'][t], venc = interpolated_data[f'venc_{vel}'][t])


    plt.imshow(velocity_data_sampled['u'][10, :, :, 60])
    plt.show()

    # save to h5
    save_as = 'results/interpolation/M1_2mm_step1_static_dynamic_kspace_sampled4.h5'
    for vel in ['u', 'v', 'w']:
        h5functions.save_to_h5(save_as, vel, velocity_data_sampled[vel], expand_dims=False)
        h5functions.save_to_h5(save_as, f'venc_{vel}', interpolated_data[f'venc_{vel}'], expand_dims=False)

    exit()
    if False: 
    # compute sinc interpolation

        #load data
        path = 'data/CARDIAC/M4_2mm_step2_static_dynamic.h5'

        
        with h5py.File(path, mode = 'r' ) as new_data:
            vx = np.asarray(new_data['u'])[::2]
            vy = np.asarray(new_data['v'])[::2]
            vz = np.asarray(new_data['w'])[::2]
            mask = np.asarray(new_data['mask']).squeeze()
        
        # sinc interpolate this
        # t_range = np.linspace(0, 1, vx.shape[0])
        eval_points = np.linspace(0, 1, 2*vx.shape[0]) # increasing resolution by 2x
        t_range = eval_points[::2]
        vx_sinc_interp = temporal_sinc_interpolation_ndarray(vx, t_range, eval_points)
        vy_sinc_interp = temporal_sinc_interpolation_ndarray(vy, t_range, eval_points)
        vz_sinc_interp = temporal_sinc_interpolation_ndarray(vz, t_range, eval_points)
        # exp_mask = np.zeros_like(sinc_interp)
        # exp_mask[::2] = mask
        # exp_mask[1::2] = mask
        # print(sinc_interp.shape)

        save_as = 'results/interpolation/M4_2mm_step2_reconstructed_sinc_interpolation.h5'
        h5functions.save_to_h5(save_as, 'u', vx_sinc_interp, expand_dims=False)
        h5functions.save_to_h5(save_as, 'v', vy_sinc_interp, expand_dims=False)
        h5functions.save_to_h5(save_as, 'w', vz_sinc_interp, expand_dims=False)
        # plot mean speed

        # mean_vx = np.average(vx, axis = (1, 2, 3), weights=mask)
        # mean_sinc = np.average(sinc_interp, axis = (1, 2, 3), weights=exp_mask)

        # plt.plot(t_range, mean_vx, 'o-', label = 'Mean speed original')
        # plt.plot(eval_points, mean_sinc, 'o-', label = 'Mean speed sinc interpolated')
        # plt.show()

    exit()
    if True:
        n = 50
        t_range = np.linspace(-5, 5, n)
        fct_values = np.sin(t_range) * np.exp(-t_range**2/2)
        eval_points = np.linspace(-5, 5, n*3)

        sinc_interp = temporal_sinc_interpolation(fct_values, t_range, eval_points)
        print(len(sinc_interp))
        sinc_fft = sinc_fft(fct_values)
        print(sinc_fft, len(sinc_fft))


        # plt.plot(range(len(sinc_fft)), sinc_fft, 'o-', label = 'Sinc FFT')
        # plt.show()
        sind_interp2 = sinc_interpolation(fct_values, t_range, eval_points)
        sinc_interp3 = sinc_interpolation_fft(fct_values, t_range, eval_points)
        plt.plot(eval_points, sind_interp2, 'o-', label = 'Sinc interpolation git')
        # plt.plot(eval_points, sinc_interp3, 'o-', label = 'Sinc interpolationbgit fft')
        plt.plot(eval_points, sinc_interp, 'o-', label = 'Sinc interpolation')
        # plt.plot(t_range, np.sinc(t_range))
        plt.plot(t_range, fct_values, 'o-', label = 'Function orginial')
        plt.legend()
        plt.show()

        exit()
        #------------test for 2d sinc interpolation

        # create a 2d function
        #create function
        t = np.linspace(-5, 5, n)
        y = np.linspace(-5, 5, n)
        T, Y = np.meshgrid(t, y)
        Z = np.exp(-T/2) * np.cos(Y)

        # set interpolation points
        t_eval = np.linspace(-5, 5, n*3) # only increasing time points
        y_eval = np.linspace(-5, 5, n)
        T_eval, Y_eval = np.meshgrid(t_eval, y_eval)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contourf3D(T, Y, Z, 50)
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('original function')
        plt.show()


        sinc_2d = sinc_interpolation(Z, t, t_eval)
        sinc_2d_2 = temporal_sinc_interpolation(Z, t, t_eval)

        ax = plt.axes(projection='3d')
        ax.contourf3D(T_eval, Y_eval, sinc_2d, 50 )
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('sinc interpolation (git)')
        plt.show()

        ax = plt.axes(projection='3d')
        ax.contourf3D(T_eval, Y_eval, sinc_2d_2.transpose(), 50 )
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('sinc interpolation - my implementation')
        plt.show()

    #------try 4d function
    # create a 4d function
    #create function

    data = np.random.rand(20, 10, 10, 10)
    t = np.linspace(0, 1, data.shape[0])
    x = np.linspace(0, 10, data.shape[1])
    y = np.linspace(0, 10, data.shape[2])
    z = np.linspace(0, 10, data.shape[3])

    t_eval = np.linspace(0, 1, data.shape[0]*2)
    
    sinc_interpol = sinc_interpolation(data, t, t_eval)
    print(sinc_interpolation.shape)

    exit()
    # load h5 file
    M4_path = 'data/CARDIAC/M1_2mm_step1_static_dynamic.h5'

    M4_interpolated_path = 'data/CARDIAC/M1_2mm_step05_static_dynamic_interpolated.h5'
    data = {}
    with h5py.File(M4_interpolated_path, mode = 'r' ) as new_data:
        print(new_data.keys())
        for vel in ['u', 'v', 'w']:
            data[vel] = np.asarray(new_data[vel]).squeeze()
    
    with h5py.File(M4_path, mode = 'r' ) as new_data:
        mask = np.asarray(new_data['mask']).squeeze()
    
    mask_hr = np.zeros_like(data['u'])
    mask_hr[::2] = mask
    mask_hr[1::2] = mask

    mean_speed = calculate_mean_speed(data['u'], data['v'], data['w'], mask_hr)

    plt.plot(mean_speed, '.-', label = 'High resolution', color = 'black')
    plt.title('Mean speed')
    plt.show()
    exit()
    # make simple linear interpolation
    M4_path = 'data/CARDIAC/M1_2mm_step1_static_dynamic.h5'
    M4_interpolated_path = 'data/CARDIAC/M1_2mm_step05_static_dynamic_interpolated.h5'
    with h5py.File(M4_path, mode = 'r' ) as new_data:
        data = {}
        print(new_data.keys())
        for vel in ['u', 'v', 'w']:
            T,  t, y, z = np.asarray(new_data[vel]).squeeze().shape
            data[vel] = temporal_linear_interpolation_np(np.asarray(new_data[vel]).squeeze(), (2*T, t, y, z))

    create_h5_file('data\CARDIAC', 'M1_2mm_step05_static_dynamic_interpolated.h5')
    with h5py.File(M4_interpolated_path, mode = 'a' ) as new_data:
        for key in data.keys():
            new_data.create_dataset(key, data = data[key])

    exit()
    mag_colnames = [ 'mag_u', 'mag_v', 'mag_w']

    mag_max = [117, 137, 77, 105]
    
    csv_dir = 'data/CARDIAC'
    data_dir = 'data/CARDIAC'

    csv_files = [f for f in glob.glob(f"{csv_dir}/*.csv")]
    lst_missmatch = []
    for csv_file in csv_files:
        lst_missmatch.extend(check_csv_patch_compatibility(csv_file))
    
    lst_missmatch = list(set(lst_missmatch)) #exlcude duplicates
    print("Checked all csv files. Missmatch in the following files: ", lst_missmatch)

    mag_colnames = [ 'mag_u', 'mag_v', 'mag_w']

    # make new datasets with flower magnitude
    new_model_names = ['M1_2mm_step2_flowermagn_boxavg_LRfct_noise.h5', 'M2_2mm_step2_flowermagn_boxavg_LRfct_noise.h5', 'M3_2mm_step2_flowermagn_boxavg_LRfct_noise.h5', 'M4_2mm_step2_flowermagn_boxavg_LRfct_noise.h5']
    merge_models    = ['M1_2mm_step2_invivoP01_boxavg_LRfct_noise.h5' , 'M2_2mm_step2_invivoP04_boxavg_LRfct_noise.h5', 'M3_2mm_step2_invivoP03_boxavg_LRfct_noise.h5', 'M4_2mm_step2_invivoP02_boxavg_LRfct_noise.h5']
    flower_magn = 'data/flower_magn_data.h5'

    for new_model, orig_model in zip(new_model_names, merge_models):
        MX = new_model.split('_')[0]
        
        with h5py.File(flower_magn, mode = 'r' ) as new_data:
            flower_magn_data = np.array(new_data[MX])

        # if os.path.isfile(f'{data_dir}/{new_model}'):
        #     print(f"File {new_model} already exists. Not writing to it.")
        #     continue
        create_h5_file(data_dir, new_model)

        with h5py.File(f'{data_dir}/{new_model}', mode = 'a' ) as new_data:
            with h5py.File(f'{data_dir}/{orig_model}', mode = 'r' ) as orig_data:
                for key in orig_data.keys():
                    print(key, orig_data.get(key).shape)
                    new_data.create_dataset(key, data = orig_data.get(key))
                
                for mag in mag_colnames:
                    del new_data[mag]
                    new_data.create_dataset(mag, data = flower_magn_data)


    

    #
    # for file in lst_missmatch:
    #     with h5py.File(f'{data_dir}/{file}', mode = 'a' ) as p1:
    #         for mag in mag_colnames:
    #             # new_mag = np.pad(np.array(p1[mag]), ((0, 0), (0, 1), (0, 1), (0, 1)), mode='constant', constant_values=0)
    #             new_mag = np.array(p1[mag])[:, :-5, :-5, :-5]
    #             print("Padded", mag, p1[mag].shape, new_mag.shape,' goal ', p1['u'].shape)
    #             #delete old mag
    #             del p1[mag]
    #             # add new mag
    #             h5functions.save_to_h5(f'{data_dir}/{file}', mag, new_mag, expand_dims=False)
    #             #save_to_h5(output_filename, "mag_u", mag_u, expand_dims=False)


    # file = data_dir + '/M3_2mm_step2_invivoP03_boxavg_HRfct.h5'
    # with h5py.File(file, mode = 'a' ) as p1:
    #     # del p1['mag_u']
    #     new_mag = np.array(p1['mag_v'])
    #     h5functions.save_to_h5(file, 'mag_u', new_mag, expand_dims=False)



    # white_gauss_noise = "data/CARDIAC/M4_2mm_step2_invivoP02_boxavg_LRfct_noise.h5"
    # no_noise = "data/CARDIAC/M4_2mm_step2_invivoP02_boxavg_LRfct_NOnoise_TBD.h5" 
    # # in_vivo = "C:\Users\piacal\Code\PipelineDICOMtoh5\2.DICOMtoMAT\Python\data_test2.h5"

    # with h5py.File(white_gauss_noise, mode = 'r' ) as p1:
    #     with h5py.File(no_noise, mode = 'r' ) as p2:
    #             g_noise = np.asarray(p1['u'])
    #             no_noise = np.asarray(p2['u'])
    #             mask = np.asarray(p1['mask']).squeeze()

    #             for key in p2.keys():
    #                 print(key, p1.get(key).shape)


    # plot histogram of noise
    # plt.hist(g_noise[np.where(mask ==1)].ravel(), bins = 100, alpha = 0.5, label = 'White Gaussian noise')
    # plt.hist(no_noise[np.where(mask ==1)].ravel(), bins = 100, alpha = 0.5, label = 'Noise from conversion')

    # plt.legend()
    # plt.show()

    if False: 
        base_path = 'data/CARDIAC'
        path1 = f'{base_path}/Temporal16MODEL2_2mm_step2_invivoP04_boxavg.csv'
        path2 = f'{base_path}/Temporal16MODEL3_2mm_step2_invivoP03_boxavg.csv'
        output_filename = f'{base_path}/Temporal16MODEL23_2mm_step2_invivoP04P03_boxavg.csv'
        
        concatenate_csv_files([path1, path2], output_filename)



    #-----------Create data with invivo magn----------------
    if False: 
        path_to_save =      'data/CARDIAC/M1_2mm_step2_invivoP01_boxavg_HRfct.h5'
        in_vivo_magn_path = 'data/CARDIAC/M1_2mm_step2_invivoP01_magnitude.h5'
        box_avg_path =      'data/CARDIAC/M1_2mm_step2_boxavg0_HR.h5'


        keys_to_delete = [ 'mag_u', 'mag_v', 'mag_w']

        # create a new file
        create_h5_file('data/CARDIAC/', 'M1_2mm_step2_invivoP01_boxavg_HRfct.h5')

        # merge the data from box averaged data
        temporal_downsampling.merge_data_to_h5(path_to_save, box_avg_path)

        # delete magnitude 
        temporal_downsampling.delete_data_from_h5(path_to_save, keys_to_delete)
        
        # put in invivo magnitude
        temporal_downsampling.merge_data_to_h5(path_to_save, in_vivo_magn_path)
    




    #---------------------------

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

    
    # hr_file = 'data/CARDIAC/M1_2mm_step2_static_dynamic.h5'

    # with h5py.File(hr_file, mode = 'r' ) as p1:
    #       hr_u = np.asarray(p1['u']) 
    #       hr_v = np.asarray(p1['v'])
    #       hr_w = np.asarray(p1['w'])

    # t_range = (0, 1, hr_u.shape[0])
    # dt_window = 0.00
    # mean_u, mean_v, mean_w, xx, xy, xz, yy, yz, zz = calculate_RST(hr_u, hr_v, hr_w, dt_window, np.linspace(0, 1, hr_u.shape[0]))
    # save_meanu_RST_to_vti("0", mean_u, mean_v, mean_w, xx, xy, xz, yy, yz, zz, 'results/data/RST_M1_2mm_step2_static_dynamic')

    

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
            with h5py.File(in_vivo_path, mode = 'r' ) as new_data:
                        # print(p1.keys())
                        print('M', m)
                        print('shape', new_data['u'].shape)
                        print(np.array(new_data['dx']))
                        mask =  np.asarray(new_data['mask'])
                        temporal_mask = mask.copy()

                        # data_original['mask'] = temporal_mask
                        for vel, venc, mag in zip(vel_colnames, venc_colnames, mag_colnames):
                            data_original[vel] = np.asarray(new_data[vel])
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

   