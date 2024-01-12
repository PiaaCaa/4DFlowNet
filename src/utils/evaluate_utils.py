import tensorflow as tf
import numpy as np
import time
import os
import cv2
import h5py
import scipy
from scipy.signal import convolve2d
from scipy.ndimage import convolve
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.ndimage import binary_erosion
from matplotlib import pyplot as plt

from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from prepare_data.visualize_utils import generate_gif_volume
from Network.loss_utils import calculate_divergence
from utils.colors import *




def normalize_to_0_1(data):
    """
    Normalize data to 0-1 range
    """
    return (np.array(data, dtype=float)- np.min(data))/(np.max(data)-np.min(data))

def signal_to_noise_ratio_db(Px, Pn):
    #TODO
    return 10*np.log(Px/Pn)

def signaltonoise_fluid_region(data, mask):
    assert len(data.shape) == 3 # look at three dimensional data
    return signaltonoise(normalize_to_0_1(data)[np.where(mask==1)], normalize_to_0_1(data)[np.where(mask ==0)], axis=0)

def signaltonoise(fluid_region, non_fluid_region, axis=0, ddof=0):
    '''
    source: https://stackoverflow.com/questions/63177236/how-to-calculate-signal-to-noise-ratio-using-python
    '''
    m = fluid_region.mean(axis)
    sd = non_fluid_region.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def signaltonoise_db(a, axis=0, ddof=0):
    return 20*np.log10(np.abs(signaltonoise(a, axis, ddof)))

def peak_signal_to_noise_ratio(img, noisy_img):
    #TODO
    ''' 
    Compute PSNR with PSNR=20log10(max()/RMSE)
    '''
    mse = np.mean((img - noisy_img) ** 2)
    max_pixel = np.max(img)-np.min(img) #since smallest values can be smaller than 0
    psnr = 20*np.log10(max_pixel/np.sqrt(mse))
    return psnr

def cv2_psnr(img, noisy_img):
    '''
    Compute PSNR with cv2 library
    '''
    #TODO check against own method
    return cv2.PSNR(img, noisy_img)   

def power_of_signal():
    '''
    Compute power of signal
    '''
    #TODO
    # SNRdb = 10 log SNR
    # SNRdb / 10 = log SNR
    # SNR = 10 ^ (SNRdb/10
    # Pn = Pn / SNR
    # Pn = variance
    # Deconstruct the complex numbers into real and imaginary
    mag_signal = np.abs(imgfft)
    
    signal_power = np.mean((mag_signal) ** 2)

    logSNR = targetSNRdb / 10
    snr = 10 ** logSNR

    noise_power = signal_power / snr
    return


# Crop mask to match desired shape * downsample
def crop_gt(gt, desired_shape):
    '''
    This function crops the ground truth to match the desired shape.
    It assumes that the ground truth is a 4D array.'''
    crop = np.array(gt.shape) - np.array(desired_shape)
    if crop[0]:
        gt = gt[1:-1,:,:]
    if crop[1]:
        gt = gt[:,1:-1,:]
    if crop[2]:
        gt = gt[:,:,1:-1]
    if len(crop)>3 and crop[3]:
        gt = gt[:,:,:, 1:-1]
        
    return gt

def random_indices3D(mask, n):
    '''
    This function generates random indices in a 3D mask based on a given threshold.
    It assumes that the mask is a 3D array.
    The function randomly selects 'n' samples from the mask that have values greater than the threshold.
    It returns the x, y, and z indices of the selected samples.
    '''

    assert(len(mask.shape)==3) # Ensure that the mask is 3D

    mask_threshold = 0.9
    sample_pot = np.where(mask > mask_threshold)  # Find indices where mask values are greater than the threshold
    rng = np.random.default_rng()

    # Sample 'n' random samples without replacement
    sample_idx = rng.choice(len(sample_pot[0]), replace=False, size=n)

    # Get the x, y, and z indices of the selected samples
    x_idx = sample_pot[0][sample_idx]
    y_idx = sample_pot[1][sample_idx]
    z_idx = sample_pot[2][sample_idx]
    return x_idx, y_idx, z_idx


def sigmoid(x):
  '''
  Sigmoid function
  '''
  return 1 / (1 + np.exp(-x))


def create_dynamic_mask(mask, n_frames):
    '''
    from static mask create dynamic mask of shape (n_frames, h, w, d)
    '''
    assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
    print('Create static temporal mask.')
    return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)


def calculate_relative_error_np(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    '''
    Relative error calculation for numpy arrays as in training
    '''
    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5

    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = np.sqrt(u_diff + v_diff + w_diff)
    actual_speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi)) 

    # actual speed can be 0, resulting in inf
    relative_speed_loss = diff_speed / (actual_speed + epsilon)
    
    # Make sure the range is between 0 and 1
    relative_speed_loss = np.clip(relative_speed_loss, 0., 1.)

    # Apply correction, only use the diff speed if actual speed is zero
    condition = np.not_equal(actual_speed, np.array(tf.constant(0.)))
    corrected_speed_loss = np.where(condition, relative_speed_loss, diff_speed)

    multiplier = 1e4 # round it so we don't get any infinitesimal number
    corrected_speed_loss = np.round(corrected_speed_loss * multiplier) / multiplier
    # print(corrected_speed_loss)
    
    # Apply mask
    # binary_mask_condition = (mask > threshold)
    binary_mask_condition = np.equal(binary_mask, 1.0)          
    corrected_speed_loss = np.where(binary_mask_condition, corrected_speed_loss, np.zeros_like(corrected_speed_loss))
    # print(found_indexes)
    # Calculate the mean from the total non zero accuracy, divided by the masked area
    # reduce first to the 'batch' axis
    mean_err = np.sum(corrected_speed_loss, axis=(1,2,3)) / (np.sum(binary_mask, axis=(0,1,2)) + 1) 

    # now take the actual mean
    # mean_err = tf.reduce_mean(mean_err) * 100 # in percentage
    mean_err = mean_err * 100

    return mean_err

def calculate_relative_error_normalized(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    '''
    Calculate relative error with tanh as normalization
    '''

    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5 

    if len(binary_mask.squeeze().shape) ==3:
        print('Create temporal mask to calculate relative error')
        binary_mask = create_dynamic_mask(binary_mask, u_hi.shape[0])

    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = np.sqrt(u_diff + v_diff + w_diff)
    actual_speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi)) 

    print("max/min before tanh", np.max(diff_speed / (actual_speed + epsilon)), np.min(diff_speed / (actual_speed + epsilon)))

    # actual speed can be 0, resulting in inf
    #relative_speed_loss = np.arctan(diff_speed / (actual_speed + epsilon))
    relative_speed_loss = np.tanh(diff_speed / (actual_speed + epsilon))
    print("max/min after tanh", np.max(relative_speed_loss), np.min(relative_speed_loss))
    # Make sure the range is between 0 and 1
    #relative_speed_loss = np.clip(relative_speed_loss, 0., 1.)

    # Apply correction, only use the diff speed if actual speed is zero
    condition = np.not_equal(actual_speed, np.array(tf.constant(0.)))
    corrected_speed_loss = np.where(condition, relative_speed_loss, diff_speed)

    multiplier = 1e4 # round it so we don't get any infinitesimal number
    corrected_speed_loss = np.round(corrected_speed_loss * multiplier) / multiplier
    # print(corrected_speed_loss)
    
    # Apply mask
    # binary_mask_condition = (mask > threshold)
    binary_mask_condition = np.equal(binary_mask, 1.0)          
    corrected_speed_loss = np.where(binary_mask_condition, corrected_speed_loss, np.zeros_like(corrected_speed_loss))
    # print(found_indexes)
    # Calculate the mean from the total non zero accuracy, divided by the masked area
    # reduce first to the 'batch' axis
    mean_err = np.sum(corrected_speed_loss, axis=(1,2,3)) / (np.sum(binary_mask, axis=(1, 2, 3)) + 1) 

    # now take the actual mean
    # mean_err = tf.reduce_mean(mean_err) * 100 # in percentage
    mean_err = mean_err * 100

    return mean_err




def get_fluid_region_points(data, binary_mask):
    '''
    reshapes input such that we get data of form frames, n_fluid_points
    '''
    assert len(binary_mask.shape) == 1 #remove this function from here
    if len(binary_mask.squeeze().shape) ==3:
            binary_mask = create_dynamic_mask(binary_mask, data.shape[0])
        
    points_in_mask = np.where(binary_mask !=0)
    return data[:, points_in_mask[1], points_in_mask[2], points_in_mask[3]].reshape(data.shape[0], -1)

def get_fluid_region_points_frame(data_frame, binary_mask):
    '''
    returns flattened array with all the fluid boundary points in 3D data frame
    '''
    assert len(binary_mask.shape) == 3 # mask should be 3D
    assert len(data_frame.shape) == 3 # data should be 3D
        
    return data_frame[np.where(binary_mask != 0 )].flatten()



def calculate_rmse(pred,gt, binary_mask, return_std= False):
    '''
    Calculate root mean squared error between prediction and ground truth for each frame
    i.e. rmse(t) = sqrt((pred - gt)**2/N), where N number of point in fluid region
    If return_std is set to true, the standard deviation of pred - gt)**2 is returned as well
    '''
    if len(pred.shape)==3: pred = np.expand_dims(pred, 0)
    if len(gt.shape)==3:  gt = np.expand_dims(gt, 0)
    
    if len(binary_mask.squeeze().shape) ==3:
        print('Create temporal mask for RMSE caculation', binary_mask.shape, pred.shape, gt.shape )
        binary_mask = create_dynamic_mask(binary_mask, pred.shape[0])
        print('Reshaped to', binary_mask.shape)
    
    binary_mask[np.where(binary_mask > 0.5)] = 1
    binary_mask[np.where(binary_mask <= 0.5)] = 0
    
    bool_mask = np.equal(binary_mask, 1.0)
    # points_in_mask = np.where(binary_mask !=0)

    # reshaped_pred = pred[:, points_in_mask[1], points_in_mask[2], points_in_mask[3]].reshape(gt.shape[0], -1)
    # reshaped_gt     = gt[:, points_in_mask[1], points_in_mask[2], points_in_mask[3]].reshape(gt.shape[0], -1)

    rmse = np.sqrt(np.mean((pred - gt)**2, axis = (1, 2, 3), where=bool_mask))
    
    if return_std:
        #std(a) = sqrt(mean(x)), where x = abs(a - a.mean())**2.
        var = np.std((pred - gt)**2, axis = (1, 2, 3), where=bool_mask)
        # var = np.std((reshaped_pred - np.repeat(np.expand_dims(np.mean(reshaped_gt, axis=1), -1), reshaped_pred.shape[1], axis = 1))**2, axis = 1) 
        return rmse, var
    return rmse

def calculate_pointwise_error(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    '''
    Returns a relative pointswise error and a dictionary with the absolute difference between prediction and ground truth
    '''
    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5
    if len(binary_mask.squeeze().shape) ==3:
        binary_mask = create_dynamic_mask(binary_mask, u_hi.shape[0])

    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = np.sqrt(u_diff + v_diff + w_diff)
    actual_speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi)) 

    # actual speed can be 0, resulting in inf
    relative_speed_loss = diff_speed / (actual_speed + epsilon)
    
    # Make sure the range is between 0 and 1
    #relative_speed_loss = np.clip(relative_speed_loss, 0., 1.)

    idx_mask = np.where(binary_mask == 0)
    relative_speed_loss[:,idx_mask[1], idx_mask[2], idx_mask[3]] = 0

    error_absolut = {} 
    error_absolut["u"] = np.sqrt(u_diff)
    error_absolut["v"] = np.sqrt(v_diff)
    error_absolut["w"] = np.sqrt(w_diff)
    error_absolut["speed"] = np.abs(np.sqrt(np.square(u_pred) + np.square(v_pred) + np.square(w_pred)) - actual_speed)

    return relative_speed_loss, error_absolut


def calculate_mean_speed(u_hi, v_hi, w_hi, binary_mask):
    '''
    Calculate mean speed of given values. Assumption: Values are in m/sec and mean speed returned in cm/sec
    Important: Set values of u, v, w outside of fluid region to zero 
    '''
    if len(binary_mask.squeeze().shape) ==3:
        binary_mask = create_dynamic_mask(binary_mask, u_hi.shape[0])


    u_hi = np.multiply(u_hi, binary_mask)
    v_hi = np.multiply(v_hi, binary_mask)
    w_hi = np.multiply(w_hi, binary_mask)

    speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi))
    mean_speed = np.sum(speed, axis=(1,2,3)) / (np.sum(binary_mask, axis=(1, 2, 3)) + 1) *100
    return mean_speed

def compare_mask_and_velocitymask(u_hi, v_hi, w_hi, binary_mask):
    '''
    Compares the given binary mask with created mask on the nonzero values of u, v and w and returns the overlap mask
    '''

    # create binary mask for u, v and w
    overlap_mask= np.zeros_like(u_hi)
    overlap_mask[np.where(u_hi != 0)] = 1
    overlap_mask[np.where(v_hi != 0)] = 1
    overlap_mask[np.where(w_hi != 0)] = 1

    mask = overlap_mask.copy()    
    extended_mask = np.repeat(binary_mask, 3, axis=0)

    # 0: overlap of zero values, 1: overlap of nonzero values, 2: created mask has value and binary mask odes not, 3: binary mask has value and created mask does not
    overlap_mask[np.where((extended_mask == 0) & (overlap_mask == 1))] = 2
    overlap_mask[np.where((extended_mask == 1) & (overlap_mask == 0))] = 3
    
    return overlap_mask, mask[0].squeeze()

def calculate_k_R2( pred, gt, binary_mask):
    '''
    Calculate r^2 and k in fluid region with line y = kx+m
    Note that this takes a 3D data frame as input, i.e. it only considers a single frame
    '''
    assert len(pred.shape) == 3 # this should be a 3D data frame

    # extract values within fluid region for prediction (SR) and ground truth (HR)
    sr_vals = get_fluid_region_points_frame(pred,binary_mask)
    hr_vals = get_fluid_region_points_frame(gt,binary_mask )

    # calculate linear regression parameters with scipy
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(hr_vals, sr_vals)
    return slope,  r_value**2


def plot_correlation(gt, prediction, bounds, frame_idx,color_b = KTH_colors['pink100'], save_as = None):
    '''
    Plot correlation plot between ground truth and prediction at a given frame
    '''
    # set percentage of how many random points are used
    p = 0.1
    mask_threshold = 0.6

    mask = np.asarray(gt['mask']).squeeze()

    # if mask static make dynamic mask 
    if len(mask.shape) == 3:
        mask = create_dynamic_mask(mask, prediction['u'].shape[0])
    
    # threshold mask
    mask[np.where(mask > mask_threshold)] = 1 

    # get indices of core and boundary
    idx_core = np.where((mask[frame_idx]-bounds[frame_idx]) == 1)
    idx_bounds = np.where(bounds[frame_idx] == 1)

    # get random indices for core and boundary to plot a subset of the points
    # core (subtract bounds from mask such that mask only contains core points)
    x_idx, y_idx, z_idx = random_indices3D((mask-bounds)[frame_idx], n=int(p*np.count_nonzero(mask[frame_idx])))
    # boundary 
    x_idx_b, y_idx_b, z_idx_b = random_indices3D(bounds[frame_idx], n=int(p*np.count_nonzero(bounds[frame_idx])))
    
    # Get velocity values in all directions
    # HR
    hr_u = np.asarray(gt['u'][frame_idx])
    hr_u_core = hr_u[x_idx, y_idx, z_idx]
    hr_u_bounds = hr_u[x_idx_b, y_idx_b, z_idx_b]
    hr_v = np.asarray(gt['v'][frame_idx])
    hr_v_core = hr_v[x_idx, y_idx, z_idx]
    hr_v_bounds = hr_v[x_idx_b, y_idx_b, z_idx_b]
    hr_w = np.asarray(gt['w'][frame_idx])
    hr_w_core = hr_w[x_idx, y_idx, z_idx]
    hr_w_bounds = hr_w[x_idx_b, y_idx_b, z_idx_b]

    # SR 
    sr_u = np.asarray(prediction['u'][frame_idx])
    sr_u_vals = sr_u[x_idx, y_idx, z_idx]
    sr_u_bounds = sr_u[x_idx_b, y_idx_b, z_idx_b]
    sr_v = np.asarray(prediction['v'][frame_idx])
    sr_v_vals = sr_v[x_idx, y_idx, z_idx]
    sr_v_bounds = sr_v[x_idx_b, y_idx_b, z_idx_b]
    sr_w = np.asarray(prediction['w'][frame_idx])
    sr_w_vals = sr_w[x_idx, y_idx, z_idx]
    sr_w_bounds = sr_w[x_idx_b, y_idx_b, z_idx_b]

    def plot_regression_points(hr_vals, sr_vals, hr_vals_bounds, sr_vals_bounds,all_hr, all_sr, all_hr_bounds, all_sr_bounds, direction = 'u'):
        dimension = 2 #TODO
        N = 100
        # make sure that the range is the same for all plots and make square range
        x_range = np.linspace(-abs_max, abs_max, N)
        
        corr_line, text = get_corr_line_and_r2(all_hr, all_sr, x_range)
        corr_line_bounds, text_bounds = get_corr_line_and_r2(all_hr_bounds, all_sr_bounds, x_range)

        # plot linear correlation line and parms
        plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
        plt.gca().text(0.05, 0.85, text_bounds,transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', color=color_b)
        plt.plot(x_range, x_range, color= 'grey', label = 'diagonal line')
        plt.plot(x_range, corr_line_bounds, '--', color = color_b)
        plt.plot(x_range, corr_line, 'k--')
        plt.scatter(hr_vals, sr_vals, s=0.3, c=["black"], label = 'core voxels')
        plt.scatter(hr_vals_bounds, sr_vals_bounds, s=0.3, c=[color_b], label = 'boundary voxels')
        
        plt.title(direction)
        plt.xlabel("V HR (m/s)")
        plt.ylabel("V prediction (m/s)")
        plt.legend(loc = 'lower right')
        plt.ylim(-abs_max, abs_max)
        plt.xlim(-abs_max, abs_max)

    def get_corr_line_and_r2(hr_vals, sr_vals, x_range):
        '''
        Returns correlation line and text for plot
        '''
        z = np.polyfit(hr_vals, sr_vals, 1)
        corr_line = np.poly1d(z)(x_range)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(hr_vals, sr_vals)
        text = f"$y={z[0]:0.4f}\;x{z[1]:+0.4f}$\n$R^2 = {r_value**2:0.4f}$"
        
        return corr_line, text

    
    print("Plotting correlation lines...")

    min_vals = np.min([np.min(sr_u_vals), np.min(sr_v_vals), np.min(sr_w_vals)])
    max_vals = np.max([np.max(sr_u_vals), np.max(sr_v_vals), np.max(sr_w_vals)])
    abs_max = np.max([np.abs(min_vals), np.abs(max_vals)])
    print('min/max/abs max', min_vals, max_vals, abs_max)

    plt.clf()

    # plot regression line for Vx, Vy and Vz
    plt.figure(figsize=(5, 5))
    plot_regression_points(hr_u_core, sr_u_vals, hr_u_bounds, sr_u_bounds,hr_u[idx_core], sr_u[idx_core], hr_u[idx_bounds], sr_u[idx_bounds],direction=r'$V_x$')
    if save_as is not None: plt.savefig(f"{save_as}_LRXplot.svg")
    
    plt.clf()
    plt.figure(figsize=(5, 5))
    plot_regression_points(hr_v_core, sr_v_vals, hr_v_bounds, sr_v_bounds,hr_v[idx_core], sr_v[idx_core], hr_v[idx_bounds], sr_v[idx_bounds],direction=r'$V_y$')
    if save_as is not None: plt.savefig(f"{save_as}_LRYplot.svg")

    plt.clf()
    plt.figure(figsize=(5, 5))
    plot_regression_points(hr_w_core, sr_w_vals, hr_w_bounds, sr_w_bounds,hr_w[idx_core], sr_w[idx_core], hr_w[idx_bounds], sr_w[idx_bounds], direction=r'$V_z$')
    plt.tight_layout()
    if save_as is not None: plt.savefig(f"{save_as}_LRZplot.svg")

    plt.clf()
    save_subplots = True

    # plot Vx, Vy and Vz in subplots
    if save_subplots: 
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plot_regression_points(hr_u_core, sr_u_vals, hr_u_bounds, sr_u_bounds,hr_u[idx_core], sr_u[idx_core], hr_u[idx_bounds], sr_u[idx_bounds],direction=r'$V_x$')
        plt.subplot(1, 3, 2)
        plot_regression_points(hr_v_core, sr_v_vals, hr_v_bounds, sr_v_bounds,hr_v[idx_core], sr_v[idx_core], hr_v[idx_bounds], sr_v[idx_bounds],direction=r'$V_y$')
        plt.subplot(1, 3, 3)
        plot_regression_points(hr_w_core, sr_w_vals, hr_w_bounds, sr_w_bounds,hr_w[idx_core], sr_w[idx_core], hr_w[idx_bounds], sr_w[idx_bounds], direction=r'$V_z$')
        plt.tight_layout()
        if save_as is not None: plt.savefig(f"{save_as}_LRXYZ_subplots.svg")
    

    



def get_2Dslice(data, frame, axis, slice_idx):
    '''
    Returns 2D slice from 4D data with given time frame, axis and index
    '''
    if len(data.squeeze().shape) == 3:
        frame = 0
        print("Only one frame available: take first frame.")
        if len(data.shape) == 3:
            data = np.expand_dims(data, 0)
        
    if axis == 0 :
        return data[frame, slice_idx, :, :]
    elif axis == 1:
        return data[frame, :, slice_idx, :]
    elif axis == 2:
        return data[frame, :, :, slice_idx]
    else: 
        print("Invalid axis! Axis must be 0, 1 or 2")

def get_indices(frames, axis, slice_idx):
    '''
    Returns indices for 4D data with given time frames, axis and index
    '''
    if axis == 0 :
        return np.index_exp[frames, slice_idx, :, :]
    elif axis == 1:
        return np.index_exp[frames, :, slice_idx, :]
    elif axis == 2:
        return np.index_exp[frames, :, :, slice_idx]
    else: 
        print("Invalid axis! Axis must be 0, 1 or 2")


def crop_center(img,cropx,cropy):
    '''
    Crop center of image given size of the new image
    '''
    # from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def get_boundaries(binary_mask):
    '''
    returns boundary and core mask given a binary mask. 
    Note that mask values should be 0 and 1
    '''

    if (len(binary_mask.shape)==3):
        print("Create boundary mask for 3D data")
        core_mask = binary_erosion(binary_mask)
        boundary_mask = binary_mask - core_mask
        return boundary_mask, core_mask
    
    core_mask       = np.zeros_like(binary_mask)
    boundary_mask   = np.zeros_like(binary_mask)

    for t in range(binary_mask.shape[0]):
        core_mask[t, :, :, :] = binary_erosion(binary_mask[t, :, :, :])
        boundary_mask[t, :, :, :] = binary_mask[t, :, :, :] - core_mask[t, :, :, :]

        
    assert(np.linalg.norm(binary_mask - (boundary_mask + core_mask))== 0 ) # check that there is no overlap between core and boundary mask
    return boundary_mask, core_mask


def plot_spatial_comparison(low_res, ground_truth, prediction, frame_idx = 9, axis=1, slice_idx = 50):

    if slice_idx% 2 != 0 : print("Slice index should be even!")

    patch = [40, 40]

    vel_colnames = ['u', 'v', 'w', 'div_x']#, 'divergence_y', 'divergence_z']
    vel_plotnames = ['Vx', r'Vy', r'Vz']
    n = 1

    #calculate divergence
    ground_truth['div_x'], ground_truth['div_y'], ground_truth['div_z'] = np.asarray(calculate_divergence(ground_truth['u'], ground_truth['v'], ground_truth['w']))
    low_res['div_x'], low_res['div_y'], low_res['div_z'] = np.asarray(calculate_divergence(low_res['u'], low_res['v'], low_res['w']))
    prediction['div_x'], prediction['div_y'], prediction['div_z'] = np.asarray(calculate_divergence(prediction['u'], prediction['v'], prediction['w']))


    for i, vel in enumerate(vel_colnames):
        slice_lr = get_2Dslice(low_res[vel], frame_idx, axis, slice_idx//2)
        slice_gt = get_2Dslice(ground_truth[vel], frame_idx, axis, slice_idx)
        slice_sr = get_2Dslice(prediction[vel], frame_idx, axis, slice_idx)

        slice_lr = crop_center(slice_lr, patch[0]//2, patch[1]//2)
        slice_gt = crop_center(slice_gt, patch[0], patch[1])
        slice_sr = crop_center(slice_sr, patch[0], patch[1])

        max_v = np.max(np.stack((np.resize(slice_lr, slice_gt.shape), slice_gt, slice_sr)))
        min_v = np.min(np.stack((np.resize(slice_lr, slice_gt.shape), slice_gt, slice_sr)))
        
        plt.subplot(len(vel_colnames), 4, n)
        plt.imshow(slice_lr, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 0: plt.title("LR")
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(vel)

        plt.subplot(len(vel_colnames), 4, n+1)
        plt.imshow(slice_gt, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 0: plt.title("HR")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(len(vel_colnames), 4, n+2)
        plt.imshow(slice_sr, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 0: plt.title("4DFlowNet")
        plt.xticks([])
        plt.yticks([])

        #TODO real linear interpolation
        plt.subplot(len(vel_colnames), 4, n+3)
        plt.imshow(slice_lr, vmin = min_v, vmax = max_v, cmap='jet', interpolation='bilinear')
        if i == 0: plt.title("bilinear")
        plt.xticks([])
        plt.yticks([])
        
        # plt.subplot(len(vel_colnames), 5, n+4)
        # plt.imshow(slice_lr, vmin = min_v, vmax = max_v, cmap='jet', interpolation='bicubic')
        # if i == 0: plt.title("bicubic")
        # plt.xticks([])
        # plt.yticks([])

        plt.colorbar()
        n+=4

    #fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("4DFlowNet/results/plots/Comparison_prediction.png")


def plot_comparison_temporal(low_res, ground_truth, prediction, frame_idx = 9, axis=1, slice_idx = 50, save_as = "visualize_interporalion_comparison.png"):
    #TODO check for downsampling rate and create frame idx from there for lowres

    if frame_idx% 2 != 0 : print("Slice index should be even!")

    patch = [40, 40]

    vel_colnames = ['u', 'v', 'w', 'div_x']#, 'divergence_y', 'divergence_z']
    vel_plotnames = ['Vx', r'Vy', r'Vz']
    n = 1

    #calculate divergence
    ground_truth['div_x'], ground_truth['div_y'], ground_truth['div_z'] = np.asarray(calculate_divergence(ground_truth['u'], ground_truth['v'], ground_truth['w']))
    low_res['div_x'], low_res['div_y'], low_res['div_z'] = np.asarray(calculate_divergence(low_res['u'], low_res['v'], low_res['w']))
    prediction['div_x'], prediction['div_y'], prediction['div_z'] = np.asarray(calculate_divergence(prediction['u'], prediction['v'], prediction['w']))


    for i, vel in enumerate(vel_colnames):
        #TODO change this with downsampling rate
        slice_lr = get_2Dslice(low_res[vel], frame_idx//2, axis, slice_idx)
        slice_gt = get_2Dslice(ground_truth[vel], frame_idx, axis, slice_idx)
        slice_sr = get_2Dslice(prediction[vel], frame_idx, axis, slice_idx)

        slice_lr = crop_center(slice_lr, patch[0], patch[1])
        slice_gt = crop_center(slice_gt, patch[0], patch[1])
        slice_sr = crop_center(slice_sr, patch[0], patch[1])

        max_v = np.max(np.stack((np.resize(slice_lr, slice_gt.shape), slice_gt, slice_sr)))
        min_v = np.min(np.stack((np.resize(slice_lr, slice_gt.shape), slice_gt, slice_sr)))
        
        plt.subplot(len(vel_colnames), 4, n)
        plt.imshow(slice_lr, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 0: plt.title("LR")
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(vel)

        plt.subplot(len(vel_colnames), 4, n+1)
        plt.imshow(slice_gt, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 0: plt.title("HR")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(len(vel_colnames), 4, n+2)
        plt.imshow(slice_sr, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 0: plt.title("4DFlowNet")
        plt.xticks([])
        plt.yticks([])

        #TODO real linear interpolation
        plt.subplot(len(vel_colnames), 4, n+3)
        plt.imshow(slice_lr, vmin = min_v, vmax = max_v, cmap='jet', interpolation='bilinear')
        if i == 0: plt.title("bilinear")
        plt.xticks([])
        plt.yticks([])
        
        # plt.subplot(len(vel_colnames), 5, n+4)
        # plt.imshow(slice_lr, vmin = min_v, vmax = max_v, cmap='jet', interpolation='bicubic')
        # if i == 0: plt.title("bicubic")
        # plt.xticks([])
        # plt.yticks([])

        plt.colorbar()
        n+=4

    #fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_as)

def show_temporal_development_line(gt, lr, pred, mask, axis, indices, save_as = "Temporal_development.png"):
    mask[np.where(mask !=0)] = 1
    gt = np.multiply(gt, mask)
    lr = np.multiply(lr, mask)
    pred = np.multiply(pred, mask)

    def get_line(data):
        #returns line in 4D data over all time steps
        x,y = indices
        if axis == 1:
            return data[:, :, x, y]
        elif axis ==2:
            return data[:, x, :, y]
        elif axis ==3:
            return data[:, x,  y, :]
        else:
            print("Invalid axis: Please choose axis 1, 2, 3")

    prediction = get_line(pred).transpose()
    ground_truth = get_line(gt).transpose()
    low_resolution= get_line(lr).transpose()
    print('prediction shape', prediction.shape)

    min_v = np.min([np.min(prediction), np.min(ground_truth), np.min(low_resolution)])
    max_v = np.max([np.max(prediction), np.max(ground_truth), np.max(low_resolution)])


    plt.subplot(1, 3, 1)
    plt.imshow(low_resolution, vmin = min_v, vmax = max_v, cmap='jet')
    plt.title("LR")
    plt.xlabel('t')
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, vmin = min_v, vmax = max_v, cmap='jet')
    plt.title("GT")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, vmin = min_v, vmax = max_v, cmap='jet')
    plt.title("SR")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(save_as,bbox_inches='tight')

def show_quiver( u, v, w, mask,frame,save_as = "3DFlow.png"):
    x_len, y_len, z_len = u.shape
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(y_len),np.arange(x_len),np.arange(z_len))
    print("x shape:", x.shape, y.shape, "u: ", u.shape)
    
    set_to_zero = 0.9
    if len(binary_mask.shape) ==3:
        mask = create_dynamic_mask(binary_mask, pred.shape[0])
    

    x_idx, y_idx, z_idx = random_indices3D(mask[frame], int(np.count_nonzero(mask[frame])*set_to_zero))
    u[x_idx, y_idx, z_idx] = 0
    v[x_idx, y_idx, z_idx] = 0
    w[x_idx, y_idx, z_idx] = 0
    
    cropx = cropy = cropz = 10
    startx = x_len//2-(cropx//2)
    starty = y_len//2-(cropy//2)    
    startz = z_len//2-(cropz//2)
    u = u[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
    v = v[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
    w = w[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 

    x =x[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
    y =y[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
    z =z[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 

    ax.quiver(x, y, z, u, v, w, length=0.3, normalize=True, color=plt.cm.viridis([200, 50, 100, 200, 200, 50, 50, 100, 100]))
    fig.savefig(save_as)
    plt.clf()

def make_3D_quiver_plot(data,mask,  frame, set_to_zero=0.9):
        
        u_quiver = data['u'][frame].copy() 
        v_quiver = data['v'][frame].copy() 
        w_quiver = data['w'][frame].copy() 

        x_len, y_len, z_len = data['u'].shape[1::]

        # ev = np.array([1, 0, 0])
        # angles = np.arccos(np.dot(ev, [u, v, w]) / (np.linalg.norm(ev) * np.linalg.norm(v2)))
        # angles = np.multiply

        # Make the grid
        x, y, z = np.meshgrid(np.arange(y_len),np.arange(x_len),np.arange(z_len))
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Erase (i.e. set zo zero) multiple points to increase visibility
        x_idx, y_idx, z_idx = random_indices3D(mask[frame], int(np.count_nonzero(mask[frame])*set_to_zero))
        u_quiver[x_idx, y_idx, z_idx] = 0
        v_quiver[x_idx, y_idx, z_idx] = 0
        w_quiver[x_idx, y_idx, z_idx] = 0

        # crop image from middle
        cropx = cropy = cropz = 25
        startx = x_len//2-(cropx//2)
        starty = y_len//2-(cropy//2)    
        startz = z_len//2-(cropz//2)
        u_quiver = u_quiver[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
        v_quiver = v_quiver[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
        w_quiver = w_quiver[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 

        x =x[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
        y =y[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
        z =z[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz]

        u_new = u_quiver[np.where(u_quiver != 0)]
        v_new = v_quiver[np.where(u_quiver != 0)]
        w_new = w_quiver[np.where(u_quiver != 0)]

        x_new = x[np.where(u_quiver!=0)] 
        y_new = y[np.where(u_quiver!=0)] 
        z_new = z[np.where(u_quiver!=0)]


        # Color by magnitude
        c = np.sqrt(u_new**2+ v_new**2+ w_new**2) #np.arctan2(w_new, u_new)
        # Flatten and normalize

        c = (c.ravel() - c.min()) / c.ptp()
        # Repeat for each body line and two head lines
        c = np.concatenate((c, np.repeat(c, 2)))
        # Colormap
        c = plt.cm.jet(c)


        ax.quiver(x_new, y_new, z_new, u_new, v_new, w_new, length=10, normalize=False,  pivot='middle', color = c)
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.zlabel('z')
        # plt.show()


def show_timeframes(gt,lr,  pred,mask, rel_error, comparison_lst, comparison_name, timepoints, axis, idx,min_v, max_v,save_as = "Frame_comparison.png"):
    '''
    Plots a series of frames next to eachother to compare 
    '''
    plt.clf()
    T = len(timepoints)
    N = 3 + len(comparison_lst)
    i = 1
    for j,t in enumerate(timepoints):
        
        gt_slice = get_2Dslice(gt, t,  axis=axis, slice_idx=idx )
        pred_slice = get_2Dslice(pred, t, axis=axis, slice_idx=idx )

        lr_slice = np.zeros_like(gt_slice)
        if t%2 == 0: lr_slice = get_2Dslice(lr, t//2, axis=axis, slice_idx=idx )
        
        # min_v = np.min([np.min(pred_slice ), np.min(gt_slice), np.min(lr_slice)])
        # max_v = np.max([np.max(pred_slice), np.max(gt_slice), np.max(lr_slice)])  

        plt.subplot(T, N, i)

        if t%2 == 0:
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i == 1: plt.title("LR")
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('frame = '+ str(t))
            
        else:
            #plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='jet', aspect='auto')
            plt.axis('off')
        

        i +=1
        plt.subplot(T, N, i)
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
        if i == 2: plt.title("GT")
        plt.xticks([])
        plt.yticks([])

        i +=1
        plt.subplot(T, N, i)
        plt.imshow(pred_slice, vmin = min_v, vmax = max_v, cmap='viridis',aspect='auto')
        if i == 3: plt.title("4DFlowNet")
        plt.xticks([])
        plt.yticks([])
        for comp, name in zip(comparison_lst, comparison_name):
            i +=1
            plt.subplot(T, N, i)
            plt.imshow(get_2Dslice(comp,t, axis=axis, slice_idx=idx), vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i-1 == (i-1)%N: plt.title(name)
            plt.xticks([])
            plt.yticks([])

        i +=1

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    #plt.tight_layout()
    plt.savefig(save_as,bbox_inches='tight')
    
    if False:
        plt.clf()
        mask[np.where(mask !=0)] = 1
        gt = np.multiply(gt, mask)
        lr = np.multiply(lr, mask)
        pred = np.multiply(pred, mask)

        N = 4
        i = 1
        for j,t in enumerate(timepoints):
            
            gt_slice = get_2Dslice(gt, t,  axis=axis, slice_idx=idx )
            pred_slice = get_2Dslice(pred, t, axis=axis, slice_idx=idx )
            err_slice = get_2Dslice(rel_error, t, axis=axis, slice_idx=idx )
            #dt_slice = get_slice(dt, t, axis=axis, slice_idx=idx )
            #print("shape dt:", dt.shape, dt_slice.shape, gt_slice.shape )

            lr_slice = np.zeros_like(gt_slice)
            if t%2 == 0: lr_slice = get_2Dslice(lr, t//2, axis= axis, slice_idx= idx )

            #min_v = np.min([np.min(pred_slice ), np.min(gt_slice), np.min(lr_slice)])
            #max_v = np.max([np.max(pred_slice), np.max(gt_slice), np.max(lr_slice)])  

            plt.subplot(T, N, i)
            if t%2 == 0:
                plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis')
                if i == 1: plt.title("LR")
                plt.xticks([])
                plt.yticks([])
                plt.ylabel('frame = '+ str(t))
            else:
                plt.axis('off')

            i += 1
            plt.subplot(T, N, i)
            plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis')
            if i == 2: plt.title("GT")
            plt.xticks([])
            plt.yticks([])

            i += 1
            plt.subplot(T, N, i)
            plt.imshow(pred_slice, vmin = min_v, vmax = max_v, cmap='viridis')
            if i == 3: plt.title("SR")
            plt.xticks([])
            plt.yticks([])

            i += 1
            plt.subplot(T, N, i)
            plt.imshow(err_slice, cmap='jet')
            if i == 4: plt.title("Relative error")
            plt.xticks([])
            plt.yticks([])

            # i +=1
            # plt.subplot(T, 5, i)
            # plt.imshow(dt_slice, cmap='jet')
            # if i == 5: plt.title("|dt|")
            # plt.xticks([])
            # plt.yticks([])

            # plt.colorbar()
            

            
            i +=1
            

        save_under = save_as[:-4]+ "_fluidregion.png"
        print("save with only fluid region visible", save_under)
        #plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(save_under,bbox_inches='tight')
        #plt.clf()


def calculate_temporal_derivative(data, timestep=1):
    '''
    Calculate difference between two time frames and each voxel
    i.e. for u: dt u(t) = |u(t+1) - u(t)| / timestep
    '''

    n_frames = data.shape[0]
    dt =  np.zeros_like(data)
    for t in range(n_frames-1):
        dt[t, :, :, :] = (data[t+timestep, :, :, :] - data[t, :, :, :])/timestep

    dt = np.abs(dt) 
    
    return dt


def plot_relative_error(lst_hgt_paths, lst_hpred_paths,lst_names, save_as = 'Relative_error_comparison.png'):
    '''
    Plots relative error from all the files given in the list of paths in the same plot
    '''
    assert(len(lst_hgt_paths)==len(lst_hpred_paths))
    vel_colnames=['u', 'v', 'w']

    for gt_path, pred_path, name in zip(lst_hgt_paths, lst_hpred_paths, lst_names):
        gt = {}
        pred = {}
        with h5py.File(pred_path, mode = 'r' ) as h_pred:
            with h5py.File(gt_path, mode = 'r' ) as h_gt:

                # load gt and predcition values
                for vel in vel_colnames:
                    
                    gt[vel] = np.asarray(h_gt[vel])
                    pred[vel] = np.asarray(h_pred[vel])

                    #transpose for temporal resolution
                    #TODO change if needed
                    pred[vel] = pred[vel].transpose(1, 0, 2, 3)
                    # gt[vel] = crop_gt(gt[vel], pred[vel].shape 

                    #load prediction values
                gt["mask"] = np.asarray(h_gt["mask"])
                #compute relative error

                error_gt = calculate_relative_error_normalized(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
                #Plot Relative error
                plt.plot(error_gt, '-', label = name)


    plt.plot(50*np.ones(len(error_gt)), 'k:')
    plt.xlabel("Frame")
    plt.ylabel("Relative error (%)")
    plt.ylim((0, 100))
    #plt.legend(lst_names)
    plt.title("Relative error")
    #plt.savefig(save_as)
    #plt.clf()




def create_temporal_comparison_gif(lr, hr, pred, vel, save_as):

    v_lr = lr[vel]
    v_hr = hr[vel]
    v_pred = pred[vel]

    v_NN = temporal_NN_interpolation(v_lr,v_hr.shape )

    combined_image = np.concatenate((v_NN, v_hr, v_pred), axis = 3)
    print(combined_image.shape)
    idx = 30

    generate_gif_volume(combined_image[:,idx, :, : ], axis = 0, save_as = save_as)


def plot_slices_over_time(gt_cube,lr_cube,  mask_cube, rel_error_cube, comparison_lst, comparison_name, timepoints, axis, idx,min_v, max_v,exclude_rel_error = True, save_as = "Frame_comparison.png", figsize = (30,20)):
    def row_based_idx(num_rows, num_cols, idx):
        return np.arange(1, num_rows*num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx-1]

    
    T = 3 + len(comparison_lst)  #len(timepoints)
    N = len(timepoints)  #4 + len(comparison_lst)
    print(T, N)
    if exclude_rel_error: T -=1
    print(T, N)

    # fig = plt.figure(figsize=(10,10))
    fig, axes = plt.subplots(nrows=T, ncols=N, constrained_layout=True, figsize=figsize)

    i = 1
    idxs = get_indices(timepoints, axis, idx)
    gt_cube = gt_cube[idxs]
    mask_cube = mask_cube[idxs]
    
    # pred_cube = pred_cube[idxs]
    #lr = lr[idxs]

    min_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.01)
    max_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.99)
    if not exclude_rel_error:
        rel_error_slices =[get_2Dslice(rel_error_cube, t, axis, idx) for t in timepoints]
        min_rel_error = np.min(np.array(rel_error_slices))
        max_rel_error = np.max(np.array(rel_error_slices))
    for j,t in enumerate(timepoints):
        
        gt_slice = gt_cube[j]
        # pred_slice = pred_cube[j]

        lr_slice = np.zeros_like(gt_slice)
        if t%2 == 0: lr_slice = get_2Dslice(lr_cube, t//2, axis=axis, slice_idx=idx )
        plt.subplot(T, N, row_based_idx(T, N, i))

        if t%2 == 0:
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i == 1: plt.ylabel("LR")
            plt.xticks([])
            plt.yticks([])
            
        plt.title('frame '+ str(t))
        plt.xticks([])
        plt.yticks([])
        # plt.axis('off')
        

        i +=1
        plt.subplot(T, N, row_based_idx(T, N, i))
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
        if i == 2: plt.ylabel("HR")
        plt.xticks([])
        plt.yticks([])

        # i +=1
        # plt.subplot(T, N, row_based_idx(T, N, i))
        # plt.imshow(pred_slice, vmin = min_v, vmax = max_v, cmap='viridis',aspect='auto')
        # if i == 3: plt.ylabel("4DFlowNet")
        # plt.xticks([])
        # plt.yticks([])


        for comp, name in zip(comparison_lst, comparison_name):
            i +=1
            plt.subplot(T, N, row_based_idx(T, N, i))
            im = plt.imshow(get_2Dslice(comp,t, axis=axis, slice_idx=idx), vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel(name)
            plt.xticks([])
            plt.yticks([])

        if not exclude_rel_error:
            i +=1
            plt.subplot(T, N, row_based_idx(T, N, i))
            re_img = plt.imshow(get_2Dslice(rel_error_cube, t, axis, idx),vmin=min_rel_error, vmax=max_rel_error, cmap='viridis',aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel("abs. error")
            plt.xticks([])
            plt.yticks([])
            if t == timepoints[-1]:
                plt.colorbar(re_img, ax = axes[-1], aspect = 10, label = 'abs. error ')

        
        i +=1

    fig.colorbar(im, ax=axes.ravel().tolist(), aspect = 50, label = 'velocity (m/s)')
    plt.savefig(save_as,bbox_inches='tight' )
    # plt.tight_layout()

def plot_slices_over_time1(gt_cube,lr_cube,  mask_cube, rel_error_cube, comparison_lst, comparison_name, timepoints, axis, idx,min_v, max_v,exclude_rel_error = True, save_as = "Frame_comparison.png", figsize = (30,20)):
    def row_based_idx(num_rows, num_cols, idx):
        return np.arange(1, num_rows*num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx-1]

    
    T = 3 + len(comparison_lst)#len(timepoints)
    N = len(timepoints)#4 + len(comparison_lst)
    print(T, N)
    if exclude_rel_error: T -=1
    print(T, N)

    # fig = plt.figure(figsize=(10,10))
    fig, axes = plt.subplots(nrows=T, ncols=N, constrained_layout=True, figsize=figsize)

    i = 1
    idxs = get_indices(timepoints, axis, idx)
    gt_cube = gt_cube[idxs]
    mask_cube = mask_cube[idxs]
    
    # pred_cube = pred_cube[idxs]
    #lr = lr[idxs]

    min_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.01)
    max_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.99)
    if not exclude_rel_error:
        rel_error_slices =[get_2Dslice(rel_error_cube, t, axis, idx) for t in timepoints]
        min_rel_error = np.min(np.array(rel_error_slices))
        max_rel_error = np.max(np.array(rel_error_slices))
    for j,t in enumerate(timepoints):
        
        gt_slice = gt_cube[j]
        # pred_slice = pred_cube[j]

        lr_slice = np.zeros_like(gt_slice)
        if t%2 == 0: lr_slice = get_2Dslice(lr_cube, t//2, axis=axis, slice_idx=idx )
        plt.subplot(T, N, row_based_idx(T, N, i))

        if t%2 == 0:
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i == 1: plt.ylabel("LR")
            plt.xticks([])
            plt.yticks([])
            
        plt.title('frame '+ str(t))
        plt.xticks([])
        plt.yticks([])
        # plt.axis('off')
        

        i +=1
        plt.subplot(T, N, row_based_idx(T, N, i))
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
        if i == 2: plt.ylabel("HR")
        plt.xticks([])
        plt.yticks([])

        # i +=1
        # plt.subplot(T, N, row_based_idx(T, N, i))
        # plt.imshow(pred_slice, vmin = min_v, vmax = max_v, cmap='viridis',aspect='auto')
        # if i == 3: plt.ylabel("4DFlowNet")
        # plt.xticks([])
        # plt.yticks([])


        for comp, name in zip(comparison_lst, comparison_name):
            i +=1
            plt.subplot(T, N, row_based_idx(T, N, i))
            im = plt.imshow(get_2Dslice(comp,t, axis=axis, slice_idx=idx), vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel(name)
            plt.xticks([])
            plt.yticks([])

        if not exclude_rel_error:
            i +=1
            plt.subplot(T, N, row_based_idx(T, N, i))
            re_img = plt.imshow(get_2Dslice(rel_error_cube, t, axis, idx),vmin=min_rel_error, vmax=max_rel_error, cmap='viridis',aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel("abs. error")
            plt.xticks([])
            plt.yticks([])
            if t == timepoints[-1]:
                plt.colorbar(re_img, ax = axes[-1], aspect = 10, label = 'abs. error ')

        
        i +=1
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.colorbar(im, ax=axes.ravel().tolist(), aspect = 50, label = 'velocity (m/s)')
    plt.savefig(save_as,bbox_inches='tight' )

    
# def plot_k_r2_vals(frames, k,k_bounds, r2,  r2_bounds, peak_flow_frame, name_evaluation, eval_dir):
#     '''Plots the k and the r^2 values for each velocity component over time in one and separate plots '''

#     vel_plotname = [r'$V_x$', r'$V_y$', r'$V_z$']
#     vel_colnames = ['u', 'v', 'w']
#     min_val = np.minimum(0.05, np.minimum(np.min(k_bounds), np.min(r2_bounds)))
#     max_val = np.maximum(np.max(k), np.max(r2))
#     plt.figure(figsize=(15, 5))
    
#     # make subplots
#     # make plot for regression slope 
#     for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
#         plt.subplot(2, 3, i+1)
#         plt.plot(range(frames), k[i*frames:i*frames+frames] , label = 'k core', color = 'black')
#         plt.plot(range(frames), k_bounds[i*frames:i*frames+frames] ,'--',  label = 'k boundary', color = KTH_colors['pink100'])
#         plt.plot(np.ones(frames), 'k:')
#         plt.ylim([min_val, np.maximum(max_val, 1.05)])
#         plt.title(title)
#         plt.xlabel('frames')
#         plt.ylabel('k')
#         plt.scatter(np.ones(2)*peak_flow_frame, [k[i*frames+peak_flow_frame],k_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KTH_colors['grey80'])
#         plt.legend()
#         print(f'Average k vals core {np.average(k[i*frames:i*frames+frames])}')
#         print(f'Average k vals boundary {np.average(k_bounds[i*frames:i*frames+frames])}')
#         print(f'Min k vals core {np.min(k[i*frames:i*frames+frames])}')
#         print(f'Min k vals boundary {np.min(k_bounds[i*frames:i*frames+frames])}')
#         print(f'Max k vals core {np.max(k[i*frames:i*frames+frames])}')
#         print(f'Max k vals boundary {np.max(k_bounds[i*frames:i*frames+frames])}')

#     # make plot for r^2
#     for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
#         plt.subplot(2, 3, i+4)
#         plt.plot(range(frames), r2[i*frames:i*frames+frames] ,label = r'$R^2$ core', color = 'black')
#         plt.plot(range(frames), r2_bounds[i*frames:i*frames+frames] ,'--', label = r'$R^2$ boundary', color = KTH_colors['pink100'])
#         plt.plot(np.ones(frames), 'k:')
#         plt.ylim([min_val, np.maximum(max_val, 1.05)])
#         plt.title(title)
#         plt.xlabel('frame')
#         plt.ylabel(r'$R^2$')
#         plt.scatter(np.ones(2)*peak_flow_frame, [r2[i*frames+peak_flow_frame], r2_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KTH_colors['blue80'])
#         plt.legend()

#     plt.tight_layout()
#     plt.savefig(f'{eval_dir}/{name_evaluation}_k_vals.svg')

#     # save each plot separately
#     plt.figure(figsize=(5, 5))
#     for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
#         plt.clf()
#         plt.plot(range(frames), k[i*frames:i*frames+frames] , label = 'k core', color = 'black')
#         plt.plot(range(frames), k_bounds[i*frames:i*frames+frames] ,'--',  label = 'k boundary', color = KTH_colors['pink100'])
#         plt.plot(np.ones(frames), 'k:')
#         plt.ylim([min_val, np.maximum(max_val, 1.05)])
#         plt.title(title)
#         plt.xlabel('frame')
#         plt.ylabel('k')
#         plt.scatter(np.ones(2)*peak_flow_frame, [k[i*frames+peak_flow_frame],k_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KTH_colors['blue80'])
#         plt.legend()
#         plt.savefig(f'{eval_dir}/{name_evaluation}_k_vals_{vel}_.svg', bbox_inches='tight')
#     for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
#         plt.clf()
#         plt.plot(range(frames), r2[i*frames:i*frames+frames] ,label = r'$R^2$ core', color = 'black')
#         plt.plot(range(frames), r2_bounds[i*frames:i*frames+frames] ,'--', label = r'$R^2$ boundary', color = KTH_colors['pink100'])
#         plt.plot(np.ones(frames), 'k:')
#         plt.ylim([min_val, np.maximum(max_val, 1.05)])
#         plt.title(title)
#         plt.xlabel('frame')
#         plt.ylabel(r'$R^2$')
#         plt.scatter(np.ones(2)*peak_flow_frame, [r2[i*frames+peak_flow_frame], r2_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KTH_colors['blue80'])
#         plt.legend()
#         plt.savefig(f'{eval_dir}/{name_evaluation}_R2_vals_{vel}__.svg', bbox_inches='tight')


def plot_k_r2_vals(gt, pred, bounds, peak_flow_frame,color_b = KTH_colors['pink100'] , save_as= ''):

    vel_plotname = [r'$V_x$', r'$V_y$', r'$V_z$']


    print('Peak flow frame:', peak_flow_frame)
    frames = gt['mean_speed'].shape[0]
    k, r2, k_bounds,r2_bounds  = np.zeros(3*frames), np.zeros(3*frames), np.zeros(3*frames), np.zeros(3*frames)
    bounds_mask = bounds.copy()
    core_mask = gt['mask'] - bounds_mask

    plt.figure(figsize=(8, 8))
    #calculate k values in core and boundary region
    for i, vel in enumerate(vel_colnames):
        for t in range(frames):
            k[t+i*frames], r2[t+i*frames]  = calculate_k_R2( pred[vel][t], gt[vel][t], core_mask[t])
            k_bounds[t+i*frames], r2_bounds[t+i*frames]  = calculate_k_R2( pred[vel][t], gt[vel][t], bounds[t])

    min_val = np.minimum(0.05, np.minimum(np.min(k_bounds), np.min(r2_bounds)))
    max_val = np.maximum(np.max(k), np.max(r2))
    #make subplots
    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
        plt.subplot(2, 3, i+1)
        plt.plot(range(frames), k[i*frames:i*frames+frames] , label = 'k core', color = 'black')
        plt.plot(range(frames), k_bounds[i*frames:i*frames+frames] ,'--',  label = 'k boundary', color = color_b)
        plt.plot(np.ones(frames), 'k:')
        plt.ylim([min_val, np.maximum(max_val, 1.05)])
        plt.title(title)
        plt.xlabel('frames')
        plt.ylabel('k')
        plt.scatter(np.ones(2)*peak_flow_frame, [k[i*frames+peak_flow_frame],k_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KTH_colors['grey80'])
        plt.legend()
        print(f'Average k vals core {np.average(k[i*frames:i*frames+frames])}')
        print(f'Average k vals boundary {np.average(k_bounds[i*frames:i*frames+frames])}')
        print(f'Min k vals core {np.min(k[i*frames:i*frames+frames])}')
        print(f'Min k vals boundary {np.min(k_bounds[i*frames:i*frames+frames])}')
        print(f'Max k vals core {np.max(k[i*frames:i*frames+frames])}')
        print(f'Max k vals boundary {np.max(k_bounds[i*frames:i*frames+frames])}')
    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
        plt.subplot(2, 3, i+4)
        print(f'Average R2 vals core {np.average(r2[i*frames:i*frames+frames])}')
        print(f'Average R2 vals boundary {np.average(r2_bounds[i*frames:i*frames+frames])}')
        plt.plot(range(frames), r2[i*frames:i*frames+frames] ,label = r'$R^2$ core', color = 'black')
        plt.plot(range(frames), r2_bounds[i*frames:i*frames+frames] ,'--', label = r'$R^2$ boundary', color = color_b)
        plt.plot(np.ones(frames), 'k:')
        plt.ylim([min_val, np.maximum(max_val, 1.05)])
        plt.title(title)
        plt.xlabel('frames')
        plt.ylabel(r'$R^2$')
        plt.scatter(np.ones(2)*peak_flow_frame, [r2[i*frames+peak_flow_frame], r2_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KTH_colors['grey80'])
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{eval_dir}/k_vals_{set_name}_{save_as}.svg')

    #save each plot separately
    plt.figure(figsize=(5, 5))
    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
        plt.clf()
        plt.plot(range(frames), k[i*frames:i*frames+frames] , label = 'k core', color = 'black')
        plt.plot(range(frames), k_bounds[i*frames:i*frames+frames] ,'--',  label = 'k boundary', color = color_b)
        plt.plot(np.ones(frames), 'k:')
        plt.ylim([min_val, np.maximum(max_val, 1.05)])
        plt.title(title)
        plt.xlabel('frames')
        plt.ylabel('k')
        plt.scatter(np.ones(2)*peak_flow_frame, [k[i*frames+peak_flow_frame],k_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KTH_colors['grey80'])
        plt.legend()
        plt.savefig(f'{eval_dir}/k_vals_{vel}_{set_name}_{save_as}.svg', bbox_inches='tight')
    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
        plt.clf()
        plt.plot(range(frames), r2[i*frames:i*frames+frames] ,label = r'$R^2$ core', color = 'black')
        plt.plot(range(frames), r2_bounds[i*frames:i*frames+frames] ,'--', label = r'$R^2$ boundary', color = color_b)
        plt.plot(np.ones(frames), 'k:')
        plt.ylim([min_val, np.maximum(max_val, 1.05)])
        plt.title(title)
        plt.xlabel('frames')
        plt.ylabel(r'$R^2$')
        plt.scatter(np.ones(2)*peak_flow_frame, [r2[i*frames+peak_flow_frame], r2_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KTH_colors['grey80'])
        plt.legend()
        plt.savefig(f'{eval_dir}/R2_vals_{vel}_M{model_name}_{set_name}_{save_as}_.svg', bbox_inches='tight')



# ------------------- INTERPOLATION FUNCTIONS---------------------------


def temporal_linear_interpolation(lr, hr_shape):
    '''
    Linear interpolation in time, from (t, h, w, d) to (2t, h, w, d)
    Be aware that if the hr shape is twice as high the last frame will be set to zero, since it it not in-between slices.
    Using a equidistant grid, leading to taking the average of two consequtive frames
    '''

    # only temporal resolution increases 
    t_lr = np.arange(0, lr.shape[0])
    x_lr = np.arange(0, lr.shape[1])
    y_lr = np.arange(0, lr.shape[2])
    z_lr = np.arange(0, lr.shape[3])

    t_hr = np.linspace(0, lr.shape[0]-0.5,  hr_shape[0])
    
    tg, xg, yg ,zg = np.meshgrid(t_hr, x_lr, y_lr, z_lr, indexing='ij', sparse=True)

    interp = RegularGridInterpolator((t_lr, x_lr, y_lr, z_lr), lr, method='linear', bounds_error=False, fill_value=0)
    interpolate = interp((tg, xg, yg ,zg))


    return interpolate

def temporal_linear_interpolation_np(lr, hr_shape, offset = 0):
    '''
    Linear interpolation in time, from (t, h, w, d) to (2t, h, w, d)
    Be aware that if the hr shape is twice as high the last frame will be set to zero, since it it not in-between slices.
    Using a equidistant grid, leading to taking the average of two consequtive frames
    '''
    T, x, y, z = hr_shape
    interpolate = np.zeros((T, x, y, z))

    interpolate[offset::2, :, :, :] = lr
    for t in range(0, T-2, 2):
        interpolate[1+t, :, :, :] = (interpolate[t, :, :, :] + interpolate[1+t+1, :, :, :]) /2

    return interpolate

def temporal_NN_interpolation(lr, hr_shape):
    '''
    Nearest neighbor interpolation in time, from (t, h, w, d) to (2t, h, w, d)
    For an equidistant grid this is only the doubling of the temporal resolution
    '''
    # t_lr = np.arange(0, lr.shape[0])
    # x_lr = np.arange(0, lr.shape[1])
    # y_lr = np.arange(0, lr.shape[2])
    # z_lr = np.arange(0, lr.shape[3])

    # t_hr = np.linspace(0, lr.shape[0]-0.5,  hr_shape[0])
    
    # tg, xg, yg ,zg = np.meshgrid(t_hr, x_lr, y_lr, z_lr, indexing='ij', sparse=True)

    # interp = RegularGridInterpolator((t_lr, x_lr, y_lr, z_lr), lr, method='nearest', bounds_error=False, fill_value=0)
    # interpolate = interp((tg, xg, yg ,zg))

    

    interpolate = np.zeros(hr_shape)
    interpolate[::2, :, :, :] = lr
    interpolate[1::2, :, :, :] = lr

    return interpolate

def temporal_cubic_interpolation(lr, hr_shape):
    '''
    Cubic interpolation in time , from (t, h, w, d) to (2t, h, w, d)
    '''
    # x_lr = np.arange(0, lr.shape[0])
    # x_hr = np.linspace(0, lr.shape[0]-0.5,  hr_shape[0])
    # cs = CubicSpline(x_lr, lr, axis=0)

    # interpolate = cs(x_hr)
    t_lr = np.arange(0, lr.shape[0])
    x_lr = np.arange(0, lr.shape[1])
    y_lr = np.arange(0, lr.shape[2])
    z_lr = np.arange(0, lr.shape[3])

    t_hr = np.linspace(0, lr.shape[0]-0.5,  hr_shape[0])
    
    tg, xg, yg ,zg = np.meshgrid(t_hr, x_lr, y_lr, z_lr, indexing='ij', sparse=True)

    interp = RegularGridInterpolator((t_lr, x_lr, y_lr, z_lr), lr, method='cubic', bounds_error=False, fill_value=0)
    interpolate = interp((tg, xg, yg ,zg))

    return interpolate


def temporal_sinc_interpolation(lr, hr_shape):
    '''
    TODO create sinc interpolation
    '''
    a = 1
    return None


def temporal_linear_interpolation_np(lr, hr_shape):
    T, x, y, z = hr_shape
    interpolate = np.zeros((hr_shape))
    interpolate[::2, :, :, :] = lr
    for t in range(0, T-2, 2):
        interpolate[1+t, :, :, :] = (interpolate[t, :, :, :] + interpolate[1+t+1, :, :, :]) /2

    interpolate[-1, :, :, :] = interpolate[-2, :, :, :] 

    return interpolate

def spatial3D_NN_interpolation(lr, hr_shape, method = 'nearest'):

    x_lr = np.arange(0, lr.shape[0])
    y_lr = np.arange(0, lr.shape[1])
    z_lr = np.arange(0, lr.shape[2])

    x_hr = np.linspace(0, lr.shape[0],  int(hr_shape[0]))
    y_hr = np.linspace(0, lr.shape[1],  int(hr_shape[1]))
    z_hr = np.linspace(0, lr.shape[2],  int(hr_shape[2]))
    
    xg, yg ,zg = np.meshgrid(x_hr, y_hr, z_hr, indexing='ij', sparse=True)

    interp = RegularGridInterpolator((x_lr, y_lr, z_lr), lr, method=method, bounds_error=False, fill_value=0)
    interpolate = interp((xg, yg ,zg))
    return interpolate