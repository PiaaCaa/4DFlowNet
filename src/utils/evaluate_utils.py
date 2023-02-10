import tensorflow as tf
import numpy as np
import time
import os
from Network.SR4DFlowNet import SR4DFlowNet
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset import ImageDataset
import h5py
from prepare_data.visualize_utils import generate_gif_volume
from Network.loss_utils import calculate_divergence
from scipy.signal import convolve2d
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt

# Crop mask to match desired shape * downsample
def crop_gt(gt, desired_shape):
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


def calculate_relative_error_np(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
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


def calculate_pointwise_error(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
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
    #relative_speed_loss = np.clip(relative_speed_loss, 0., 1.)

    idx_mask = np.where(binary_mask == 0)
    relative_speed_loss[:,idx_mask[0], idx_mask[1], idx_mask[2]] = 0

    return relative_speed_loss, np.sqrt(u_diff), np.sqrt(v_diff), np.sqrt(w_diff)#, np.sum(diff_speed, axis = 0) #corrected_speed_loss#mean_err


def calculate_mean_speed(u_hi, v_hi, w_hi, binary_mask):

    speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi))
    mean_speed = np.sum(speed, axis=(1,2,3)) / (np.sum(binary_mask, axis=(0,1,2)) + 1) *100
    return mean_speed

def compare_masks(u_hi, v_hi, w_hi, binary_mask):
    overlap_mask= np.zeros_like(u_hi)
    overlap_mask[np.where(u_hi != 0)] = 1
    overlap_mask[np.where(v_hi != 0)] = 1
    overlap_mask[np.where(w_hi != 0)] = 1

    mask = overlap_mask.copy()
    extended_mask =  np.zeros_like(u_hi)
    for i in range(extended_mask.shape[0]):
        extended_mask[i, :, :, :] = binary_mask

    overlap_mask[np.where((extended_mask == 0) & (overlap_mask == 1))] = 2
    overlap_mask[np.where((extended_mask == 1) & (overlap_mask == 0))] = 3
    
    return overlap_mask, mask[0].squeeze()


def plot_regression(gt, prediction, frame_idx, save_as):
    """ Plot a linear regression between HR and predicted SR in given frame """
    #
    # Parameters
    #
    mask_threshold = 0.8

    mask = np.asarray(gt['mask']).squeeze()  #assume static mask
    bounds = np.zeros_like(mask)

    for i in range(mask.shape[0]):
        bounds[i, :, :] = get_boundaries(mask[i, :, :])
    
    mask[np.where(mask > mask_threshold)] = 1 

    idx_inner = np.where(mask ==1)
    idx_bounds = np.where(bounds ==1)
    # # Use mask to find interesting samples
    # sample_pot = np.where(mask > mask_threshold)[1:]
    # sample_boundary_points = np.where(bounds > mask_threshold)[1:]
    # rng = np.random.default_rng()

    # # Sample <scatter_percent> samples
    # sample_idx = rng.choice(len(sample_pot[0]), replace=False, size=(int(len(sample_pot[0])*scatter_percent)))

    # # Get indexes
    # x_idx = sample_pot[0][sample_idx]
    # y_idx = sample_pot[1][sample_idx]
    # z_idx = sample_pot[2][sample_idx]

    # Get velocity values in all directions

    hr_u = np.asarray(gt['u'][frame_idx])
    hr_u_vals = hr_u[idx_inner]
    hr_u_bounds = hr_u[idx_bounds]
    hr_v = np.asarray(gt['v'][frame_idx])
    hr_v_vals = hr_v[idx_inner]
    hr_v_bounds = hr_v[idx_bounds]
    hr_w = np.asarray(gt['w'][frame_idx])
    hr_w_vals = hr_w[idx_inner]
    hr_w_bounds = hr_w[idx_bounds]

  
    sr_u = np.asarray(prediction['u'][frame_idx])
    sr_u_vals = sr_u[idx_inner]
    sr_u_bounds = sr_u[idx_bounds]
    sr_v = np.asarray(prediction['v'][frame_idx])
    sr_v_vals = sr_v[idx_inner]
    sr_v_bounds = sr_v[idx_bounds]
    sr_w = np.asarray(prediction['w'][frame_idx])
    sr_w_vals = sr_w[idx_inner]
    sr_w_bounds = sr_w[idx_bounds]

    def plot_regression(hr_vals, sr_vals, hr_vals_bounds, sr_vals_bounds):
        dimension = 2 #TODO
        plt.scatter(hr_vals, sr_vals, s=0.3, c=["black"])
        plt.scatter(hr_vals_bounds, sr_vals_bounds, s=0.3, c=["red"])
        plt.plot(np.linspace(np.min(hr_vals), np.max(hr_vals)), np.linspace(np.min(hr_vals), np.max(hr_vals)), '--', color= 'grey')
        plt.title(f"V_{dimension}")
        plt.xlabel("V_HR [m/s]")
        plt.ylabel("V_SR [m/s]")

    
    print(f"Plotting regression lines...")

    plot_regression(hr_u_vals, sr_u_vals, hr_u_bounds, sr_u_bounds)
    plt.savefig(f"{save_as}_LRXplot.png")
    plot_regression(hr_v_vals, sr_v_vals, hr_v_bounds, sr_v_bounds)
    plt.savefig(f"{save_as}_LRYplot.png")
    plot_regression(hr_w_vals, sr_w_vals, hr_w_bounds, sr_w_bounds)
    plt.savefig(f"{save_as}_LRZplot.png")



def get_slice(data, frame, axis, slice_idx):
    if len(data.squeeze().shape) == 3:
        frame = 0
        print("Only one frame available: take first frame.")
        if len(data.shape) ==3:
            data = np.expand_dims(data, 0)
        
    if axis == 0 :
        return data[frame, slice_idx, :, :]
    elif axis == 1:
        return data[frame, :, slice_idx, :]
    elif axis == 2:
        return data[frame, :, :, slice_idx]
    else: 
        print("Invalid axis! Axis must be 0, 1 or 2")

def crop_center(img,cropx,cropy):
    #from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def get_boundaries(binary_mask):
    '''
    returns a 2d array with same shape as binary mask, 1: boundary point, 0 no boundary point
    '''

    kernel_x = np.array([[-1, 0, 1]])
    kernel_y = kernel_x.transpose()

    boundary = convolve2d(binary_mask, kernel_x, mode ='same') + convolve2d(binary_mask, kernel_y, mode = 'same' )
    boundary[np.where(boundary !=0)] = 1

    return boundary


def plot_comparison(low_res, ground_truth, prediction, frame_idx = 10, axis=1, slice_idx = 50):

    if frame_idx% 2 != 0 : print("Slice index should be even!")
    fig, ax = plt.subplots(3, 3)

    patch = [40, 40]

    vel_colnames = ['u', 'v', 'w', 'div_x']#, 'divergence_y', 'divergence_z']
    vel_plotnames = ['Vx', r'Vy', r'Vz']
    n = 1

    #calculate divergence
    ground_truth['div_x'], ground_truth['div_y'], ground_truth['div_z'] = np.asarray(calculate_divergence(ground_truth['u'], ground_truth['v'], ground_truth['w']))
    low_res['div_x'], low_res['div_y'], low_res['div_z'] = np.asarray(calculate_divergence(low_res['u'], low_res['v'], low_res['w']))
    prediction['div_x'], prediction['div_y'], prediction['div_z'] = np.asarray(calculate_divergence(prediction['u'], prediction['v'], prediction['w']))


    for i, vel in enumerate(vel_colnames):
        slice_lr = get_slice(low_res[vel], frame_idx, axis, slice_idx//2)
        slice_gt = get_slice(ground_truth[vel], frame_idx, axis, slice_idx)
        slice_sr = get_slice(prediction[vel], frame_idx, axis, slice_idx)

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




