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
from scipy.ndimage import convolve
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

#copied from evulation utils
def random_indices3D(mask, n):
    mask_threshold = 0.9
    sample_pot = np.where(mask > mask_threshold)
    rng = np.random.default_rng()
    scatter_percent = 0.05

    # # Sample <scatter_percent> samples
    sample_idx = rng.choice(len(sample_pot[0]), replace=False, size=n)

    # # Get indexes
    x_idx = sample_pot[0][sample_idx]
    y_idx = sample_pot[1][sample_idx]
    z_idx = sample_pot[2][sample_idx]
    return x_idx, y_idx, z_idx


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

def calculate_relative_error_normalized(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5

    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = np.sqrt(u_diff + v_diff + w_diff)
    actual_speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi)) 

    print("max/min before arctan", np.max(diff_speed / (actual_speed + epsilon)), np.min(diff_speed / (actual_speed + epsilon)))

    # actual speed can be 0, resulting in inf
    relative_speed_loss = np.arctan(diff_speed / (actual_speed + epsilon))
    print("max/min after arctan", np.max(relative_speed_loss), np.min(relative_speed_loss))
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


def plot_regression(gt, prediction, frame_idx, save_as = None):
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

    def plot_regression_points(hr_vals, sr_vals, hr_vals_bounds, sr_vals_bounds):
        dimension = 2 #TODO
        plt.scatter(hr_vals, sr_vals, s=0.3, c=["black"])
        plt.scatter(hr_vals_bounds, sr_vals_bounds, s=0.3, c=["red"])
        plt.plot(np.linspace(np.min(hr_vals), np.max(hr_vals)), np.linspace(np.min(hr_vals), np.max(hr_vals)), '--', color= 'grey')
        plt.title(f"V_{dimension}")
        plt.xlabel("V_HR [m/s]")
        plt.ylabel("V_SR [m/s]")

    
    print(f"Plotting regression lines...")

    plt.subplot(1, 3, 1)
    plot_regression_points(hr_u_vals, sr_u_vals, hr_u_bounds, sr_u_bounds)
    if save_as is not None: plt.savefig(f"{save_as}_LRXplot.png")
    plt.subplot(1 ,3, 2)
    plot_regression_points(hr_v_vals, sr_v_vals, hr_v_bounds, sr_v_bounds)
    if save_as is not None: plt.savefig(f"{save_as}_LRYplot.png")
    plt.subplot(1, 3, 3)
    plot_regression_points(hr_w_vals, sr_w_vals, hr_w_bounds, sr_w_bounds)
    if save_as is not None: plt.savefig(f"{save_as}_LRZplot.png")

    # fig, axs = plt.subplots(nrows=1, ncols=3)
    # plt.subplot(1, 3, 1)
    # plot_regression_points()
    # axs[1].plot(xs, np.sqrt(xs))



def get_slice(data, frame, axis, slice_idx):
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
        slice_lr = get_slice(low_res[vel], frame_idx//2, axis, slice_idx)
        slice_gt = get_slice(ground_truth[vel], frame_idx, axis, slice_idx)
        slice_sr = get_slice(prediction[vel], frame_idx, axis, slice_idx)

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

    plt.savefig(save_as)

def show_quiver( u, v, w, mask,save_as = "3DFlow.png"):
    x_len, y_len, z_len = u.shape
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(y_len),np.arange(x_len),np.arange(z_len))
    print("x shape:", x.shape, y.shape, "u: ", u.shape)
    
    set_to_zero = 0.9
    x_idx, y_idx, z_idx = random_indices3D(mask, int(np.count_nonzero(mask)*set_to_zero))
    # u[x_idx, y_idx, z_idx] = 0
    # v[x_idx, y_idx, z_idx] = 0
    # w[x_idx, y_idx, z_idx] = 0
    
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


def show_timeframes(gt,lr,  pred,mask, rel_error, dt,  timepoints, axis, idx, save_as = "Frame_comparison.png"):
    plt.clf()
    T = len(timepoints)
    i = 1
    for j,t in enumerate(timepoints):
        
        gt_slice = get_slice(gt, t,  axis=axis, slice_idx=idx )
        pred_slice = get_slice(pred, t, axis=axis, slice_idx=idx )

        lr_slice = np.zeros_like(gt_slice)
        if t%2 == 0: lr_slice = get_slice(lr, t//2, axis=axis, slice_idx=idx )
        
        min_v = np.min([np.min(pred_slice ), np.min(gt_slice), np.min(lr_slice)])
        max_v = np.max([np.max(pred_slice), np.max(gt_slice), np.max(lr_slice)])  

        plt.subplot(T, 3, i)

        if t%2 == 0:
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='jet')
            if i == 1: plt.title("LR")
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('frame = '+ str(t))
            
        else:
            plt.axis('off')
        

        i +=1
        plt.subplot(T, 3, i)
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 2: plt.title("GT")
        plt.xticks([])
        plt.yticks([])

        i +=1
        plt.subplot(T, 3, i)
        plt.imshow(pred_slice, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 3: plt.title("SR")
        plt.xticks([])
        plt.yticks([])
        i +=1

    
    plt.savefig(save_as)
    plt.clf()

    mask[np.where(mask !=0)] = 1
    gt = np.multiply(gt, mask)
    lr = np.multiply(lr, mask)
    pred = np.multiply(pred, mask)

    i=1
    for j,t in enumerate(timepoints):
        
        gt_slice = get_slice(gt, t,  axis=axis, slice_idx=idx )
        pred_slice = get_slice(pred, t, axis=axis, slice_idx=idx )
        err_slice = get_slice(rel_error, t, axis=axis, slice_idx=idx )
        dt_slice = get_slice(dt, t, axis=axis, slice_idx=idx )
        print("shape dt:", dt.shape, dt_slice.shape, gt_slice.shape )

        lr_slice = np.zeros_like(gt_slice)
        if t%2 == 0: lr_slice = get_slice(lr, t//2, axis= axis, slice_idx= idx )

        min_v = np.min([np.min(pred_slice ), np.min(gt_slice), np.min(lr_slice)])
        max_v = np.max([np.max(pred_slice), np.max(gt_slice), np.max(lr_slice)])  

        plt.subplot(T, 5, i)
        if t%2 == 0:
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='jet')
            if i == 1: plt.title("LR")
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('frame = '+ str(t))
        else:
            plt.axis('off')

        i += 1
        plt.subplot(T, 5, i)
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 2: plt.title("GT")
        plt.xticks([])
        plt.yticks([])

        i += 1
        plt.subplot(T, 5, i)
        plt.imshow(pred_slice, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 3: plt.title("SR")
        plt.xticks([])
        plt.yticks([])

        

        i += 1
        plt.subplot(T, 5, i)
        plt.imshow(err_slice, cmap='jet')
        if i == 4: plt.title("Relative error")
        plt.xticks([])
        plt.yticks([])

        i +=1
        plt.subplot(T, 5, i)
        plt.imshow(dt_slice, cmap='jet')
        if i == 5: plt.title("|dt|")
        plt.xticks([])
        plt.yticks([])

        plt.colorbar()
        

        
        i +=1
        

    save_under = save_as[:-4]+ "_fluidregion.png"
    print("save with only fluid region visible", save_under)
    plt.savefig(save_under)
    #plt.clf()


def calculate_temporal_derivative(data, timestep=1):
    '''
    Calculate difference between two time frames and each voxel
    i.e. for u: dt u(t) = |u(t+1) - u(t)| / timestep
    '''
    # assume data of shape t, h, w, d
    # kernel_t = np.zeros((2, 1, 1, 1))
    # kernel_t[0] = -1
    # kernel_t[1] = 1
    # kernel_t = kernel_t/timestep
    # print(kernel_t)
    # #kernel_t = kernel_t.transpose((3, 1, 2, 0))
    # print(kernel_t.shape)

    n_frames = data.shape[0]
    dt =  np.zeros_like(data)
    for t in range(n_frames-1):
        dt[t, :, :, :] = (data[t+timestep, :, :, :] - data[t, :, :, :])/timestep

    dt = np.abs(dt)
    # dt = convolve(data, kernel_t)
    # print("dt shape:", dt.shape, data.shape)

    # #TODO delete later
    # #check one slice
    # slice_t1 = get_slice(data, 2, axis=0, slice_idx=20)
    # slice_t2 = get_slice(data, 3, axis=0, slice_idx=20)

    # res = slice_t2 - slice_t1

    # res_1 = get_slice(convolve(data, kernel_t.transpose(0, 1, 2, 3)), 2, axis= 0, slice_idx=20)
    # res_2 = get_slice(convolve(data, kernel_t.transpose(1, 0, 2, 3)), 2, axis= 0, slice_idx=20)
    # res_3 = get_slice(convolve(data, kernel_t.transpose(2, 1, 0, 3)), 2, axis= 0, slice_idx=20)
    # res_4 = get_slice(convolve(data, kernel_t.transpose(3, 1, 2, 0)), 2, axis= 0, slice_idx=20)

    # print("shapes", res_1.shape, res.shape)
    # print("Check if kernel is correct:")
    # print("Norm 1", np.linalg.norm(res_1-res))
    # print("Norm 2", np.linalg.norm(res_2-res))
    # print("Norm 3", np.linalg.norm(res_3-res))
    # print("Norm 3", np.linalg.norm(res_4-res))
    


    return dt


def plot_relative_error(lst_hgt_paths, lst_hpred_paths,lst_names, save_as = 'Relative_error_comparison.png'):
    assert(len(lst_hgt_paths)==len(lst_hpred_paths))
    vel_colnames=['u', 'v', 'w']

    for gt_path, pred_path in zip(lst_hgt_paths, lst_hpred_paths):
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
                plt.plot(error_gt, '-')


    plt.plot(50*np.ones(len(error_gt)), 'k:')
    plt.xlabel("Frame")
    plt.ylabel("Relative error (%)")
    plt.ylim((0, 100))
    plt.legend(lst_names)
    plt.title("Relative error")
    plt.savefig(save_as)
    #plt.clf()

def create_temporal_mask(mask, n_frames):
    '''
    from static mask create temporal mask of shape (n_frames, h, w, d)
    '''
    assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
    temporal_mask = np.zeros((n_frames, mask.shape[0], mask.shape[1], mask.shape[2]))
    for i in range(n_frames):
        temporal_mask[i, :, :, :] = mask
    return temporal_mask


def temporal_bilinear_interpolation(lr, hr_shape):
    interpolate = np.zeros(hr_shape)
    interpolate[::2, :, :, :] = lr
     # TODO double check this
    interpolate[1::2, :,:, :] = np.average(lr, axis=0)

    return interpolate






