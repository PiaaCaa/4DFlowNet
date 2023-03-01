import tensorflow as tf
import numpy as np
import time
import os
from Network.SR4DFlowNet import SR4DFlowNet
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset import ImageDataset
from matplotlib import pyplot as plt
import h5py
from prepare_data.visualize_utils import generate_gif_volume
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.figsize'] = [10, 8]
from utils.evaluate_utils import *

if __name__ == "__main__":


    # Define directories and filenames
    model_name = '20230220-0908'    
    set_name = 'Training'               
    data_model= '2'
    step = 2

    #directories
    gt_dir = 'Temporal4DFlowNet/data/CARDIAC'
    result_dir = f'Temporal4DFlowNet/results/Temporal4DFlowNet_{model_name}'
    eval_dir = f'{result_dir}/plots'
    lr_dir = 'Temporal4DFlowNet/data/CARDIAC'
    model_dir = 'Temporal4DFlowNet/models'

    #filenames
    gt_filename = f'M{data_model}_2mm_step{step}_static.h5'
    lr_filename = f'M{data_model}_2mm_step{step}_static_noise.h5'
    result_filename = f'{set_name}set_result_model{data_model}_2mm_step{step}_{model_name[-4::]}_temporal_.h5'#newpadding.h5'
    evaluation_filename = f'eval_rel_err_{data_model}_2mm_step{step}_{model_name[-4::]}_temporal.h5'
    model_filename = f'Temporal4DFlowNet_{model_name}/Temporal4DFlowNet-best.h5'


    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    
    #Params for evalation
    save_relative_error_file= False

    # Setting up
    gt_filepath = '{}/{}'.format(gt_dir, gt_filename)
    res_filepath = '{}/{}'.format(result_dir, result_filename)
    lr_filepath = '{}/{}'.format(lr_dir, lr_filename)
    model_path = '{}/{}'.format(model_dir, model_filename)

    if save_relative_error_file:
        assert(not os.path.exists(f'{result_dir}/{evaluation_filename}')) #STOP if relative error file is already created

    vel_colnames=['u', 'v', 'w']
    gt = {}
    lr = {}
    pred = {}
    dt = {}

    #load predictions 
    with h5py.File(res_filepath, mode = 'r' ) as h_pred:
        with h5py.File(gt_filepath, mode = 'r' ) as h_gt:
            with h5py.File(lr_filepath, mode = 'r' ) as h_lr:
                
                gt["mask"] = np.asarray(h_gt["mask"])
                gt["mask"][np.where(gt["mask"] !=0)] = 1
                temporal_mask = create_temporal_mask(gt["mask"], h_gt['u'].shape[0])

                # adapt dimension
                for vel in vel_colnames:
                    
                    gt[vel] = np.asarray(h_gt[vel])
                    pred[vel] = np.asarray(h_pred[f'{vel}_combined']) # TODO chnaged this with new combination of all axis 
                    lr[vel] = np.asarray(h_lr[vel])[::2, :, :, :] #TODO: this chnaged with the new loading modules
                    #transpose for temporal resolution
                    #pred[vel] = pred[vel].transpose(1, 0, 2, 3) #TODO changed for new csv file

                    pred[f'{vel}_fluid'] = np.multiply(pred[vel], temporal_mask)
                    lr[f'{vel}_fluid'] = np.multiply(lr[vel], temporal_mask[::2, :, :, :])
                    gt[f'{vel}_fluid'] = np.multiply(gt[vel], temporal_mask)

    #check that dimension fits
    assert(gt["u"].shape == pred["u"].shape)  ,str(pred["u"].shape) + str(gt["u"].shape) # dimensions need to be the same
    
    #show_quiver(gt["u"][4, :, :, :], gt["v"][4, :, :, :], gt["w"][4, :, :, :],gt["mask"], save_as=f'{result_dir}/plots/test_quiver.png')
    min_v = {}
    max_v = {}
    for vel in vel_colnames:
        min_v[vel] = np.quantile(gt[vel][np.where(temporal_mask !=0)].flatten(), 0.01)
        max_v[vel] = np.quantile(gt[vel][np.where(temporal_mask !=0)].flatten(), 0.99)

    # get interpoaltion results
    interpolate_NN = {}
    interpolate_linear = {}
    interpolate_cubic = {}

    for vel in vel_colnames:
        interpolate_linear[vel] = temporal_linear_interpolation(lr[vel], gt[vel].shape)
        interpolate_linear[f'{vel}_fluid'] = np.multiply(interpolate_linear[vel], gt['mask'])

        interpolate_cubic[vel] = temporal_cubic_interpolation(lr[vel], gt[vel].shape)
        interpolate_cubic[f'{vel}_fluid'] = np.multiply(interpolate_cubic[vel], gt['mask'])

        interpolate_NN[vel] = temporal_NN_interpolation(lr[vel], gt[vel].shape)
        interpolate_NN[f'{vel}_fluid'] = np.multiply(interpolate_NN[vel], gt['mask'])

    show_temporal_development_line(gt["u"], interpolate_linear["u"], pred["u"],gt["mask"], axis=3, indices=(20,20), save_as=f'{eval_dir}/{set_name}_temporal_development.png')
    plt.clf()
    
    T_peak_flow = np.unravel_index(np.argmax(gt["u"]), shape =gt["u"].shape)[0]
    print("Peak flow frame", T_peak_flow)
    if T_peak_flow%2==0: T_peak_flow +=1
    #mask_diff, mask_pred = compare_masks(gt["u"], gt["v"] , gt["w"], gt["mask"])
    
    #calculate relative error
    rel_error = calculate_relative_error_normalized(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
    error_pointwise, error_absolut = calculate_pointwise_error(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
    rel_error_lin_interpolation =   calculate_relative_error_normalized(interpolate_linear["u"], interpolate_linear["v"], interpolate_linear["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
    rel_error_cubic_interpolation = calculate_relative_error_normalized(interpolate_cubic["u"], interpolate_cubic["v"], interpolate_cubic["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])

    error_pointwise_cap = error_pointwise.copy()
    error_pointwise_cap[np.where(error_pointwise_cap>1)] = 1
    for vel in vel_colnames:
        error_absolut[f'{vel}_fluid'] = np.multiply(error_absolut[vel], gt["mask"])

    # mean speed of gt and prediction
    mean_speed_gt =                 calculate_mean_speed(gt["u_fluid"], gt["v_fluid"] , gt["w_fluid"], gt["mask"])
    mean_speed_pred =               calculate_mean_speed(pred["u_fluid"], pred["v_fluid"] , pred["w_fluid"], gt["mask"])
    mean_speed_lin_interpolation =  calculate_mean_speed(interpolate_linear["u_fluid"], interpolate_linear["v_fluid"] ,interpolate_linear["w_fluid"], gt["mask"])
    mean_speed_cubic_interpolation = calculate_mean_speed(interpolate_cubic["u_fluid"], interpolate_cubic["v_fluid"] , interpolate_cubic["w_fluid"], gt["mask"])


    # dt["u"] = calculate_temporal_derivative(gt["u"], timestep=1)
    # dt["v"] = calculate_temporal_derivative(gt["v"], timestep=1)
    # dt["w"] = calculate_temporal_derivative(gt["w"], timestep=1)

    print("Plot example time frames..")
    show_timeframes(gt["u"], lr["u"], pred["u"],gt["mask"],error_pointwise_cap ,[interpolate_linear["u"], interpolate_cubic["u"]], ["linear", "cubic"] ,timepoints=[4, 5, 6], axis=0, idx = 22,min_v = min_v["u"],max_v =max_v["u"], save_as=f'{eval_dir}/{set_name}_time_frame_examples_VX.png')
    show_timeframes(gt["v"], lr["v"], pred["v"],gt["mask"],error_pointwise_cap ,[interpolate_linear["v"], interpolate_cubic["v"]], ["linear", "cubic"] ,timepoints=[4, 5, 6], axis=0, idx = 22,min_v = min_v["v"],max_v =max_v["v"], save_as=f'{eval_dir}/{set_name}_time_frame_examples_VY.png')
    show_timeframes(gt["w"], lr["w"], pred["w"],gt["mask"],error_pointwise_cap ,[interpolate_linear["w"], interpolate_cubic["w"]], ["linear", "cubic"] ,timepoints=[4, 5, 6], axis=0, idx = 22,min_v = min_v["w"],max_v =max_v["w"], save_as=f'{eval_dir}/{set_name}_time_frame_examples_VZ.png')
    plt.clf()
    #evaluate where the higest shift (temporal derivative) is for each frame
    
    #save
    if save_relative_error_file:
        prediction_utils.save_to_h5(f'{eval_dir}/{evaluation_filename}', "dt_u", dt["u"], compression='gzip')
        prediction_utils.save_to_h5(f'{eval_dir}/{evaluation_filename}', "relative_error", error_pointwise, compression='gzip')
        prediction_utils.save_to_h5(f'{eval_dir}/{evaluation_filename}', "relative_error_cap", error_pointwise_cap, compression='gzip')
        prediction_utils.save_to_h5(f'{eval_dir}/{evaluation_filename}', "error_u", err_u, compression='gzip')
        prediction_utils.save_to_h5(f'{eval_dir}/{evaluation_filename}', "error_v", err_v, compression='gzip')
        prediction_utils.save_to_h5(f'{eval_dir}/{evaluation_filename}', "error_w", err_w, compression='gzip')
        prediction_utils.save_to_h5(f'{eval_dir}/{evaluation_filename}', "mask_check", mask_diff, compression='gzip')
        # prediction_utils.save_to_h5(f'{eval_dir}/{evaluation_filename}', "mask_u", mask_pred, compression='gzip')
    
    plt.clf()
    plot_regression(gt, pred, frame_idx = T_peak_flow, save_as=f'{result_dir}/plots/{set_name}_regression_')
    plt.clf()

    # plot_comparison_temporal(lr, gt, pred, frame_idx = 8, axis=1, slice_idx = 40, save_as=f'{eval_dir}/{set_name}_visualize_interporalion_comparison.png')
    # plt.clf()

    print("Plot relative error and mean speed")
    plt.subplot(2, 1, 1)
    plt.title("Relative error")
    plt.plot(rel_error, label = 'averaged prediction')
    plt.plot(rel_error_lin_interpolation[:-1], label = 'linear interpolation',color = 'yellowgreen')
    plt.plot(rel_error_cubic_interpolation, label = 'cubic interpolation', color = 'forestgreen')
    plt.plot(50*np.ones(len(rel_error)), 'k:')
    plt.xlabel("Frame")
    plt.ylabel("Relative error (%)")
    plt.ylim((0, 100))
    plt.legend()
    plt.savefig(f'{eval_dir}/{set_name}_rel_error_1.svg')

    plt.subplot(2, 1, 2)
    plt.plot(mean_speed_gt, label ='Ground truth',color = 'black')
    plt.plot(mean_speed_pred,'b', label= set_name, color = 'steelblue')
    plt.plot(mean_speed_lin_interpolation[:-1], label = 'linear interpolation', color = 'yellowgreen')
    plt.plot(mean_speed_cubic_interpolation[:-1], label = 'cubic interpoaltion', color = 'forestgreen')
    plt.xlabel("Frame")
    plt.ylabel("Mean speed (cm/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{eval_dir}/{set_name}_rel_error_.svg')
    plt.show()

            

