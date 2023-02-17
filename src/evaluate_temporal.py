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
from utils.evaluate_utils import *

if __name__ == "__main__":


    # Define directories and filenames
    model_name = '20230210-0333'
    set_name = 'Test'
    data_model= '3'

    #directories
    gt_dir = 'Temporal4DFlowNet/data/CARDIAC'
    result_dir = f'Temporal4DFlowNet/results/Temporal4DFlowNet_{model_name}'
    eval_dir = f'{result_dir}/plots'
    lr_dir = 'Temporal4DFlowNet/data/CARDIAC'
    model_dir = 'Temporal4DFlowNet/models'

    #filenames
    gt_filename = f'M{data_model}_2mm_step5_static.h5'
    lr_filename = f'M{data_model}_2mm_step5_static_TLR.h5'
    result_filename = f'{set_name}set_result_model{data_model}_2_{model_name[-4::]}_temporal_new.h5'
    evaluation_filename = f'eval_rel_err_{data_model}_2_{model_name[-4::]}_temporal.h5'
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

    with h5py.File(res_filepath, mode = 'r' ) as h_pred:
        with h5py.File(gt_filepath, mode = 'r' ) as h_gt:
            with h5py.File(lr_filepath, mode = 'r' ) as h_lr:

                # adapt dimension
                for vel in vel_colnames:
                    
                    gt[vel] = np.asarray(h_gt[vel])#
                    
                    
                    pred[vel] = np.asarray(h_pred[vel])
                    lr[vel]= np.asarray(h_lr[vel])

                    #transpose for temporal resolution
                    pred[vel] = pred[vel].transpose(1, 0, 2, 3)
                    print("gt shape:", gt[vel].shape, pred[vel].shape)
                    gt[vel] = crop_gt(gt[vel], pred[vel].shape)
                    print("gt shape:", gt[vel].shape, pred[vel].shape)                    

                gt["mask"] = crop_gt(np.asarray(h_gt["mask"]), pred["u"].shape[1::])

    #show_quiver(gt["u"][4, :, :, :], gt["v"][4, :, :, :], gt["w"][4, :, :, :],gt["mask"], save_as=f'{result_dir}/plots/test_quiver.png')

    show_temporal_development_line(gt["u"], lr["u"], pred["u"],gt["mask"], axis=3, indices=(20,20), save_as=f'{eval_dir}/{set_name}_temporal_development.png')
    plt.clf()

    #check that dimension fits
    assert(gt["u"].shape == pred["u"].shape)  ,str(pred["u"].shape) + str(gt["u"].shape) # dimensions need to be the same
    
    mask_diff, mask_pred = compare_masks(gt["u"], gt["v"] , gt["w"], gt["mask"])
    
    
    # calculate relative error
    error_gt = calculate_relative_error_normalized(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
    error_pred = calculate_relative_error_normalized(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], mask_pred)
    
    # get pointwise error
    error_pointwise, err_u, err_v, err_w = calculate_pointwise_error(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
    error_pointwise_cap = error_pointwise.copy()
    error_pointwise_cap[np.where(error_pointwise_cap>1)] = 1

    # mean speed of gt and prediction
    mean_speed_gt = calculate_mean_speed(gt["u"], gt["v"] , gt["w"], gt["mask"])
    mean_speed_pred = calculate_mean_speed(pred["u"], pred["v"] , pred["w"], gt["mask"])

    dt["u"] = calculate_temporal_derivative(gt["u"], timestep=1)
    dt["v"] = calculate_temporal_derivative(gt["v"], timestep=1)
    dt["w"] = calculate_temporal_derivative(gt["w"], timestep=1)

    show_timeframes(gt["u"], lr["u"], pred["u"],gt["mask"],error_pointwise_cap ,dt["u"] ,timepoints=[4, 5, 6], axis=0, idx = 22, save_as=f'{eval_dir}/{set_name}_time_frame_examples_VX.png')
    show_timeframes(gt["v"], lr["v"], pred["u"],gt["mask"],error_pointwise_cap ,dt["v"] ,timepoints=[4, 5, 6 ], axis=0, idx = 22, save_as=f'{eval_dir}/{set_name}_time_frame_examples_VY.png')
    show_timeframes(gt["w"], lr["w"], pred["u"],gt["mask"],error_pointwise_cap ,dt["w"] ,timepoints=[4, 5, 6], axis=0, idx = 22, save_as=f'{eval_dir}/{set_name}_time_frame_examples_VZ.png')

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
    

    plot_regression(gt, pred, frame_idx= 10, save_as=f'{result_dir}/plots/{set_name}_regression_')
    plt.clf()

    # plot_comparison_temporal(lr, gt, pred, frame_idx = 8, axis=1, slice_idx = 40, save_as=f'{eval_dir}/{set_name}_visualize_interporalion_comparison.png')
    # plt.clf()

    #Plot Relative error
    plt.plot(error_gt, '-',  error_pred, '--',50*np.ones(len(error_gt)), 'k:')
    plt.xlabel("Frame")
    plt.ylabel("Relative error (%)")
    plt.ylim((0, 100))
    plt.legend(["gt mask", "mask from u"])
    plt.title("Relative error - " + str(set_name) +" - " + str(model_name))
    plt.savefig(f'{eval_dir}/{set_name}_error_t.png')
    plt.clf()

    #Plot Mean speed
    plt.plot(mean_speed_gt, '-', mean_speed_pred, '--')
    plt.xlabel("Frame")
    plt.ylabel("Mean speed (cm/s)" )
    plt.title("Mean speed (cm/s) - " + str(set_name)+ " - " + str(model_name))
    plt.legend(["gt", "pred"])
    plt.savefig(f'{eval_dir}/{set_name}_mean_speed.png')

            

