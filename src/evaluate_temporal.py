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
    gt_dir = 'Temporal4DFlowNet/data/CARDIAC'
    result_dir = 'Temporal4DFlowNet/results'
    lr_dir = 'Temporal4DFlowNet/data/CARDIAC'

    gt_filename = 'M1_2mm_step5_static.h5'
    lr_filename = 'M1_2mm_step5_static_TLR.h5'
    result_filename = 'test_result_MODEL3_2_temporal_test_test.h5'
    evaluation_filename = 'plots/eval_rel_err_1_2_temporal.h5'
    
    model_path = "4DFlowNet/models/Temporal4DFlowNet_20230208-0717/Temporal4DFlowNet-best.h5"

    # Setting up
    gt_filepath = '{}/{}'.format(gt_dir, gt_filename)
    res_filepath = '{}/{}'.format(result_dir, result_filename)
    lr_filepath = '{}/{}'.format(lr_dir, lr_filename)


    vel_colnames=['u', 'v', 'w']
    gt = {}
    lr = {}
    pred = {}

    with h5py.File(res_filepath, mode = 'r' ) as h_pred:
        with h5py.File(gt_filepath, mode = 'r' ) as h_gt:
            with h5py.File(lr_filepath, mode = 'r' ) as h_lr:

                # adapt dimension
                for vel in vel_colnames:
                    
                    gt[vel] = np.asarray(h_gt[vel])#
                    pred[vel] = np.asarray(h_pred[vel])
                    lr[vel]= np.asarray(h_lr[vel])

                    #transnpose for temporal resolution
                    pred[vel] = pred[vel].transpose(1, 0, 2, 3)

                    print("prediction shape:", pred[vel].shape, "gt:", gt[vel].shape)

                    gt[vel] = crop_gt(gt[vel], pred[vel].shape)
                

                gt["mask"] = crop_gt(np.asarray(h_gt["mask"]), pred["u"].shape[1::])

                #check that dimension fits
                assert(gt["u"].shape == pred["u"].shape)  ,str(pred["u"].shape) + str(gt["u"].shape) # dimensions need to be the same
                
                mask_diff, mask_pred = compare_masks(gt["u"], gt["v"] , gt["w"], gt["mask"])

                # calculate relative error
                error_gt = calculate_relative_error_np(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
                error_pred = calculate_relative_error_np(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], mask_pred)
                # get pointwise error
                error_pointwise, err_u, err_v, err_w = calculate_pointwise_error(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
                error_pointwise_cap = error_pointwise.copy()
                error_pointwise_cap[np.where(error_pointwise_cap>1)] = 1

                # mean speed of gt and prediction
                mean_speed_gt = calculate_mean_speed(gt["u"], gt["v"] , gt["w"], gt["mask"])
                mean_speed_pred = calculate_mean_speed(pred["u"], pred["v"] , pred["w"], gt["mask"])

                #save
                prediction_utils.save_to_h5(f'{result_dir}/{evaluation_filename}', "relative_error", error_pointwise, compression='gzip')
                prediction_utils.save_to_h5(f'{result_dir}/{evaluation_filename}', "relative_error_cap", error_pointwise_cap, compression='gzip')
                prediction_utils.save_to_h5(f'{result_dir}/{evaluation_filename}', "error_u", err_u, compression='gzip')
                prediction_utils.save_to_h5(f'{result_dir}/{evaluation_filename}', "error_v", err_v, compression='gzip')
                prediction_utils.save_to_h5(f'{result_dir}/{evaluation_filename}', "error_w", err_w, compression='gzip')
                prediction_utils.save_to_h5(f'{result_dir}/{evaluation_filename}', "mask_check", mask_diff, compression='gzip')
                prediction_utils.save_to_h5(f'{result_dir}/{evaluation_filename}', "mask_u", mask_pred, compression='gzip')
    

    plot_regression(gt, pred, frame_idx= 10, save_as="4DFlowNet/results/plots/reg_example")
    plt.clf()

    plot_comparison(lr, gt, pred)
    plt.clf()

    plt.plot(error_gt, '-',  error_pred, '--',50*np.ones(len(error_gt)), 'k:')
    plt.xlabel("Frame")
    plt.ylabel("Relative error (%)")
    plt.ylim((0, 100))
    plt.legend(["gt mask", "mask from u"])
    plt.title("Relative error")
    plt.savefig("4DFlowNet/results/plots/error_t.png")
    plt.clf()

    plt.plot(mean_speed_gt, '-', mean_speed_pred, '--')
    plt.xlabel("Frame")
    plt.ylabel("Mean speed (cm/s)")
    plt.legend(["gt", "pred"])
    plt.savefig("4DFlowNet/results/plots/mean_speed.png")

            

