import numpy as np
import time
import os
from matplotlib import pyplot as plt
import h5py
from collections import defaultdict

from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.evaluate_utils import *

plt.rcParams['figure.figsize'] = [10, 8]


def load_data(gt_filepath, lr_filepath, pred_filepath,  vel_colnames = ['u', 'v', 'w'],res_colnames = ['u_combined', 'v_combined', 'w_combined'], threshold = 0.5, offset = 0, factor = 2):
    

    gt = {}
    lr = {}
    pred = {}

    with h5py.File(pred_filepath, mode = 'r' ) as h5_pred:
        with h5py.File(gt_filepath, mode = 'r' ) as h5_gt:
            with h5py.File(lr_filepath, mode = 'r' ) as h5_lr:
                
                # load mask
                gt["mask"] = np.asarray(h5_gt["mask"]).squeeze()
                gt["mask"][np.where(gt["mask"] >= threshold)] = 1
                gt["mask"][np.where(gt["mask"] <  threshold)] = 0

                if len(gt['mask'].shape) == 3 : # check for dynamical mask, otherwise create one
                    gt["mask"] = create_dynamic_mask(gt["mask"], h5_gt['u'].shape[0])
                
                # check for LR dimension, two options: 
                # 1. LR has same temporal resolution as HR: downsampling is done here (on the fly)
                # 2. LR is already downsampled: only load dataset
                if h5_gt[vel_colnames[0]].shape[0] == h5_lr[vel_colnames[0]].shape[0]:
                    downsample_lr = True
                else:
                    downsample_lr = False

                if 'mask' in h5_lr.keys():
                    print('Load mask from low resolution file')
                    lr['mask'] = np.asarray(h5_lr['mask']).squeeze()
                else:
                    print('Create LR mask from HR mask')
                    lr['mask'] = gt["mask"][offset::factor, :, :, :].copy()

                # load velocity fields
                for vel, r_vel in zip(vel_colnames, res_colnames):
                    
                    gt[vel] = np.asarray(h5_gt[vel]).squeeze()
                    pred[vel] = np.asarray(h5_pred[r_vel]).squeeze()
                    if downsample_lr:
                        lr[vel] = np.asarray(h5_lr[vel])[offset::factor, :, :, :]
                    else:
                        lr[vel] = np.asarray(h5_lr[vel]).squeeze()  

                    # take away background outside mask
                    pred[f'{vel}_fluid'] =np.multiply(pred[vel], gt["mask"])
                    lr[f'{vel}_fluid'] =  np.multiply(lr[vel], lr['mask'])
                    gt[f'{vel}_fluid'] =  np.multiply(gt[vel], gt["mask"])

                    # Check that shapes match
                    assert gt[vel].shape == pred[vel].shape, f"Shape mismatch HR/SR: {gt[vel].shape} != {pred[vel].shape}"
                    
                #include speed calculations
                gt['speed']   = np.sqrt(gt["u"]**2 + gt["v"]**2 + gt["w"]**2)
                lr['speed']   = np.sqrt(lr["u"]**2 + lr["v"]**2 + lr["w"]**2)
                pred['speed'] = np.sqrt(pred["u"]**2 + pred["v"]**2 + pred["w"]**2)

                gt['speed_fluid']   = np.multiply(gt['speed'], gt["mask"])
                lr['speed_fluid']   = np.multiply(lr['speed'], lr['mask'])
                pred['speed_fluid'] = np.multiply(pred['speed'], gt["mask"])
    
    return gt, lr, pred



def load_interpolation(data_model, step, lr, gt):
    vel_colnames=['u', 'v', 'w']
    interpolate_NN = {}
    interpolate_linear = {}
    interpolate_cubic = {}


    inbetween_string = ''

    lr_filename = f'M{data_model}_2mm_step{step}_static{inbetween_string}_noise.h5'


    interpolation_dir = 'results/interpolation'
    # interpolation_filename = f'{lr_filename[:-3]}_interpolation'
    #M4_2mm_step2_static_dynamic_interpolate_no_noise'
    interpolation_filename = f'M{data_model}_2mm_step{step}_static{inbetween_string}_interpolate_no_noise'
    interpolation_path = f'{interpolation_dir}/{interpolation_filename}.h5'
    if not os.path.isfile(interpolation_path):
        print("Interpolation file does not exist - calculate interpolation and save files")
        print("Save interpolation files to: ", interpolation_path)
        
        #this can take a while
        for vel in vel_colnames:
            print("Interpolate low resolution images - ", vel)
            print(gt['mask'].shape)
            interpolate_linear[vel] = temporal_linear_interpolation_np(lr[vel], gt[vel].shape)
            interpolate_linear[f'{vel}_fluid'] = np.multiply(interpolate_linear[vel], gt['mask'])

            # interpolate_cubic[vel] = temporal_cubic_interpolation(lr[vel], gt[vel].shape)
            # interpolate_cubic[f'{vel}_fluid'] = np.multiply(interpolate_cubic[vel], gt['mask'])
            print("Cubic interpolation is not performed!! This has be implemented more memory efficient!")
            interpolate_cubic[vel] = np.ones_like(interpolate_linear[vel])
            interpolate_cubic[vel] = np.ones_like(interpolate_linear[vel])

            interpolate_NN[vel] = temporal_NN_interpolation(lr[vel], gt[vel].shape)
            interpolate_cubic[f'{vel}_fluid'] = np.multiply(interpolate_cubic[vel], gt['mask'])

            
            prediction_utils.save_to_h5(interpolation_path, f'linear_{vel}' , interpolate_linear[vel], compression='gzip')
            # prediction_utils.save_to_h5(interpolation_file, f'cubic_{vel}' , interpolate_cubic[vel], compression='gzip')
            prediction_utils.save_to_h5(interpolation_path, f'NN_{vel}' , interpolate_NN[vel], compression='gzip')
    else:
        print("Load existing interpolation file")
        with h5py.File(interpolation_path, mode = 'r' ) as h_interpolate:
            for vel in vel_colnames:
                interpolate_linear[vel] = np.array(h_interpolate[f'linear_{vel}'])
                interpolate_cubic[vel] =  np.ones_like(interpolate_linear[vel])#np.array(h_interpolate[f'cubic_{vel}'])
                interpolate_NN[vel] =     np.array(h_interpolate[f'NN_{vel}'])

                print("Cubic interpolation is not performed!! This has be implemented more memory efficient!")


                interpolate_linear[f'{vel}_fluid'] = np.multiply(interpolate_linear[vel], gt['mask'])
                interpolate_cubic[f'{vel}_fluid'] = np.multiply(interpolate_cubic[vel], gt['mask'])
                interpolate_NN[f'{vel}_fluid'] = np.multiply(interpolate_NN[vel], gt['mask'])

    return interpolate_linear, interpolate_cubic, interpolate_NN


if __name__ == "__main__":

    # Define directories and filenames
    model_name = '20240605-1504' 
    set_name = 'Validation'               
    data_model= '1'
    step = 2
    load_interpolation_files = False
    ups_factor = 2

    #choose which plots to show
    show_img_plot = False
    show_RE_plot = False
    show_corr_plot = False
    tabular_eval = True
    show_animation = False #TODO: implement


    # directories
    data_dir = 'data/CARDIAC'
    pred_dir = f'results/Temporal4DFlowNet_{model_name}'
    eval_dir = f'{pred_dir}/plots'

    vel_colnames=['u', 'v', 'w']

    # filenames
    gt_filename = f'M{data_model}_2mm_step{step}_cs_invivoP01_hr.h5'
    lr_filename = f'M{data_model}_2mm_step{step}_cs_invivoP01_lr.h5'
    # gt_filename = f'M{data_model}_2mm_step2_flowermagn_boxavg_HRfct.h5'
    # lr_filename = f'M{data_model}_2mm_step2_flowermagn_boxavg_LRfct_noise.h5'

    pred_filename = f'{set_name}set_result_model{data_model}_2mm_step{step}_{model_name[-4::]}_temporal2.h5'
    
    # Setting up
    gt_filepath  = '{}/{}'.format(data_dir, gt_filename)
    pred_filepath = '{}/{}'.format(pred_dir, pred_filename)
    lr_filepath  = '{}/{}'.format(data_dir, lr_filename)

    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    # ----------Load data and interpolation files and calculate visualization params----------------

    gt, lr, pred = load_data(gt_filepath, lr_filepath, pred_filepath, vel_colnames = vel_colnames)

    N_frames = gt['u'].shape[0]

    if load_interpolation_files: 
        # interpolate_linear, interpolate_cubic, interpolate_NN = load_interpolation(data_model, step,lr, gt)
        # load sinc interpolation
        interpolate_sinc = {}
        hr_range = np.linspace(0,1,  gt['u'].shape[0])
        lr_range = hr_range[::2] # downsamplie like this to get exact same evaluation points

        for vel in vel_colnames:
            interpolate_sinc[vel] = temporal_sinc_interpolation_ndarray(lr[vel], lr_range, hr_range)
            interpolate_sinc[f'{vel}_fluid'] = np.multiply(interpolate_sinc[vel], gt['mask'])

    # check that dimension fits
    assert(gt["u"].shape == pred["u"].shape)  ,str(pred["u"].shape) + str(gt["u"].shape) # dimensions need to be the same
    assert(gt["u"].shape[1::] == lr["u"].shape[1::])    ,str(lr["u"].shape) + str(gt["u"].shape) # spatial dimensions need to be the same
    
    # calculate velocity values in 1% and 99% quantile for plotting 
    min_v = {}
    max_v = {}
    for vel in vel_colnames:
        min_v[vel] = np.quantile(gt[vel][np.where(gt['mask'] !=0)].flatten(), 0.01)
        max_v[vel] = np.quantile(gt[vel][np.where(gt['mask'] !=0)].flatten(), 0.99)

    # calculate boundaries and core mask
    boundary_mask, core_mask = get_boundaries(gt["mask"])
    bool_mask = gt['mask'].astype(bool)
    reverse_mask = np.ones_like(gt['mask']) - gt['mask']

    rel_error = calculate_relative_error_normalized(pred['u'], pred['v'], pred['w'], gt['u'], gt['v'], gt['w'], gt['mask'])

    # -------------Qualitative evaluation----------------


    # 1. Qualitative visalisation of the LR, HR and prediction

    if show_img_plot:
        print("Plot example time frames..")
        
        frames = [32, 33, 34, 35, 36, 37]
        idx_cube = np.index_exp[frames[0]:frames[-1]+1, 20, 0:40, 20:60]
        idx_cube_lr = np.index_exp[frames[0]//2:frames[-1]//2+1, 20, 0:40, 20:60]

        input_lst = []
        input_name =[]
        if load_interpolation_files:
            # input_lst = [interpolate_linear[idx_cube], interpolate_cubic[idx_cube]]
            # input_name = ['linear', 'cubic']
            # input_lst_ = [interpolate_sinc[idx_cube]]
            input_name = ['sinc']

            plot_qual_comparsion(gt['u'][idx_cube],lr['u'][idx_cube_lr],  pred['u'][idx_cube],gt['mask'][idx_cube], np.abs(gt['u'][idx_cube]- pred['u'][idx_cube]), [interpolate_sinc['u'][idx_cube]], ['sinc'], frames,min_v = min_v['u'], max_v = max_v['u'],figsize = (4, 6), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_u_test.png")
            plot_qual_comparsion(gt['v'][idx_cube],lr['v'][idx_cube_lr],  pred['v'][idx_cube],gt['mask'][idx_cube], np.abs(gt['v'][idx_cube]- pred['v'][idx_cube]), [interpolate_sinc['v'][idx_cube]], ['sinc'], frames,min_v = min_v['v'], max_v = max_v['v'],figsize = (4, 6), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_v_test.png")
            plot_qual_comparsion(gt['w'][idx_cube],lr['w'][idx_cube_lr],  pred['w'][idx_cube],gt['mask'][idx_cube], np.abs(gt['w'][idx_cube]- pred['w'][idx_cube]), [interpolate_sinc['w'][idx_cube]], ['sinc'], frames,min_v = min_v['w'], max_v = max_v['w'],figsize = (4, 6), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_w_test.png")
        else:
            plot_qual_comparsion(gt['u'][idx_cube],lr['u'][idx_cube_lr],  pred['u'][idx_cube],gt['mask'][idx_cube], np.abs(gt['u'][idx_cube]- pred['u'][idx_cube]), [], [], frames,min_v = min_v['u'], max_v = max_v['u'],figsize = (7,7), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_u_test.png")
            plot_qual_comparsion(gt['v'][idx_cube],lr['v'][idx_cube_lr],  pred['v'][idx_cube],gt['mask'][idx_cube], np.abs(gt['v'][idx_cube]- pred['v'][idx_cube]), [], [], frames,min_v = min_v['v'], max_v = max_v['v'],figsize = (7,7), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_v_test.png")
            plot_qual_comparsion(gt['w'][idx_cube],lr['w'][idx_cube_lr],  pred['w'][idx_cube],gt['mask'][idx_cube], np.abs(gt['w'][idx_cube]- pred['w'][idx_cube]), [], [], frames,min_v = min_v['w'], max_v = max_v['w'],figsize = (7,7), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_w_test.png")

        plt.show()

    # 2. Plot the relative error and mean speed over time
    if show_RE_plot:

        plt.figure(figsize=(9, 7))

        gt_meanspeed    = calculate_mean_speed(gt['u_fluid'], gt['v_fluid'], gt['w_fluid'], gt['mask'])
        lr_meanspeed    = calculate_mean_speed(lr['u_fluid'], lr['v_fluid'], lr['w_fluid'], lr['mask'])
        pred_meanspeed  = calculate_mean_speed(pred['u_fluid'], pred['v_fluid'], pred['w_fluid'], gt['mask'])

        plt.subplot(3, 1, 1)
        plt.plot(gt_meanspeed, '.-',label ='High resolution', color = 'black')
        plt.plot(pred_meanspeed,'.-', label= '4DFlowNet', color = KI_colors['Blue'])
        plt.plot(range(0, N_frames, 2),  lr_meanspeed,'.-',  label = 'Low resolution', color = KI_colors['Green'])
        if load_interpolation_files:
            sinc_mean = calculate_mean_speed(interpolate_sinc['u_fluid'], interpolate_sinc['v_fluid'], interpolate_sinc['w_fluid'], gt['mask'])
            plt.plot(sinc_mean[:-1], label = 'sinc interpolation', color = 'orange')
            # plt.plot(interpolate_linear['mean_speed'][:-1],'--', label = 'linear interpolation', color = 'pink')
            # plt.plot(interpolate_cubic['mean_speed'][:-1] , label = 'cubic interpoaltion', color = 'forestgreen')
        plt.xlabel("Frame")
        plt.ylabel("Mean speed (cm/s)")
        plt.legend(loc = 'upper left')
        plt.title('Mean speed')
        

        plt.subplot(3, 1, 3)
        #plot_relative_error([gt_filepath],[res_filepath], [set_name])
        N_frames = gt['u'].shape[0]
        #plt.legend(lst_names)
        plt.title("Relative error")
        plt.plot(rel_error, label = '4DFlowNet', color = KI_colors['Blue'])
        if load_interpolation_files:
            re_error_sinc = calculate_relative_error_normalized(interpolate_sinc['u'], interpolate_sinc['v'], interpolate_sinc['w'], gt['u'], gt['v'], gt['w'], gt['mask'])
            plt.plot(re_error_sinc, label = 'sinc interpolation', color = 'orange')
            print('AVG sinc:', np.mean(re_error_sinc))
            # rel_error_lin_interpolation = calculate_relative_error_normalized(interpolate_linear['u'], interpolate_linear['v'], interpolate_linear['w'], gt['u'], gt['v'], gt['w'], gt['mask'])
            # rel_error_cubic_interpolation = calculate_relative_error_normalized(interpolate_cubic['u'], interpolate_cubic['v'], interpolate_cubic['w'], gt['u'], gt['v'], gt['w'], gt['mask'])
            # re_sinc = calculate_relative_error_normalized(interpolate_NN['u'], interpolate_NN['v'], interpolate_NN['w'], gt['u'], gt['v'], gt['w'], gt['mask'])
            # plt.plot(rel_error_lin_interpolation[:-1], label = 'linear interpolation',color = KI_colors['Green'])
            # plt.plot(rel_error_cubic_interpolation, label = 'cubic interpolation', color = 'forestgreen')
            # plt.plot(re_sinc, label = 'sinc interpolation', color = 'orange')
        # plt.plot(50*np.ones(len(rel_error)), 'k:')
        plt.xlabel("Frame")
        plt.ylabel("Relative error (%)")
        plt.ylim((0, 100))
        plt.legend(loc = 'upper left')

        plt.subplot(3, 1, 2)
        plt.plot(calculate_rmse(pred['u'], gt['u'], gt['mask']), label = r'$V_x$ fluid region', color = KI_colors['Grey'])
        plt.plot(calculate_rmse(pred['v'], gt['v'], gt['mask']), label = r'$V_y$ fluid region', color = KI_colors['LightBlue'])
        plt.plot(calculate_rmse(pred['w'], gt['w'], gt['mask']), label = r'$V_z$ fluid region', color = KI_colors['Plum'])
        if load_interpolation_files:
            plt.plot(calculate_rmse(interpolate_sinc['u'], gt['u'], gt['mask']), label = r'$V_x$sinc', color = KI_colors['Grey'])
            # plt.plot(calculate_rmse(pred['speed'], gt['speed'], gt['mask']), label = 'speed', color = KI_colors['LightGrey'])
            # plt.plot(calculate_rmse(interpolate_linear['w'], gt['w'], gt['mask']), label = r'$V_z$ linear interpolation', color = KI_colors['Green'])
        plt.ylabel('RMSE')
        plt.xlabel('Frame')
        plt.title('RMSE ')
        plt.legend(loc = 'upper left')

        plt.plot(calculate_rmse(pred['speed'], gt['speed'], gt['mask']), label = 'speed')
        plt.plot(calculate_rmse(pred['u'], gt['u'], reverse_mask), label = r'$V_x$ non-fluid region',linestyle = '--',  color = KI_colors['Grey'])
        plt.plot(calculate_rmse(pred['v'], gt['v'], reverse_mask), label = r'$V_y$ non-fluid region',linestyle = '--',  color = KI_colors['LightBlue'])
        plt.plot(calculate_rmse(pred['w'], gt['w'], reverse_mask), label = r'$V_z$ non-fluid region',linestyle = '--',  color = KI_colors['Plum'])
        # if load_interpolation_files:
        #     plt.plot(calculate_rmse(interpolate_linear['w'], gt['w'], reverse_mask), label = r'$V_z$ non-fluid region linear interpolation',linestyle = '--',  color = KI_colors['Green'])
        plt.ylabel('RMSE')
        plt.xlabel('Frame')
        plt.title('RMSE')
        plt.legend(loc = 'upper left')

        plt.tight_layout()
        plt.savefig(f'{eval_dir}/{set_name}_M{data_model}_RE_RMSE_MEANSPEED_pred2.svg',bbox_inches='tight')
        plt.show()
        plt.clf()

    # 3. Plot the correlation between the prediction and the ground truth in peak flow frame

    if show_corr_plot:
        print("Plot linear regression plot between prediction and ground truth in peak flow frame..")

        T_peak_flow = np.unravel_index(np.argmax(gt["u"]), shape =gt["u"].shape)[0]
        print("Peak flow frame for model", model_name, T_peak_flow)
        if T_peak_flow%ups_factor == 0: 
            T_peak_flow +=1
            print('Increase peak flow frame to next frame', T_peak_flow)

        plot_correlation(gt, pred, boundary_mask, frame_idx = T_peak_flow, save_as=f'{eval_dir}/{set_name}_M{data_model}_correlation_pred_frame{T_peak_flow}')
        plt.show()
        plt.clf()

        plot_correlation_nobounds(gt, pred,frame_idx=T_peak_flow,show_text = True,  save_as=f'{eval_dir}/{set_name}_M{model_name}_correlation_pred_nobounds_frame{T_peak_flow}')
        plt.show()
        plt.clf()

        # 4. Plot slope and R2 values for core, boundary and all voxels over time

        plot_k_r2_vals(gt, pred, boundary_mask, T_peak_flow,color_b = KI_colors['Plum'] , save_as= f'{eval_dir}/{set_name}_M{model_name}_k_r2_vals_frame{T_peak_flow}_pred')
        plt.show()

    # ------------Tabular evaluations----------------
    
    # calculate k, r2 and RMSE, RE values for all frames
    if tabular_eval:

        #calculate error for each velocity component
        vel_and_speed_colnames = vel_colnames + ['speed']
        k = defaultdict(list)
        r2 = defaultdict(list)
        df_raw = pd.DataFrame()
        df_summary = pd.DataFrame(index=vel_and_speed_colnames)
        for vel in vel_and_speed_colnames:
            print(f'------------------Calculate error for {vel}---------------------')
            # rmse
            rmse_pred = calculate_rmse(pred[vel], gt[vel], gt["mask"], return_std=False)
            rmse_pred_nonfluid = calculate_rmse(pred[vel], gt[vel], reverse_mask)

            plt.plot(rmse_pred, label = f'prediction {vel}')
            if load_interpolation_files:
                rmse_lin_interpolation   = calculate_rmse(interpolate_linear[vel], gt[vel], gt["mask"])
                rmse_cubic_interpolation = calculate_rmse(interpolate_cubic[vel], gt[vel], gt["mask"])
                rmse_lin_interpolation_nonfluid = calculate_rmse(interpolate_linear[vel], gt[vel], reverse_mask)
                rmse_cubic_interpolation_nonfluid = calculate_rmse(interpolate_cubic[vel], gt[vel], reverse_mask)

                plt.plot(rmse_cubic_interpolation, label = 'cubic interpolation')
                plt.plot(rmse_lin_interpolation, label = 'linear interpolation')
            
            # absolute error
            abs_diff = np.mean(np.abs(pred[vel] - gt[vel]), where = bool_mask, axis = (1,2,3))

            # k and R2 values
            for t in range(gt["u"].shape[0]):
                k_core, r2_core     = calculate_k_R2( pred[vel][t], gt[vel][t], core_mask[t])
                k_bounds, r2_bounds = calculate_k_R2( pred[vel][t], gt[vel][t], boundary_mask[t])
                k_all, r2_all       = calculate_k_R2( pred[vel][t], gt[vel][t], gt['mask'][t])
                k[f'k_core_{vel}'].append(k_core)
                r2[f'r2_core_{vel}'].append(r2_core)
                k[f'k_bounds_{vel}'].append(k_bounds)
                r2[f'r2_bounds_{vel}'].append(r2_bounds)
                k[f'k_all_{vel}'].append(k_all)
                r2[f'r2_all_{vel}'].append(r2_all)

            # Populate df_raw with the calculated metrics
            df_raw[f'k_core_{vel}'] = k[f'k_core_{vel}']
            df_raw[f'k_bounds_{vel}'] = k[f'k_bounds_{vel}']
            df_raw[f'k_all_{vel}'] = k[f'k_all_{vel}']
            df_raw[f'r2_core_{vel}'] = r2[f'r2_core_{vel}']
            df_raw[f'r2_bounds_{vel}'] = r2[f'r2_bounds_{vel}']
            df_raw[f'r2_all_{vel}'] = r2[f'r2_all_{vel}']
            df_raw[f'rmse_pred_{vel}'] = rmse_pred
            df_raw[f'rmse_pred_nonfluid_{vel}'] = rmse_pred_nonfluid
            df_raw[f'abs_diff_{vel}'] = abs_diff
            
            # Summary statistics for df_summary
            df_summary.loc[vel, 'k_avg_core'] = np.mean(k[f'k_core_{vel}'])
            df_summary.loc[vel, 'k_avg_bounds'] = np.mean(k[f'k_bounds_{vel}'])
            df_summary.loc[vel, 'k_avg_all'] = np.mean(k[f'k_all_{vel}'])
            df_summary.loc[vel, 'k_min_core'] = np.min(k[f'k_core_{vel}'])
            df_summary.loc[vel, 'k_min_bounds'] = np.min(k[f'k_bounds_{vel}'])
            df_summary.loc[vel, 'k_min_all'] = np.min(k[f'k_all_{vel}'])
            df_summary.loc[vel, 'k_max_core'] = np.max(k[f'k_core_{vel}'])
            df_summary.loc[vel, 'k_max_bounds'] = np.max(k[f'k_bounds_{vel}'])
            df_summary.loc[vel, 'k_max_all'] = np.max(k[f'k_all_{vel}'])
            df_summary.loc[vel, 'r2_avg_core'] = np.mean(r2[f'r2_core_{vel}'])
            df_summary.loc[vel, 'r2_avg_bounds'] = np.mean(r2[f'r2_bounds_{vel}'])
            df_summary.loc[vel, 'r2_avg_all'] = np.mean(r2[f'r2_all_{vel}'])
            df_summary.loc[vel, 'r2_min_core'] = np.min(r2[f'r2_core_{vel}'])
            df_summary.loc[vel, 'r2_min_bounds'] = np.min(r2[f'r2_bounds_{vel}'])
            df_summary.loc[vel, 'r2_min_all'] = np.min(r2[f'r2_all_{vel}'])
            df_summary.loc[vel, 'r2_max_core'] = np.max(r2[f'r2_core_{vel}'])
            df_summary.loc[vel, 'r2_max_bounds'] = np.max(r2[f'r2_bounds_{vel}'])
            df_summary.loc[vel, 'r2_max_all'] = np.max(r2[f'r2_all_{vel}'])
            df_summary.loc[vel, 'rmse_avg'] = np.mean(rmse_pred)
            df_summary.loc[vel, 'rmse_avg_nonfluid'] = np.mean(rmse_pred_nonfluid)
            df_summary.loc[vel, 'abs_diff_avg'] = np.mean(abs_diff)

        # Add relative error to df_raw and df_summary
        df_raw[f'RE'] = rel_error
        df_summary[f'RE_avg'] = np.mean(rel_error)

        # Save dataframes to CSV
        df_raw.to_csv(f'{eval_dir}/{set_name}_M{model_name}_k_r2_RE_ALL.csv', float_format="%.3f")
        df_summary.to_csv(f'{eval_dir}/{set_name}_M{model_name}_k_r2_RE_summary.csv', float_format="%.3f")

        print(df_summary)
        print(df_summary.to_latex(index=False, float_format="%.2f"))
        print(df_raw.to_latex(index=False, float_format="%.2f"))


        plt.legend()
        plt.show()
    print('--------------------------------------')
    # relative error
    REL_error_pred = rel_error
    if load_interpolation_files:
        REL_error_lin_interpolation = rel_error_lin_interpolation
        REL_error_cubic_interpolation = rel_error_cubic_interpolation

    exit()


    diastole_end = 25

    boundary_mask, core_mask = get_boundaries(gt["mask"])
    bounds_boolmask = boundary_mask.astype(bool)
    core_boolmask = core_mask.astype(bool)
    # print(f'avg bound velocities {np.mean(np.multiply(gt["u"], bounds), axis = (1, 2, 3))} {np.mean(np.multiply(gt["v"], bounds), axis = (1, 2, 3))} {np.mean(np.multiply(gt["w"], bounds), axis = (1, 2, 3))}')
    # print(f'avg core velocities {np.mean(np.multiply(gt["u"], core_mask), axis = (1,2,3))} {np.mean(np.multiply(gt["v"], core_mask), axis = (1, 2, 3))} {np.mean(np.multiply(gt["w"], core_mask), axis = (1, 2, 3))}')

    max_b = np.max(np.mean(gt['speed'], axis = (1, 2, 3), where =bounds_boolmask))
    min_b = np.min(np.mean(gt['speed'], axis = (1, 2, 3), where =bounds_boolmask))
    print(f'speed min and max boundary {min_b}  {max_b}')
    max_c = np.max(np.mean(gt['speed'], axis = (1, 2, 3), where =core_boolmask))
    min_c = np.min(np.mean(gt['speed'], axis = (1, 2, 3), where =core_boolmask))
    print(f'speed min and max core {min_c}  {max_c}')
    # print(f'speed min and max boundary {np.min(np.mean(gt['speed'], axis = (1, 2, 3), where =bounds_boolmask))}  {np.max(np.mean(gt['speed'], axis = (1, 2, 3), where =bounds_boolmask))}')
    # print(f'speed min and max core {np.min(np.mean(gt['speed'], axis = (1, 2, 3), where =core_boolmask))}  {np.min(np.mean(gt['speed'], axis = (1, 2, 3), where =core_boolmask))}')
    
    # for t in range(0, gt["u"].shape[0]):
    #     total_voxels = np.sum(gt["mask"][t])
    #     print(f'{t} - BoundaryVoxels vs core  {np.sum(bounds[t])/total_voxels:.4f} core {np.sum(core_mask[t])/total_voxels:.4f}' )
    
    #calculate error for each velocity component
    vel_and_speed_colnames = vel_colnames + ['speed']
    k = defaultdict(list)
    r2 = defaultdict(list)
    for vel in vel_and_speed_colnames:
        print(f'------------------Calculate error for {vel}---------------------')
        # rmse
        rmse_pred= calculate_rmse(pred[vel], gt[vel], gt["mask"], return_std=False)
        if load_interpolation_files:
            rmse_lin_interpolation = calculate_rmse(interpolate_linear[vel], gt[vel], gt["mask"])
            rmse_cubic_interpolation = calculate_rmse(interpolate_cubic[vel], gt[vel], gt["mask"])
        bool_mask = gt['mask'].astype(bool)
        reverse_mask = np.ones(gt["mask"].shape) - gt["mask"]
        
        rmse_pred_nonfluid = calculate_rmse(pred[vel], gt[vel], reverse_mask)
        if load_interpolation_files:
            rmse_lin_interpolation_nonfluid = calculate_rmse(interpolate_linear[vel], gt[vel], reverse_mask)
            rmse_cubic_interpolation_nonfluid = calculate_rmse(interpolate_cubic[vel], gt[vel], reverse_mask)

        plt.plot(rmse_pred, label = 'prediction')
        if load_interpolation_files:
            plt.plot(rmse_cubic_interpolation, label = 'cubic interpolation')
            plt.plot(rmse_lin_interpolation, label = 'linear interpolation')
        

        for t in range(gt["u"].shape[0]):
            k_core, r2_core  = calculate_k_R2( pred[vel][t], gt[vel][t], core_mask[t])
            k_bounds, r2_bounds  = calculate_k_R2( pred[vel][t], gt[vel][t], boundary_mask[t])
            k_all, r2_all  = calculate_k_R2( pred[vel][t], gt[vel][t], gt['mask'][t])
            k[f'k_core_{vel}'].append(k_core)
            r2[f'r2_core_{vel}'].append(r2_core)
            k[f'k_bounds_{vel}'].append(k_bounds)
            r2[f'r2_bounds_{vel}'].append(r2_bounds)
            k[f'k_all_{vel}'].append(k_all)
            r2[f'r2_all_{vel}'].append(r2_all)

        print(f'AVG k x core ', np.mean(k[f'k_core_{vel}']) , ' - bounds:', np.mean(k[f'k_bounds_{vel}']), ' - all: ', np.mean(k[f'k_all_{vel}']))
        print(f'MIN k x core ', np.min(k[f'k_core_{vel}']) , '- bounds: ', np.min(k[f'k_bounds_{vel}']) , ' - all: ', np.min(k[f'k_all_{vel}']))
        print(f'MAX k x core ', np.max(k[f'k_core_{vel}']) , '- bounds: ', np.max(k[f'k_bounds_{vel}']) , ' - all: ', np.max(k[f'k_all_{vel}']))
        print(f'AVG r2 x core ', np.mean(r2[f'r2_core_{vel}']) , ' - bounds: ', np.mean(r2[f'r2_bounds_{vel}']), ' - all: ', np.mean(r2[f'r2_all_{vel}']))
        print(f'MIN r2 x core ', np.min(r2[f'r2_core_{vel}']) , '- bounds: ', np.min(r2[f'r2_bounds_{vel}']) , ' - all: ', np.min(r2[f'r2_all_{vel}']))
        print(f'MAX r2 x core ', np.max(r2[f'r2_core_{vel}']) , '- bounds: ', np.max(r2[f'r2_bounds_{vel}']) , ' - all: ', np.max(r2[f'r2_all_{vel}']))
        
        
        print(f'AVG RMSE ALL fluid {np.mean(rmse_pred):.4f}')
        print(f'AVG RMSE ALL nonfluid {np.mean(rmse_pred_nonfluid):.4f}')

        # print(f'AVG RMSE prediction diastole: {np.mean(rmse_pred[:25]):.4f} - linear interpolation: {np.mean(rmse_lin_interpolation[:25]):.4f} - cubic interpolation: {np.mean(rmse_cubic_interpolation[:25]):.4f}')
        # print(f'STD RMSE prediction diastole:  {np.std(rmse_pred[:25]):.4f} - linear interpolation: {np.std(rmse_lin_interpolation[:25]):.4f} - cubic interpolation: {np.std(rmse_cubic_interpolation[:25]):.4f}')
        # print(f'AVG RMSE prediction systole: {np.mean(rmse_pred[25:]):.4f} - linear interpolation: {np.mean(rmse_lin_interpolation[25:]):.4f} - cubic interpolation: {np.mean(rmse_cubic_interpolation[25:]):.4f}')
        # print(f'STD RMSE prediction systole:  {np.std(rmse_pred[25:]):.4f} - linear interpolation: {np.std(rmse_lin_interpolation[25:]):.4f} - cubic interpolation: {np.std(rmse_cubic_interpolation[25:]):.4f}')
        # #abs difference
        # abs_diff = np.abs(gt[f'{vel}_fluid'] - pred[f'{vel}_fluid'])
        # if load_interpolation_files:
        #     abs_diff_lin_interpolation = np.abs(gt[f'{vel}_fluid'] - interpolate_linear[f'{vel}_fluid'])
        #     abs_diff_cubic_interpolation = np.abs(gt[f'{vel}_fluid'] - interpolate_cubic[f'{vel}_fluid'])

        # print(f'Max abs difference diastole - prediction {np.max(abs_diff[:diastole_end]):.4f}')
        # print(f'Max abs difference diastole - linear interpolation', np.max(abs_diff_lin_interpolation[:diastole_end]))
        # print(f'Max abs difference diastole - cubic interpolation', np.max(abs_diff_cubic_interpolation[:diastole_end]))
        # print(f'Max abs difference systole - prediction', np.max(abs_diff[diastole_end:]))	
        # print(f'Max abs difference systole - linear interpolation', np.max(abs_diff_lin_interpolation[diastole_end:]))
        # print(f'Max abs difference systole - cubic interpolation', np.max(abs_diff_cubic_interpolation[diastole_end:]))

        # print(f'Max abs difference fluid - prediction {np.max(abs_diff):.4f}')
        # print(f'Max abs difference fluid - linear interpolation', np.max(abs_diff_lin_interpolation))
        # print(f'Max abs difference fluid - cubic interpolation', np.max(abs_diff_cubic_interpolation))

        # # print(f'Max lin interpolate boundary frame 35', np.max(np.multiply(interpolate_linear[vel][35], bounds[35])))
        # # print(f'Max cubic interpolate boundary frame 35', np.max(np.multiply(interpolate_linear[vel][35], bounds[35])))

        # print(f'Correlation frame 35 prediction- {np.mean(abs_diff[35], where= bool_mask[35]):.4f} std: {np.std(abs_diff[35], where= bool_mask[35]):.4f}')
        # print(f'Correlation frame 35 linear interpolation- {np.mean(abs_diff_lin_interpolation[35], where= bool_mask[35]):.4f} std: {np.std(abs_diff_lin_interpolation[35], where= bool_mask[35]):.4f}')
        # print(f'Correlation frame 35 cubic interpolation- {np.mean(abs_diff_cubic_interpolation[35], where= bool_mask[35]):.4f} std: {np.std(abs_diff_cubic_interpolation[35], where= bool_mask[35]):.4f}')
    
    plt.legend()
    plt.show()
    print('--------------------------------------')
    # relative error
    REL_error_pred = rel_error
    if load_interpolation_files:
        REL_error_lin_interpolation = rel_error_lin_interpolation
        REL_error_cubic_interpolation = rel_error_cubic_interpolation
    # print(f'RMSE {vel} last frame - pred - {rmse_pred[-1]:.4f} - linear interpolation: {rmse_lin_interpolation[-1]:.4f} - cubic interpolation: {rmse_cubic_interpolation[-1]:.4f}')
    # print(f'RMSE {vel} peak frame 35 - pred - {rmse_pred[35]:.4f} - linear interpolation: {rmse_lin_interpolation[35]:.4f} - cubic interpolation: {rmse_cubic_interpolation[35]:.4f}')
    # print(f'RMSE {vel} peak frame 34- pred - {rmse_pred[34]:.4f} - linear interpolation: {rmse_lin_interpolation[34]:.4f} - cubic interpolation: {rmse_cubic_interpolation[34]:.4f}')


    # print(f'AVG REL error prediction {np.mean(REL_error_pred):.1f}')
    # print(f'Last frame REL error prediction {REL_error_pred[-1]:.1f}')
    # print(f'AVG REL error prediction diastole: {np.mean(REL_error_pred[:25]):.1f} - linear interpolation: {np.mean(REL_error_lin_interpolation[:25]):.1f} - cubic interpolation: {np.mean(REL_error_cubic_interpolation[:25]):.1f}')
    # print(f'STD REL error prediction diastole:  {np.std(REL_error_pred[:25]):.1f} - linear interpolation: {np.std(REL_error_lin_interpolation[:25]):.1f} - cubic interpolation: {np.std(REL_error_cubic_interpolation[:25]):.1f}')
    # print(f'AVG REL error prediction systole:  {np.mean(REL_error_pred[25:]):.1f} - linear interpolation: {np.mean(REL_error_lin_interpolation[25:]):.1f} - cubic interpolation: {np.mean(REL_error_cubic_interpolation[25:]):.1f}')
    # print(f'STD REL error prediction systole:   {np.std(REL_error_pred[25:]):.1f} - linear interpolation: {np.std(REL_error_lin_interpolation[25:]):.1f} - cubic interpolation: {np.std(REL_error_cubic_interpolation[25:]):.1f}')

    # print('Max mean speed deviation', np.max(np.abs(mean_speed_gt - mean_speed_pred)))
    # print('mean speed deviation', np.mean(np.abs(mean_speed_gt - mean_speed_pred)))
    # print('Max mean speed deviation linear interpolation', np.max(np.abs(mean_speed_gt - mean_speed_lin_interpolation)))
    # print('Max mean speed deviation cubic interpolation', np.max(np.abs(mean_speed_gt - mean_speed_cubic_interpolation)))
    # print('Min mean speed deviation', np.min(mean_speed_gt - mean_speed_pred))
    # print('Max mean speed deviation', np.max(mean_speed_gt - mean_speed_pred))
    # print('Count number of frames, where difference between mean speed gt and pred is smaller than 0', np.sum(mean_speed_gt - mean_speed_pred < 0))



    # show_temporal_development_line(gt["u"], interpolate_linear["u"], pred["u"],gt["mask"], axis=3, indices=(20,20), save_as=f'{eval_dir}/{set_name}_temporal_development.png')
    plt.clf()
    #evaluate where the higest shift (temporal derivative) is for each frame
    #show_quiver(gt["u"][4, :, :, :], gt["v"][4, :, :, :], gt["w"][4, :, :, :],gt["mask"], save_as=f'{result_dir}/plots/test_quiver.png')

    
    plt.clf()
    plot_correlation(gt, pred, boundary_mask, frame_idx = T_peak_flow, save_as=f'{pred_dir}/plots/{set_name}_M{data_model}_correlation_pred_frame{T_peak_flow}')
    plt.clf()

    # plot_comparison_temporal(lr, gt, pred, frame_idx = 8, axis=1, slice_idx = 40, save_as=f'{eval_dir}/{set_name}_M{data_model}_visualize_interporalion_comparison.png')
    # plt.clf()

    print("Plot relative error and mean speed")
    plt.subplot(2, 1, 1)
    plt.title("Relative error")
    plt.plot(rel_error, label = 'averaged prediction')
    if load_interpolation_files:
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
    plt.plot(mean_speed_pred, label= set_name, color = 'steelblue')
    if load_interpolation_files:
        plt.plot(mean_speed_lin_interpolation[:-1], label = 'linear interpolation', color = 'yellowgreen')
        plt.plot(mean_speed_cubic_interpolation[:-1], label = 'cubic interpoaltion', color = 'forestgreen')
    plt.xlabel("Frame")
    plt.ylabel("Mean speed (cm/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{eval_dir}/{set_name}_M{data_model}_pred_RE_mean_speed.svg')
    plt.show()

            

