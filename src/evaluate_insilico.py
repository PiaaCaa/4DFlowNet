import numpy as np
import time
import os
from matplotlib import pyplot as plt
import h5py
from collections import defaultdict

from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.evaluate_utils import *

from utils.vtkwriter_per_dir import uvw_mask_to_vtk


plt.rcParams['figure.figsize'] = [10, 8]


def load_vel_data(gt_filepath, lr_filepath, pred_filepath,  vel_colnames = ['u', 'v', 'w'],res_colnames = ['u_combined', 'v_combined', 'w_combined'], threshold = 0.5, offset = 0, factor = 2):
    

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

                    print('Shapes', gt[vel].shape, pred[vel].shape, lr[vel].shape, gt['mask'].shape)
                    # take away background outside mask
                    pred[f'{vel}_fluid'] =np.multiply(pred[vel], gt["mask"])
                    lr[f'{vel}_fluid'] =  np.multiply(lr[vel], lr['mask'])
                    gt[f'{vel}_fluid'] =  np.multiply(gt[vel], gt["mask"])

                    # Check that shapes match
                    assert gt[vel].shape == pred[vel].shape, f"Shape mismatch HR/SR: {gt[vel].shape} != {pred[vel].shape}"
                    
                # include speed calculations
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
    nn_name = '20240709-2057'#20240617-0933
    set_name = 'Test'               
    data_model= '4'
    step = 2
    load_interpolation_files = True
    ups_factor = 2

    #choose which plots to show
    show_img_plot = False
    show_RE_plot = False
    show_corr_plot = False
    show_planeMV_plot = True
    show_planeAV_plot = True
    tabular_eval = False
    show_animation = False 
    save_as_vti = False


    vel_colnames=['u', 'v', 'w']

    # directories
    data_dir = 'data/CARDIAC'
    pred_dir = f'results/Temporal4DFlowNet_{nn_name}'
    eval_dir = f'{pred_dir}/plots'

    # filenames
    gt_filename = f'M{data_model}_2mm_step{step}_cs_invivoP02_hr.h5'
    lr_filename = f'M{data_model}_2mm_step{step}_cs_invivoP02_lr.h5'
    pred_filename = f'{set_name}set_result_model{data_model}_2mm_step{step}_{nn_name[-4::]}_temporal.h5'
    
    # Setting up
    gt_filepath   = '{}/{}'.format(data_dir, gt_filename)
    pred_filepath = '{}/{}'.format(pred_dir, pred_filename)
    lr_filepath   = '{}/{}'.format(data_dir, lr_filename)

    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    # ----------Load data and interpolation files and calculate visualization params----------------

    gt, lr, pred = load_vel_data(gt_filepath, lr_filepath, pred_filepath, vel_colnames = vel_colnames)

    N_frames = gt['u'].shape[0]

    if load_interpolation_files: 
        # interpolate_linear, interpolate_cubic, interpolate_NN = load_interpolation(data_model, step,lr, gt)
        interpolate_linear = {}
        interpolate_linear['u'] = temporal_linear_interpolation_np(lr['u'], gt['u'].shape)
        interpolate_linear['v'] = temporal_linear_interpolation_np(lr['v'], gt['v'].shape)
        interpolate_linear['w'] = temporal_linear_interpolation_np(lr['w'], gt['w'].shape)

        # load sinc interpolation
        interpolate_sinc = {}
        hr_range = np.linspace(0,1,  gt['u'].shape[0])
        lr_range = hr_range[::2] # downsamplie like this to get exact same evaluation points

        for vel in vel_colnames:
            interpolate_sinc[vel] = temporal_sinc_interpolation_ndarray(lr[vel], lr_range, hr_range)
            interpolate_sinc[f'{vel}_fluid'] = np.multiply(interpolate_sinc[vel], gt['mask'])

    # check that dimension fits
    assert(gt["u"].shape == pred["u"].shape)  ,str(pred["u"].shape) + str(gt["u"].shape) # dimensions of HR and SR need to be the same
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

    # caluclation for further plotting

    # Relative error calculation
    if show_RE_plot or tabular_eval:
        rel_error = calculate_relative_error_normalized(pred['u'], pred['v'], pred['w'], gt['u'], gt['v'], gt['w'], gt['mask'])

    # -------------Qualitative evaluation----------------


    # 1. Qualitative visalisation of the LR, HR and prediction

    if show_img_plot:
        print("Plot example time frames..")
        
        frames = [32, 33, 34, 35]
        idx_cube = np.index_exp[frames[0]:frames[-1]+1, 22, 0:40, 20:60]
        idx_cube_lr = np.index_exp[frames[0]//2:frames[-1]//2+1, 22, 0:40, 20:60]

        input_lst = []
        input_name =[]
        if load_interpolation_files:
            # input_lst = [interpolate_linear[idx_cube], interpolate_cubic[idx_cube]]
            # input_name = ['linear', 'cubic']
            # input_lst_ = [interpolate_sinc[idx_cube]]
            # input_name = ['sinc']

            plot_qual_comparsion(gt['u'][idx_cube],lr['u'][idx_cube_lr],  pred['u'][idx_cube],gt['mask'][idx_cube], np.abs(gt['u'][idx_cube]- pred['u'][idx_cube]), [interpolate_sinc['u'][idx_cube]], ['sinc'], frames,min_v = None, max_v = None,figsize = (8, 6.5), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_u_test.png")
            plot_qual_comparsion(gt['v'][idx_cube],lr['v'][idx_cube_lr],  pred['v'][idx_cube],gt['mask'][idx_cube], np.abs(gt['v'][idx_cube]- pred['v'][idx_cube]), [interpolate_sinc['v'][idx_cube]], ['sinc'], frames,min_v = None, max_v = None,figsize = (8, 6.5), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_v_test.png")
            plot_qual_comparsion(gt['w'][idx_cube],lr['w'][idx_cube_lr],  pred['w'][idx_cube],gt['mask'][idx_cube], np.abs(gt['w'][idx_cube]- pred['w'][idx_cube]), [interpolate_sinc['w'][idx_cube]], ['sinc'], frames,min_v = None, max_v = None,figsize = (8, 6.5), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_w_test.png")
        else:
            plot_qual_comparsion(gt['u'][idx_cube],lr['u'][idx_cube_lr],  pred['u'][idx_cube],gt['mask'][idx_cube], np.abs(gt['u'][idx_cube]- pred['u'][idx_cube]), [], [], frames,min_v = min_v['u'], max_v = max_v['u'],figsize = (8,5), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_u_test.png")
            plot_qual_comparsion(gt['v'][idx_cube],lr['v'][idx_cube_lr],  pred['v'][idx_cube],gt['mask'][idx_cube], np.abs(gt['v'][idx_cube]- pred['v'][idx_cube]), [], [], frames,min_v = min_v['v'], max_v = max_v['v'],figsize = (8,5), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_v_test.png")
            plot_qual_comparsion(gt['w'][idx_cube],lr['w'][idx_cube_lr],  pred['w'][idx_cube],gt['mask'][idx_cube], np.abs(gt['w'][idx_cube]- pred['w'][idx_cube]), [], [], frames,min_v = min_v['w'], max_v = max_v['w'],figsize = (8,5), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_w_test.png")

        plt.show()

    # 2. Plot the relative error and mean speed over time

    if show_RE_plot:

        fig_re = plot_relative_error(gt, pred, [interpolate_linear, interpolate_sinc], ['linear', 'sinc'], 
                                     f'{eval_dir}/{set_name}_M{data_model}_RE_pred2.svg', figsize = (10, 5))
        plt.show()
        plt.close()

        # plot RMSE
        fig_rmse = plot_rmse(gt, pred, [interpolate_linear, interpolate_sinc], ['linear', 'sinc'],f'{eval_dir}/{set_name}_M{data_model}_RMSE.svg',colors_comp = None,  figsize = (10, 5))
        plt.show()

        if True:
            
            fig_ms = plot_mean_speed(gt, pred, lr, [], [],save_as = f'{eval_dir}/{set_name}_M{nn_name}_meanspeed.png',colors_comp = None,  figsize= (10, 5))
            plt.show()

            # Merge the two figures into a single figure
            fig = plt.figure(figsize=(10, 15))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            ax1.axis('off')  # Turn off the axes for the first subplot
            ax2.axis('off')  # Turn off the axes for the second subplot
            ax3.axis('off')
            fig.subplots_adjust(wspace=0)  # Adjust the spacing between subplots
            fig_re.subplots_adjust(wspace=0)  # Adjust the spacing between subplots in the first figure
            fig_rmse.subplots_adjust(wspace=0)  # Adjust the spacing between subplots in the second figure
            ax1.imshow(fig_ms.canvas.renderer._renderer)
            ax2.imshow(fig_rmse.canvas.renderer._renderer)
            ax3.imshow(fig_re.canvas.renderer._renderer)
            plt.tight_layout()
            plt.show()
            # plt.savefig(f'{eval_dir}/{set_name}_M{model_name}.png')


    # 3. Plot the correlation between the prediction and the ground truth in peak flow frame

    if show_corr_plot:
        print("Plot linear regression plot between prediction and ground truth in peak flow frame..")

        T_peak_flow = np.unravel_index(np.argmax(gt["u"]), shape =gt["u"].shape)[0]
        print("Peak flow frame for model", set_name, T_peak_flow)
        if T_peak_flow%ups_factor == 0: 
            T_peak_flow +=1
            print('Increase peak flow frame to next frame', T_peak_flow)

        # # 4. Plot slope and R2 values for core, boundary and all voxels over time
        k_SR, r2_SR = calculate_and_plot_k_r2_vals_nobounds(gt, pred,gt['mask'], T_peak_flow,figsize=(15, 5), save_as = f'{eval_dir}/{set_name}_M{data_model}_corr_k_r2_vals_nobounds_frame{T_peak_flow}_pred')

        fig3, axs3 = plot_k_r2_vals_nobounds(k_SR, r2_SR, T_peak_flow, figsize = (12, 4),exclude_tbounds = True,  save_as= f'{eval_dir}/{set_name}_M{data_model}_corr_k_r2_vals_nobounds_frame{T_peak_flow}_pred')
        fig1 = plot_correlation_nobounds(gt, pred, T_peak_flow, show_text = True, save_as = f'{eval_dir}/{set_name}_M{data_model}_correlation_pred_nobounds_frame{T_peak_flow}')
        
        # Merge the two figures into a single figure
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.axis('off')  
        ax2.axis('off')  
        
        fig.subplots_adjust(wspace=0)   
        fig1.subplots_adjust(wspace=0)  
        fig3.subplots_adjust(wspace=0) 
        ax1.imshow(fig1.canvas.renderer._renderer)
        ax2.imshow(fig3.canvas.renderer._renderer)
        plt.tight_layout()
        plt.savefig(f'{eval_dir}/{set_name}_M{data_model}_correlation_frame{T_peak_flow}_K_R2.png')
        
        plt.show()
        # fig_sinc = plot_correlation_nobounds(gt, interpolate_sinc, T_peak_flow,show_text = True,color_points= KTH_colors['green100']
        #                                      ,  save_as = f'{eval_dir}/{set_name}_M{model_name}_correlation_sinc_nobounds_frame{T_peak_flow}')
        # fig_linear = plot_correlation_nobounds(gt, interpolate_linear, T_peak_flow,show_text = True,color_points= 'forestgreen', 
                                                # save_as = f'{eval_dir}/{set_name}_M{model_name}_correlation_linear_nobounds_frame{T_peak_flow}')

        
        show_corr_plot_with_interpolation = True
        if load_interpolation_files and show_corr_plot_with_interpolation:
            k_linear = np.zeros((3, N_frames))
            k_sinc = np.zeros((3, N_frames))
            r2_linear = np.zeros((3, N_frames))
            r2_sinc = np.zeros((3, N_frames))
            for i, vel in enumerate(vel_colnames):
                k_linear[i, :], r2_linear[i, :] = calculate_k_R2_timeseries(interpolate_linear[vel], gt[vel], gt['mask'])
                k_sinc[i, :], r2_sinc[i, :] = calculate_k_R2_timeseries(interpolate_sinc[vel], gt[vel], gt['mask'])

            vel_colnames = ['u', 'v', 'w']
            vel_plotname = [r'$V_x$', r'$V_y$', r'$V_z$']
            fontsize = 16
            figsize = (10, 7)
            frames = k_SR.shape[1]

            t_range = range(frames)

            min_val_k = np.minimum(0.65, np.min(k_SR))
            max_val_k = np.maximum(1.05, np.max(k_SR))
            min_val_r2 = np.minimum(0.65, np.min(r2_SR))
            max_val_r2 = np.maximum(1.05, np.max(r2_SR))

            # Create subplots
            # plt.subplots_adjust(wspace=0.3)
            fig1, axs = plt.subplots(2, 3, figsize=figsize) #, sharey=True)

            for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
                # Plot k values on the primary y-axis
                
                axs[0, i].set_ylim([min_val_k, max_val_k])
                axs[0, i].set_title(title, fontsize = fontsize)
                axs[0, i].set_xlabel('frame', fontsize=fontsize)
                axs[0, i].set_ylabel(r'k', fontsize=fontsize)
                axs[0, i].locator_params(axis='y', nbins=3)
                axs[0, i].locator_params(axis='x', nbins=3)
                axs[0, i].tick_params(axis='y', labelsize = fontsize)
                axs[0, i].tick_params(axis='x', labelsize = fontsize)
                axs[0, i].scatter(np.ones(1)*T_peak_flow, [k_SR[i, T_peak_flow]], label='peak flow frame', color=KI_colors['Grey'])
                axs[0, i].plot(np.ones(frames), 'k:', label= 'ones')
                axs[0, i].legend(loc='lower right')
                
                # k-values
                axs[0, i].plot(t_range, k_SR[i, :], label='k SR', color='black')
                axs[0, i].plot(t_range, k_linear[i, :], label='k linear', color='lightgreen')
                axs[0, i].plot(t_range, k_sinc[i, :], label='k sinc', color='forestgreen')
                axs[0, i].scatter(np.ones(1)*T_peak_flow, [k_SR[i, T_peak_flow]], color=KI_colors['Grey'])

                axs[1, i].set_ylim([min_val_r2, max_val_r2])
                axs[1, i].set_title(title, fontsize = fontsize)
                axs[1, i].set_xlabel('frame', fontsize=fontsize)
                axs[1, i].set_ylabel(r'$R^2$', fontsize=fontsize)
                axs[1, i].locator_params(axis='y', nbins=3)
                axs[1, i].locator_params(axis='x', nbins=3)
                axs[1, i].tick_params(axis='y', labelsize = fontsize)
                axs[1, i].tick_params(axis='x', labelsize = fontsize)

                # R2 values 
                axs[1, i].plot(t_range, r2_SR[i, :], '--', label=r'$R^2$', color='black')
                axs[1, i].plot(t_range, r2_linear[i, :], '--', label=r'$R^2$ linear', color='lightgreen')
                axs[1, i].plot(t_range, r2_sinc[i, :], '--', label=r'$R^2$ sinc', color='forestgreen')
                axs[1, i].scatter(np.ones(1)*T_peak_flow, [r2_SR[i, T_peak_flow]], label='peak flow frame', color=KI_colors['Grey'])
                axs[1, i].plot(np.ones(frames), 'k:', label= 'ones')
                axs[1, i].legend(loc='lower right')

            plt.tight_layout()
            plt.savefig(f'{eval_dir}/{set_name}_M{data_model}_K_R2_vals_SR_lin_sinc_VXYZ.png')
            plt.show()


    # 4. Plot MV plot through Mitral valve plane
    if show_planeMV_plot:
        print("Plot MV plane flow..")

        # define plane 
        t = 0
        plane_points = [51/2, 56/2, 72/2]
        plane_normal = [0.18, 0.47, -0.86]
        order_normal = [2, 1, 0]
        plane_normal /= np.linalg.norm(plane_normal)

        # calculate the plane
        d = -np.dot(plane_points, plane_normal)
        xx, yy = np.meshgrid(np.arange(0, gt['u'].shape[1]), np.arange(0, gt['u'].shape[2]))
        zz = (-plane_normal[0] * xx - plane_normal[1] * yy - d) * 1. / plane_normal[2]

        zz[np.where(zz < 0)] = 0
        zz[np.where(zz >= gt['u'].shape[3])] = gt['u'].shape[3] - 1
        print(xx.max(), yy.max(), zz.max())
        print(xx.min(), yy.min(), zz.min())

        # Get point coordiantes in plane
        points_in_plane = np.zeros_like(gt['mask'][t])
        points_in_plane[xx.flatten().astype(int), yy.flatten().astype(int), zz.flatten().astype(int)] = 1

        #3D model: is just 1 in region, where plane AND fluid region is
        points_plane_core = points_in_plane.copy()
        points_plane_core[np.where(gt['mask'][t]==0)] = 0

        #Always adjust to different models
        points_MV = points_plane_core.copy()
        points_MV[:, :, :15] = 0
        points_MV[:, :21, :] = 0
        points_MV[:, 36:, :] = 0
        points_MV[38:, :, :] = 0

        #2. Get points in plane and cut out right region

        #get indices
        idx_intersec_plane_fluid = np.where(points_plane_core>0)
        idx_plane                = np.where(points_in_plane>0)
        idx_MV                   = np.where(points_MV>0) 

        img_mask = gt['mask'][t][idx_plane].reshape(xx.shape[1], -1)
        img_MV_mask = points_MV[idx_plane].reshape(xx.shape[1], -1)
        # plt.imshow(gt['mask'][t][idx_plane].reshape(xx.shape[1], -1))
        # plt.imshow(img_MV_mask+img_mask)
        # plt.show()

        lr_vel   = velocity_through_plane(idx_plane, lr, plane_normal, order_normal = order_normal).reshape(lr['u'].shape[0], xx.shape[1], -1)
        hr_vel   = velocity_through_plane(idx_plane, gt, plane_normal, order_normal = order_normal).reshape(gt['u'].shape[0], xx.shape[1], -1)
        pred_vel = velocity_through_plane(idx_plane, pred, plane_normal, order_normal = order_normal).reshape(pred['u'].shape[0], xx.shape[1], -1)

        #-----plot MV 1; Qualitave plot----- 
        idx_crop = np.index_exp[:, 10:37, 15:40]
        idx_crop2 = np.index_exp[10:37, 15:40]

        # crop to important region
        lr_vel_crop = lr_vel[idx_crop]
        hr_vel_crop = hr_vel[idx_crop ]
        pred_vel_crop = pred_vel[idx_crop]
        img_MV_mask_crop = img_MV_mask[idx_crop2]

        timepoints = [6, 7, 8, 9]
        timepoints_lr = [3, 4, 5, 6]
        plot_qual_comparsion(hr_vel_crop[timepoints[0]:timepoints[-1]+1], lr_vel_crop[timepoints_lr[0]:timepoints_lr[-1]+1], pred_vel_crop[timepoints[0]:timepoints[-1]+1], img_MV_mask,None,  [], [], min_v=None, max_v=None,  timepoints = timepoints,figsize=(8, 5),  save_as = f'{eval_dir}/{set_name}_M{data_model}_Velocity_through_MVplane_meanV_prediction.png')

        if False: 
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            #-----plot MV 2; 3D plot with plane and intersection----- 
            a = 30
            x_bounds, y_bounds, z_bounds = np.where(boundary_mask[t, :, :,:]==1)
            xp, yp, zp = idx_intersec_plane_fluid
            xl, yl, zl = np.where(points_plane_core ==2) 
            x_MV, y_MV, z_MV = np.where(points_MV ==1)
            ax.plot_surface(xx, yy, zz, alpha = 0.33, color = KI_colors['Grey']) # plot plane
            ax.scatter3D(x_bounds, y_bounds, z_bounds, s= 3, alpha = 0.1) #plot boundary points
            ax.scatter3D(plane_points[0], plane_points[1], plane_points[2],'x', color = 'red') #show point in plane
            ax.plot([plane_normal[0]*a, 0], [plane_normal[1]*a, 0], [plane_normal[2]*a, 0], color = 'black')
            ax.scatter3D(plane_points[0], plane_points[1] , plane_points[2] , s = 3, color = 'black') # plot normal point
            ax.scatter3D(x_MV, y_MV, z_MV, alpha = 0.2, s = 3, color = 'red') #plot MV points
            plt.xlabel('x')
            plt.ylabel('y')
            ax.set_zlabel('z')
            plt.show()

        #-----plot MV 3; Plot Flow profile within mask----- 

        #plot flow profile
        hr_flow_rate = calculate_flow_profile(hr_vel, img_MV_mask, [2, 2, 2])
        lr_flow_rate = calculate_flow_profile(lr_vel, img_MV_mask, [2, 2, 2])
        pred_flow_rate = calculate_flow_profile(pred_vel, img_MV_mask, [2, 2, 2])

        plt.figure(figsize=(8, 5))
        t_range_lr = np.arange(0, N_frames)[::2]
        plt.plot(hr_flow_rate, '-o',  label = 'HR', color = 'black', markersize = 3)
        plt.plot(t_range_lr, lr_flow_rate,'--o',  label = 'LR', color = 'forestgreen', markersize = 3)
        plt.plot(pred_flow_rate,'-o',  label = 'SR', color = KI_colors['Plum'], markersize = 3)
        plt.xlabel('Frame', fontsize = 16)
        plt.ylabel('Flow rate (ml/s)',  fontsize = 16)
        plt.legend(fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.tight_layout()
        plt.savefig(f'{eval_dir}/{set_name}_M{data_model}_MV_Flow_rate.png',bbox_inches='tight',transparent=True)
        plt.show()
    
    if show_planeAV_plot:
        print("Plot AV plane flow..")

        # define plane 
        t = 0
        plane_points = [68.5/2, 35.8/2, 79.18/2]
        plane_normal = [0.6147, -0.1981, -0.7635]
        order_normal = [2, 1, 0]
        plane_normal /= np.linalg.norm(plane_normal)

        # calculate the plane
        d = -np.dot(plane_points, plane_normal)
        xx, yy = np.meshgrid(np.arange(0, gt['u'].shape[1]), np.arange(0, gt['u'].shape[2]))
        zz = (-plane_normal[0] * xx - plane_normal[1] * yy - d) * 1. / plane_normal[2]

        zz[np.where(zz < 0)] = 0
        zz[np.where(zz >= gt['u'].shape[3])] = gt['u'].shape[3] - 1
        print(xx.max(), yy.max(), zz.max())
        print(xx.min(), yy.min(), zz.min())

        # Get point coordiantes in plane
        points_in_plane = np.zeros_like(gt['mask'][t])
        points_in_plane[xx.flatten().astype(int), yy.flatten().astype(int), zz.flatten().astype(int)] = 1

        #3D model: is just 1 in region, where plane AND fluid region is
        points_plane_core = points_in_plane.copy()
        points_plane_core[np.where(gt['mask'][t]==0)] = 0

        #Always adjust to different models
        points_MV = points_plane_core.copy()
        points_MV[:, :, :15] = 0
        points_MV[:, 22:, :] = 0
        points_MV[35:, :, :] = 0

        #2. Get points in plane and cut out right region

        #get indices
        idx_intersec_plane_fluid = np.where(points_plane_core>0)
        idx_plane                = np.where(points_in_plane>0)
        idx_MV                   = np.where(points_MV>0) 

        img_mask = gt['mask'][t][idx_plane].reshape(xx.shape[1], -1)
        img_MV_mask = points_MV[idx_plane].reshape(xx.shape[1], -1)
        plt.imshow(gt['mask'][t][idx_plane].reshape(xx.shape[1], -1))
        plt.imshow(img_MV_mask+img_mask)
        plt.show()
        lr_vel   = velocity_through_plane(idx_plane, lr, plane_normal, order_normal = order_normal).reshape(lr['u'].shape[0], xx.shape[1], -1)
        hr_vel   = velocity_through_plane(idx_plane, gt, plane_normal, order_normal = order_normal).reshape(gt['u'].shape[0], xx.shape[1], -1)
        pred_vel = velocity_through_plane(idx_plane, pred, plane_normal, order_normal = order_normal).reshape(pred['u'].shape[0], xx.shape[1], -1)

        #-----plot MV 1; Qualitave plot----- 
        idx_crop = np.index_exp[:, 17:37, 5:25]
        idx_crop2 = np.index_exp[17:37, 5:25]

        # crop to important region
        lr_vel_crop = lr_vel[idx_crop]
        hr_vel_crop = hr_vel[idx_crop ]
        pred_vel_crop = pred_vel[idx_crop]
        img_MV_mask_crop = img_MV_mask[idx_crop2]

        print('Shapes AV cubes:', hr_vel_crop.shape, lr_vel_crop.shape, pred_vel_crop.shape, img_MV_mask_crop.shape)
        timepoints = [6, 7, 8, 9]
        timepoints_lr = [3, 4, 5, 6]
        plot_qual_comparsion(hr_vel_crop[timepoints[0]:timepoints[-1]+1], lr_vel_crop[timepoints_lr[0]:timepoints_lr[-1]+1], pred_vel_crop[timepoints[0]:timepoints[-1]+1], img_MV_mask,None,  [], [], min_v=None, max_v=None,  timepoints = timepoints,figsize=(8, 5),  save_as = f'{eval_dir}/{set_name}_M{data_model}_Velocity_through_AVplane_3D_img_meanV_prediction.png')

        if False: 
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            #-----plot MV 2; 3D plot with plane and intersection----- 
            a = 30
            x_bounds, y_bounds, z_bounds = np.where(boundary_mask[t, :, :,:]==1)
            xp, yp, zp = idx_intersec_plane_fluid
            xl, yl, zl = np.where(points_plane_core ==2) 
            x_MV, y_MV, z_MV = np.where(points_MV ==1)
            ax.plot_surface(xx, yy, zz, alpha = 0.33, color = KI_colors['Grey']) # plot plane
            ax.scatter3D(x_bounds, y_bounds, z_bounds, s= 3, alpha = 0.1) #plot boundary points
            ax.scatter3D(plane_points[0], plane_points[1], plane_points[2],'x', color = 'red') #show point in plane
            ax.plot([plane_normal[0]*a, 0], [plane_normal[1]*a, 0], [plane_normal[2]*a, 0], color = 'black')
            ax.scatter3D(plane_points[0], plane_points[1] , plane_points[2] , s = 3, color = 'black') # plot normal point
            ax.scatter3D(x_MV, y_MV, z_MV, alpha = 0.2, s = 3, color = 'red') #plot MV points
            plt.xlabel('x')
            plt.ylabel('y')
            ax.set_zlabel('z')
            plt.show()

        #-----plot AV 3; Plot Flow profile within mask----- 

        #plot flow profile
        hr_flow_rate = calculate_flow_profile(hr_vel, img_MV_mask, [2, 2, 2])
        lr_flow_rate = calculate_flow_profile(lr_vel, img_MV_mask, [2, 2, 2])
        pred_flow_rate = calculate_flow_profile(pred_vel, img_MV_mask, [2, 2, 2])

        plt.figure(figsize=(8, 5))
        t_range_lr = np.arange(0, N_frames)[::2]
        plt.plot(hr_flow_rate, '-o',  label = 'HR', color = 'black', markersize = 3)
        plt.plot(t_range_lr, lr_flow_rate,'-o',  label = 'LR', color = 'forestgreen', markersize = 3)
        plt.plot(pred_flow_rate,'--o',  label = 'SR', color = KI_colors['Plum'], markersize = 3)
        plt.xlabel('Frame', fontsize = 16)
        plt.ylabel('Flow rate (ml/s)',  fontsize = 16)
        plt.legend(fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.tight_layout()
        plt.savefig(f'{eval_dir}/{set_name}_M{data_model}_AV_Flow_rate.png',bbox_inches='tight', transparent = True)
        plt.show()
        
        #-----plot AV 4; Plot velocity profile within mask----- 

        #plot flow profile
        plt.figure(figsize=(8, 5))
        t_range_lr = np.arange(0, N_frames)[::2]
        plt.plot(np.mean(hr_vel, where=img_MV_mask.astype(bool), axis=(1, 2)), label = 'HR', color = 'black')
        plt.plot(t_range_lr, np.mean(lr_vel, where=img_MV_mask.astype(bool), axis=(1, 2)), label = 'LR', color = KI_colors['Orange'])
        plt.plot(np.mean(pred_vel, where=img_MV_mask.astype(bool), axis=(1, 2)), label = 'SR', color = KI_colors['Plum'])
        plt.xlabel('Frame', fontsize = 16)
        plt.ylabel('velocity (m/s)',  fontsize = 16)
        plt.legend(fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.tight_layout()
        plt.savefig(f'{eval_dir}/{set_name}_M{data_model}_AV_velocity_plane.svg',bbox_inches='tight')
        plt.show()


    if show_animation:
        print("Plot animation..")

        idx_slice = np.index_exp[20, :, :]
        fps = 20
        # animate prediction (slice)
        animate_data_over_time_gif(idx_slice, pred['u'], min_v['u'], max_v['u'], fps = fps , save_as = f'{eval_dir}/{set_name}_animate_u_pred_cbar_jet', show_colorbar = True, colormap='jet')
        animate_data_over_time_gif(idx_slice, pred['v'], min_v['v'], max_v['v'], fps = fps , save_as = f'{eval_dir}/{set_name}_animate_v_pred_cbar_jet', show_colorbar = True, colormap='jet')
        animate_data_over_time_gif(idx_slice, pred['w'], min_v['w'], max_v['w'], fps = fps , save_as = f'{eval_dir}/{set_name}_animate_w_pred_cbar_jet', show_colorbar = True, colormap='jet')

        if False: 
            # animate HR (slice)
            animate_data_over_time_gif(idx_slice, gt['u'], min_v['u'], max_v['u'], fps = fps , save_as = f'{eval_dir}/{set_name}_animate_u_gt')
            animate_data_over_time_gif(idx_slice, gt['v'], min_v['v'], max_v['v'], fps = fps , save_as = f'{eval_dir}/{set_name}_animate_v_gt')
            animate_data_over_time_gif(idx_slice, gt['w'], min_v['w'], max_v['w'], fps = fps , save_as = f'{eval_dir}/{set_name}_animate_w_gt')
            # animate LR (slice)
            animate_data_over_time_gif(idx_slice, lr['u'], min_v['u'], max_v['u'], fps = fps//2 , save_as = f'{eval_dir}/{set_name}_animate_u_lr')
            animate_data_over_time_gif(idx_slice, lr['v'], min_v['v'], max_v['v'], fps = fps//2 , save_as = f'{eval_dir}/{set_name}_animate_v_lr')
            animate_data_over_time_gif(idx_slice, lr['w'], min_v['w'], max_v['w'], fps = fps//2 , save_as = f'{eval_dir}/{set_name}_animate_w_lr')

        # animate abs error
        max_abs_err = np.max([np.abs(gt['u'] - pred['u'])[idx_slice], np.abs(gt['v'] - pred['v'])[idx_slice], np.abs(gt['w'] - pred['w'])[idx_slice]])
        animate_data_over_time_gif(idx_slice, np.abs(gt['u'] - pred['u']), None, None, fps = fps , save_as = f'{eval_dir}/{set_name}_animate_u_abs_error', show_colorbar = True)
        animate_data_over_time_gif(idx_slice, np.abs(gt['v'] - pred['v']), None, None, fps = fps , save_as = f'{eval_dir}/{set_name}_animate_v_abs_error', show_colorbar = True)
        animate_data_over_time_gif(idx_slice, np.abs(gt['w'] - pred['w']), None, None, fps = fps , save_as = f'{eval_dir}/{set_name}_animate_w_abs_error', show_colorbar = True)

    if save_as_vti:
        print("Save as vti files..")
        if not os.path.isdir(f'{eval_dir}/vti'):
            os.makedirs(f'{eval_dir}/vti')
        for t in range(N_frames):
            output_filepath = f'{eval_dir}/vti/M{data_model}_HR_frame{t}_uvw.vti'
            if os.path.isfile(output_filepath):
                print(f'File {output_filepath} already exists')
            else:
                spacing = [2, 2, 2]
                uvw_mask_to_vtk((gt['u'][t],gt['v'][t],gt['w'][t]),gt['mask'][t], spacing, output_filepath, include_mask = True)


    # ------------Tabular evaluations----------------
    
    # calculate k, r2 and RMSE, RE values for all frames
    if tabular_eval:

        if True:
            #calculate error for each velocity component
            vel_and_speed_colnames = vel_colnames + ['speed']
            k_SR = defaultdict(list)
            r2_SR = defaultdict(list)
            df_raw = pd.DataFrame()
            df_summary = pd.DataFrame(index=vel_and_speed_colnames)
            k_SR  = np.zeros((len(vel_and_speed_colnames), N_frames))
            r2_SR = np.zeros((len(vel_and_speed_colnames), N_frames))

            for vel in vel_and_speed_colnames:
                print(f'------------------Calculate error for {vel}---------------------')
                # rmse
                rmse_pred = calculate_rmse(pred[vel], gt[vel], gt["mask"], return_std=False)
                rmse_pred_nonfluid = calculate_rmse(pred[vel], gt[vel], reverse_mask)

                # absolute error
                abs_err = np.mean(np.abs(pred[vel] - gt[vel]), where = bool_mask, axis = (1,2,3))

                # k and R2 values
                k_core, r2_core     = calculate_k_R2_timeseries(pred[vel], gt[vel], core_mask)
                k_bounds, r2_bounds = calculate_k_R2_timeseries(pred[vel], gt[vel], boundary_mask)
                k_all, r2_all       = calculate_k_R2_timeseries(pred[vel], gt[vel], gt['mask'])

                # Populate df_raw with the calculated metrics
                df_raw[f'k_core_{vel}'] = k_core
                df_raw[f'k_bounds_{vel}'] = k_bounds
                df_raw[f'k_all_{vel}'] = k_all
                df_raw[f'r2_core_{vel}'] = r2_core
                df_raw[f'r2_bounds_{vel}'] = r2_bounds
                df_raw[f'r2_all_{vel}'] = r2_all
                df_raw[f'rmse_pred_{vel}'] = rmse_pred
                df_raw[f'rmse_pred_nonfluid_{vel}'] = rmse_pred_nonfluid
                df_raw[f'abs_err_{vel}'] = abs_err

                # Summary statistics for df_summary
                metrics = {
                    'k_core': k_core,
                    'k_bounds': k_bounds,
                    'k_all': k_all,
                    '|1-k|_core': np.abs(1 - k_core),
                    '|1-k|_bounds': np.abs(1 - k_bounds),
                    '|1-k|_all': np.abs(1 - k_all),
                    'r2_core': r2_core,
                    'r2_bounds': r2_bounds,
                    'r2_all': r2_all
                }

                # Convert metrics dictionary to a DataFrame for easier aggregation
                metrics_df = pd.DataFrame(metrics)

                # Calculate summary statistics and assign to df_summary
                for metric in metrics.keys():
                    df_summary.loc[vel, f'{metric}_avg_SR'] = metrics_df[metric].mean()
                    df_summary.loc[vel, f'{metric}_min_SR'] = metrics_df[metric].min()
                    df_summary.loc[vel, f'{metric}_max_SR'] = metrics_df[metric].max()

                df_summary.loc[vel, 'rmse_avg_SR'] = np.mean(rmse_pred)
                df_summary.loc[vel, 'rmse_avg_nonfluid_SR'] = np.mean(rmse_pred_nonfluid)
                df_summary.loc[vel, 'abs_err_avg_SR'] = np.mean(abs_err)


            # Add relative error to df_raw and df_summary
            df_raw[f'RE'] = rel_error
            df_summary[f'RE_avg_SR'] = np.mean(rel_error)

            cos_similarity = cosine_similarity( gt['u'], gt['v'], gt['w'],pred['u'], pred['v'], pred['w'])
            df_raw['cos_sim'] = np.mean(cos_similarity, axis = (1,2,3), where=bool_mask)
            df_summary['cos_sim_avg_SR'] = np.mean(df_raw['cos_sim'])

            # add avg row over all vel. components
            df_summary.loc['v_all'] = df_summary.loc[['u', 'v', 'w']].mean()


            # Save dataframes to CSV
            df_raw.to_csv(f'{eval_dir}/{set_name}_M{data_model}_metric_evaluation_ALL.csv', float_format="%.3f")
            df_summary.to_csv(f'{eval_dir}/{set_name}_M{data_model}_metric_evaluation_core_bound_summary.csv', float_format="%.3f")

            df_summary_whole = pd.DataFrame(index=vel_and_speed_colnames.extend(['v_all']))
            columns = ['k_all_avg', '|1-k|_all_avg', 'r2_all_avg',  'rmse_avg', 'rmse_avg_nonfluid', 'abs_err_avg', 'cos_sim_avg', 'RE_avg']
            columns_SR = [f'{col}_SR' for col in columns]
            df_summary_whole[columns_SR] = df_summary[columns_SR]
            df_summary_whole.to_csv(f'{eval_dir}/{set_name}_M{nn_name}metric_evaluation_summary_selected.csv', float_format="%.3f")
            print(df_summary_whole.columns)

            print(df_summary)
            print(df_summary.to_latex(index=False, float_format="%.2f"))
            # print(df_raw.to_latex(index=False, float_format="%.2f"))


        if load_interpolation_files:
            interpolate_sinc['speed'] = np.sqrt(interpolate_sinc['u']**2 + interpolate_sinc['v']**2 + interpolate_sinc['w']**2 )
            interpolate_linear['speed'] = np.sqrt(interpolate_linear['u']**2 + interpolate_linear['v']**2 + interpolate_linear['w']**2 )

            #calculate error for each velocity component
            vel_and_speed_colnames = vel_colnames + ['speed']
            k_SR = defaultdict(list)
            r2_SR = defaultdict(list)
            df_summary_interp = pd.DataFrame(index=vel_and_speed_colnames)
            k_SR  = np.zeros((len(vel_and_speed_colnames), N_frames))
            r2_SR = np.zeros((len(vel_and_speed_colnames), N_frames))

            for vel in vel_and_speed_colnames:
                print(f'------------------Calculate interpolation error for {vel}---------------------')
                # rmse
                rmse_pred_sinc = calculate_rmse(interpolate_sinc[vel], gt[vel], gt["mask"], return_std=False)
                rmse_pred_linear = calculate_rmse(interpolate_linear[vel], gt[vel], gt["mask"], return_std=False)
                rmse_pred_nonfluid_sinc = calculate_rmse(interpolate_sinc[vel], gt[vel], reverse_mask)
                rmse_pred_nonfluid_linear = calculate_rmse(interpolate_linear[vel], gt[vel], reverse_mask)

                # absolute error
                abs_err_sinc = np.mean(np.abs(interpolate_sinc[vel] - gt[vel]), where = bool_mask, axis = (1,2,3))
                abs_err_linear = np.mean(np.abs(interpolate_linear[vel] - gt[vel]), where = bool_mask, axis = (1,2,3))

                # k and R2 values
                k_core_sinc, r2_core_sinc     = calculate_k_R2_timeseries(interpolate_sinc[vel], gt[vel], core_mask)
                k_bounds_sinc, r2_bounds_sinc = calculate_k_R2_timeseries(interpolate_sinc[vel], gt[vel], boundary_mask)
                k_all_sinc, r2_all_sinc       = calculate_k_R2_timeseries(interpolate_sinc[vel], gt[vel], gt['mask'])

                k_core_linear, r2_core_linear     = calculate_k_R2_timeseries(interpolate_linear[vel], gt[vel], core_mask)
                k_bounds_linear, r2_bounds_linear = calculate_k_R2_timeseries(interpolate_linear[vel], gt[vel], boundary_mask)
                k_all_linear, r2_all_linear       = calculate_k_R2_timeseries(interpolate_linear[vel], gt[vel], gt['mask'])

                # Summary statistics for df_summary_interp
                metrics_sinc = {
                'k_core': k_core_sinc,
                'k_bounds': k_bounds_sinc,
                'k_all': k_all_sinc,
                '|1-k|_core': np.abs(1 - k_core_sinc),
                '|1-k|_bounds': np.abs(1 - k_bounds_sinc),
                '|1-k|_all': np.abs(1 - k_all_sinc),
                'r2_core': r2_core_sinc,
                'r2_bounds': r2_bounds_sinc,
                'r2_all': r2_all_sinc
                }

                metrics_linear = {
                'k_core': k_core_linear,
                'k_bounds': k_bounds_linear,
                'k_all': k_all_linear,
                '|1-k|_core': np.abs(1 - k_core_linear),
                '|1-k|_bounds': np.abs(1 - k_bounds_linear),
                '|1-k|_all': np.abs(1 - k_all_linear),
                'r2_core': r2_core_linear,
                'r2_bounds': r2_bounds_linear,
                'r2_all': r2_all_linear
                }

                # Convert metrics dictionary to a DataFrame for easier aggregation
                metrics_df_sinc = pd.DataFrame(metrics_sinc)
                metrics_df_linear = pd.DataFrame(metrics_linear)

                # Calculate summary statistics and assign to df_summary_interp
                for metric in metrics_sinc.keys():
                    df_summary_interp.loc[vel, f'{metric}_avg_sinc'] = metrics_df_sinc[metric].mean()
                    df_summary_interp.loc[vel, f'{metric}_min_sinc'] = metrics_df_sinc[metric].min()
                    df_summary_interp.loc[vel, f'{metric}_max_sinc'] = metrics_df_sinc[metric].max()

                for metric in metrics_linear.keys():
                    df_summary_interp.loc[vel, f'{metric}_avg_linear'] = metrics_df_linear[metric].mean()
                    df_summary_interp.loc[vel, f'{metric}_min_linear'] = metrics_df_linear[metric].min()
                    df_summary_interp.loc[vel, f'{metric}_max_linear'] = metrics_df_linear[metric].max()

                df_summary_interp.loc[vel, 'rmse_avg_sinc'] = np.mean(rmse_pred_sinc)
                df_summary_interp.loc[vel, 'rmse_avg_linear'] = np.mean(rmse_pred_linear)
                df_summary_interp.loc[vel, 'rmse_avg_nonfluid_sinc'] = np.mean(rmse_pred_nonfluid_sinc)
                df_summary_interp.loc[vel, 'rmse_avg_nonfluid_linear'] = np.mean(rmse_pred_nonfluid_linear)
                df_summary_interp.loc[vel, 'abs_err_avg_sinc'] = np.mean(abs_err_sinc)
                df_summary_interp.loc[vel, 'abs_err_avg_linear'] = np.mean(abs_err_linear)

            # Add relative error to df_summary_interp
            RE_sinc = calculate_relative_error_normalized(interpolate_sinc['u'], interpolate_sinc['v'], interpolate_sinc['w'], gt['u'], gt['v'], gt['w'], gt['mask'])
            RE_linear = calculate_relative_error_normalized(interpolate_linear['u'], interpolate_linear['v'], interpolate_linear['w'], gt['u'], gt['v'], gt['w'], gt['mask'])

            df_summary_interp[f'RE_avg_sinc'] = np.mean(RE_sinc)
            df_summary_interp[f'RE_avg_linear'] = np.mean(RE_linear)

            cos_similarity_sinc = cosine_similarity( gt['u'], gt['v'], gt['w'],interpolate_sinc['u'], interpolate_sinc['v'], interpolate_sinc['w'])
            cos_similarity_linear = cosine_similarity( gt['u'], gt['v'], gt['w'],interpolate_linear['u'], interpolate_linear['v'], interpolate_linear['w'])
            df_summary_interp['cos_sim_avg_linear'] = np.mean(np.mean(cos_similarity_linear, axis = (1,2,3), where=bool_mask))
            df_summary_interp['cos_sim_avg_sinc'] = np.mean(np.mean(cos_similarity_sinc, axis = (1,2,3), where=bool_mask))

            # add avg row over all vel. components
            df_summary_interp.loc['v_all'] = df_summary_interp.loc[['u', 'v', 'w']].mean()

            # Save dataframe to CSV
            
            df_summary_interp.to_csv(f'{eval_dir}/{set_name}_M{data_model}_Interpolation_metric_evaluation_summary.csv', float_format="%.3f")

            print(df_summary_interp)
            print(df_summary_interp.to_latex(index=False, float_format="%.2f"))

            df_summary_interp_whole = pd.DataFrame(index=vel_and_speed_colnames.extend(['v_all']))
            columns = ['|1-k|_all_avg_linear', 'r2_all_avg_linear',  'rmse_avg_linear', 'rmse_avg_nonfluid_linear', 'abs_err_avg_linear', 'cos_sim_avg_linear', 'RE_avg_linear', 
                    '|1-k|_all_avg_sinc', 'r2_all_avg_sinc',  'rmse_avg_sinc', 'rmse_avg_nonfluid_sinc', 'abs_err_avg_sinc', 'cos_sim_avg_sinc', 'RE_avg_sinc'
                    ]
            df_summary_interp_whole[columns] = df_summary_interp[columns]
            df_summary_interp_whole = df_summary_interp_whole.transpose()
            df_summary_interp_whole.to_csv(f'{eval_dir}/{set_name}_M{data_model}_Interpolation_metric_evaluation_summary_selected.csv', float_format="%.3f")

            eval_vel = 'v_all'

            print('SINC interpolation values of model: ', eval_vel)
            print('MAE |V|:', df_summary_interp.loc[eval_vel, 'abs_err_avg_sinc'])
            print('RMSE |V|:', df_summary_interp.loc[eval_vel, 'rmse_avg_sinc'])
            print('CO SIM:', df_summary_interp.loc[eval_vel, 'cos_sim_avg_sinc'])
            print('RE avg: ', df_summary_interp.loc[eval_vel, 'RE_avg_sinc'])

            print('LINEAR interpolation values of model: ', eval_vel)
            print('MAE |V|:', df_summary_interp.loc[eval_vel, 'abs_err_avg_linear'])
            print('RMSE |V|:', df_summary_interp.loc[eval_vel, 'rmse_avg_linear'])
            print('CO SIM:', df_summary_interp.loc[eval_vel, 'cos_sim_avg_linear'])
            print('RE avg: ', df_summary_interp.loc[eval_vel, 'RE_avg_linear'])

            # make comparison metric
            comp_metrics = ['RE_avg' ,'abs_err_avg', 'rmse_avg', 'cos_sim_avg','|1-k|_all_avg', 'r2_all_avg',  'rmse_avg_nonfluid']
            interpolation_methods = ['V_SR', 'V_linear',  'V_sinc'] #'V_cubic',

            df_comparison = pd.DataFrame(index=comp_metrics, columns=interpolation_methods)

            # Populate df_comparison using the v_all row from df_summary_whole
            for metric in comp_metrics:
                df_comparison.loc[metric, 'V_SR'] = df_summary_whole.loc[f'v_all', f'{metric}_SR']  # Assuming '_SR' suffix
                df_comparison.loc[metric, 'V_linear'] = df_summary_interp.loc[f'v_all', f'{metric}_linear']  # '_linear'
                # df_comparison.loc[metric, 'V_cubic'] = 0.00  # Placeholder, as cubic is missing
                df_comparison.loc[metric, 'V_sinc'] = df_summary_interp.loc[f'v_all', f'{metric}_sinc']  # '_sinc'

            # Display the comparison DataFrame
            print(df_comparison)
            df_comparison = df_comparison.apply(pd.to_numeric, errors='coerce')

            # Save the DataFrame to a CSV or LaTeX format if needed
            df_comparison.to_csv(f'{eval_dir}/{set_name}_M{data_model}_comparison_metrics.csv', float_format="%.3f")
            print(df_comparison.to_latex(index=True, float_format="%.3f"))


    print('---------------DONE-----------------------')