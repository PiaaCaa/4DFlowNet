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
    model_name = '20240617-0933' 
    set_name = 'Test'               
    data_model= '4'
    step = 2
    load_interpolation_files = False
    ups_factor = 2

    #choose which plots to show
    show_img_plot = True
    show_RE_plot = True
    show_corr_plot = True
    show_planeMV_plot = True
    tabular_eval = True
    show_animation = False #TODO: implement
    save_as_vti = False


    # directories
    data_dir = 'data/CARDIAC'
    pred_dir = f'results/Temporal4DFlowNet_{model_name}'
    eval_dir = f'{pred_dir}/plots'

    vel_colnames=['u', 'v', 'w']

    # filenames
    gt_filename = f'M{data_model}_2mm_step{step}_cs_invivoP02_hr.h5'
    lr_filename = f'M{data_model}_2mm_step{step}_cs_invivoP02_lr_lessnoise.h5'

    pred_filename = f'{set_name}set_result_model{data_model}_2mm_step{step}_{model_name[-4::]}_temporal.h5'
    
    # Setting up
    gt_filepath   = '{}/{}'.format(data_dir, gt_filename)
    pred_filepath = '{}/{}'.format(pred_dir, pred_filename)
    lr_filepath   = '{}/{}'.format(data_dir, lr_filename)

    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    # ----------Load data and interpolation files and calculate visualization params----------------

    gt, lr, pred = load_vel_data(gt_filepath, lr_filepath, pred_filepath, vel_colnames = vel_colnames)

    print('shapes:', )

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

    # Caluclation for further plotting

    # Relative error calculation
    if show_RE_plot or tabular_eval:
        rel_error = calculate_relative_error_normalized(pred['u'], pred['v'], pred['w'], gt['u'], gt['v'], gt['w'], gt['mask'])

    # -------------Qualitative evaluation----------------


    # 1. Qualitative visalisation of the LR, HR and prediction

    if show_img_plot:
        print("Plot example time frames..")
        
        frames = [32, 33, 34, 35]
        idx_cube = np.index_exp[frames[0]:frames[-1]+1, 20, 0:40, 20:60]
        idx_cube_lr = np.index_exp[frames[0]//2:frames[-1]//2+1, 20, 0:40, 20:60]

        input_lst = []
        input_name =[]
        if load_interpolation_files:
            # input_lst = [interpolate_linear[idx_cube], interpolate_cubic[idx_cube]]
            # input_name = ['linear', 'cubic']
            # input_lst_ = [interpolate_sinc[idx_cube]]
            input_name = ['sinc']

            plot_qual_comparsion(gt['u'][idx_cube],lr['u'][idx_cube_lr],  pred['u'][idx_cube],gt['mask'][idx_cube], np.abs(gt['u'][idx_cube]- pred['u'][idx_cube]), [interpolate_sinc['u'][idx_cube]], ['sinc'], frames,min_v = min_v['u'], max_v = max_v['u'],figsize = (8, 5), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_u_test.png")
            plot_qual_comparsion(gt['v'][idx_cube],lr['v'][idx_cube_lr],  pred['v'][idx_cube],gt['mask'][idx_cube], np.abs(gt['v'][idx_cube]- pred['v'][idx_cube]), [interpolate_sinc['v'][idx_cube]], ['sinc'], frames,min_v = min_v['v'], max_v = max_v['v'],figsize = (3, 6), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_v_test.png")
            plot_qual_comparsion(gt['w'][idx_cube],lr['w'][idx_cube_lr],  pred['w'][idx_cube],gt['mask'][idx_cube], np.abs(gt['w'][idx_cube]- pred['w'][idx_cube]), [interpolate_sinc['w'][idx_cube]], ['sinc'], frames,min_v = min_v['w'], max_v = max_v['w'],figsize = (3, 6), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_w_test.png")
        else:
            plot_qual_comparsion(gt['u'][idx_cube],lr['u'][idx_cube_lr],  pred['u'][idx_cube],gt['mask'][idx_cube], np.abs(gt['u'][idx_cube]- pred['u'][idx_cube]), [], [], frames,min_v = min_v['u'], max_v = max_v['u'],figsize = (8,5), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_u_test.png")
            plot_qual_comparsion(gt['v'][idx_cube],lr['v'][idx_cube_lr],  pred['v'][idx_cube],gt['mask'][idx_cube], np.abs(gt['v'][idx_cube]- pred['v'][idx_cube]), [], [], frames,min_v = min_v['v'], max_v = max_v['v'],figsize = (8,5), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_v_test.png")
            plot_qual_comparsion(gt['w'][idx_cube],lr['w'][idx_cube_lr],  pred['w'][idx_cube],gt['mask'][idx_cube], np.abs(gt['w'][idx_cube]- pred['w'][idx_cube]), [], [], frames,min_v = min_v['w'], max_v = max_v['w'],figsize = (8,5), save_as = f"{eval_dir}/{set_name}_M{data_model}_Qualit_frameseq_w_test.png")

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

    # 3. Plot the correlation between the prediction and the ground truth in peak flow frame

    if show_corr_plot:
        print("Plot linear regression plot between prediction and ground truth in peak flow frame..")

        T_peak_flow = np.unravel_index(np.argmax(gt["u"]), shape =gt["u"].shape)[0]
        print("Peak flow frame for model", set_name, T_peak_flow)
        if T_peak_flow%ups_factor == 0: 
            T_peak_flow +=1
            print('Increase peak flow frame to next frame', T_peak_flow)

        # # 4. Plot slope and R2 values for core, boundary and all voxels over time
        k, r2 = calculate_and_plot_k_r2_vals_nobounds(gt, pred,gt['mask'], T_peak_flow,figsize=(15, 5), save_as = f'{eval_dir}/{set_name}_M{model_name}_corr_k_r2_vals_nobounds_frame{T_peak_flow}_pred')

        fig2, axs1 = plot_k_r2_vals_nobounds(k, r2, T_peak_flow, figsize = (15, 5),exclude_tbounds = True,  save_as= f'{eval_dir}/{set_name}_M{model_name}_corr_k_r2_vals_nobounds_frame{T_peak_flow}_pred')
        fig1 = plot_correlation_nobounds(gt, pred, T_peak_flow, show_text = True, save_as = f'{eval_dir}/{set_name}_M{model_name}_correlation_pred_nobounds_frame{T_peak_flow}')
        
        # Merge the two figures into a single figure
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.axis('off')  # Turn off the axes for the first subplot
        ax2.axis('off')  # Turn off the axes for the second subplot
        fig.subplots_adjust(wspace=0)  # Adjust the spacing between subplots
        fig1.subplots_adjust(wspace=0)  # Adjust the spacing between subplots in the first figure
        fig2.subplots_adjust(wspace=0)  # Adjust the spacing between subplots in the second figure
        ax1.imshow(fig1.canvas.renderer._renderer)
        ax2.imshow(fig2.canvas.renderer._renderer)
        plt.tight_layout()
        plt.savefig(f'{eval_dir}/{set_name}_M{model_name}_correlation_frame{T_peak_flow}_K_R2.png')
        
        plt.show()

    # 4. Plot MV plot through Mitral valve

    if show_planeMV_plot:
        print("Plot MV plane plot..")

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

        # ccrop to important region
        lr_vel_crop = lr_vel[idx_crop]
        hr_vel_crop = hr_vel[idx_crop ]
        pred_vel_crop = pred_vel[idx_crop]
        img_MV_mask_crop = img_MV_mask[idx_crop2]

        timepoints = [6, 7, 8, 9]
        # plot_qual_comparsion(hr_vel_crop[timepoints[0]:timepoints[-1]+1], lr_vel_crop[timepoints[0]:timepoints[-1]+1], pred_vel_crop[timepoints[0]:timepoints[-1]+1], img_MV_mask,None,  [], [], min_v=min_v['u'], max_v=max_v['u'],  timepoints = timepoints,figsize=(8, 5),  save_as = f'{eval_dir}/{set_name}_M{data_model}_Velocity_through_plane_3D_img_meanV_prediction.png')

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

        print('shapes', hr_vel.shape, img_MV_mask.shape, lr_vel.shape, pred_vel.shape)
        #plot flow profile
        hr_flow_rate = calculate_flow_profile(hr_vel, img_MV_mask, [2, 2, 2])
        lr_flow_rate = calculate_flow_profile(lr_vel, img_MV_mask, [2, 2, 2])
        pred_flow_rate = calculate_flow_profile(pred_vel, img_MV_mask, [2, 2, 2])

        plt.figure(figsize=(8, 5))
        t_range_lr = np.arange(0, N_frames)[::2]
        plt.plot(hr_flow_rate, label = 'HR', color = 'black')
        plt.plot(t_range_lr, lr_flow_rate, label = 'LR', color = KI_colors['Orange'])
        plt.plot(pred_flow_rate, label = 'SR', color = KI_colors['Plum'])
        plt.xlabel('Frame', fontsize = 16)
        plt.ylabel('Flow rate (ml/s)',  fontsize = 16)
        plt.legend(fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.tight_layout()
        plt.savefig(f'{eval_dir}/{set_name}_M{data_model}_MV_Flow_rate_MV.svg',bbox_inches='tight')
        plt.show()




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

        #calculate error for each velocity component
        vel_and_speed_colnames = vel_colnames + ['speed']
        k = defaultdict(list)
        r2 = defaultdict(list)
        df_raw = pd.DataFrame()
        df_summary = pd.DataFrame(index=vel_and_speed_colnames)
        k  = np.zeros((len(vel_and_speed_colnames), N_frames))
        r2 = np.zeros((len(vel_and_speed_colnames), N_frames))

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
                'r2_core': r2_core,
                'r2_bounds': r2_bounds,
                'r2_all': r2_all
            }

            # Convert metrics dictionary to a DataFrame for easier aggregation
            metrics_df = pd.DataFrame(metrics)

            # Calculate summary statistics and assign to df_summary
            for metric in metrics.keys():
                df_summary.loc[vel, f'{metric}_avg'] = metrics_df[metric].mean()
                df_summary.loc[vel, f'{metric}_min'] = metrics_df[metric].min()
                df_summary.loc[vel, f'{metric}_max'] = metrics_df[metric].max()

            df_summary.loc[vel, 'rmse_avg'] = np.mean(rmse_pred)
            df_summary.loc[vel, 'rmse_avg_nonfluid'] = np.mean(rmse_pred_nonfluid)
            df_summary.loc[vel, 'abs_err_avg'] = np.mean(abs_err)


        # Add relative error to df_raw and df_summary
        df_raw[f'RE'] = rel_error
        df_summary[f'RE_avg'] = np.mean(rel_error)

        cos_similarity = cosine_similarity( gt['u'], gt['v'], gt['w'],pred['u'], pred['v'], pred['w'])
        df_raw['cos_sim'] = np.mean(cos_similarity, axis = (1,2,3), where=bool_mask)
        df_summary['cos_sim_avg'] = np.mean(df_raw['cos_sim'])

        # Save dataframes to CSV
        df_raw.to_csv(f'{eval_dir}/{set_name}_M{model_name}_k_r2_RE_ALL.csv', float_format="%.3f")
        df_summary.to_csv(f'{eval_dir}/{set_name}_M{model_name}_k_r2_RE_core_bound_summary.csv', float_format="%.3f")

        df_summary_whole = pd.DataFrame(index=vel_and_speed_colnames)
        columns = ['k_all_avg', 'r2_all_avg',  'rmse_avg', 'rmse_avg_nonfluid', 'abs_err_avg', 'cos_sim_avg', 'RE_avg']
        df_summary_whole[columns] = df_summary[columns]
        df_summary_whole.to_csv(f'{eval_dir}/{set_name}_M{model_name}_k_r2_RE_summary_whole.csv', float_format="%.3f")
        print(df_summary_whole.columns)

        print(df_summary)
        print(df_summary.to_latex(index=False, float_format="%.2f"))
        # print(df_raw.to_latex(index=False, float_format="%.2f"))

    print('---------------DONE-----------------------')