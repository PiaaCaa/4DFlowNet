import numpy as np
import tensorflow as tf
import time
import h5py
# from Network.PatchHandler3D import PatchHandler3D
from Network.PatchHandler3D_temporal import PatchHandler4D
from test_iterator import check_compatibility, load_indexes
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from utils.evaluate_utils import *
from scipy.ndimage import binary_erosion
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
from prepare_data.h5functions import save_to_h5
import matplotlib
from utils.colors import *
import matplotlib.animation as animation
plt.rcParams['figure.figsize'] = [10, 8]

def create_temporal_comparison_gif_single(spatial_idx, data,  min_v, max_v, eval_dir,title='', fps =10,  colormap = 'viridis'):

    print(title, 'nframes:', data.shape[0])
    fig = plt.figure(frameon=False)
    im1 = plt.imshow(data[0, spatial_idx[0], spatial_idx[1], spatial_idx[2]],interpolation='none', vmin=min_v, vmax=max_v, cmap = colormap)
    plt.axis('off')
    plt.tight_layout()

    #initialization function: plot the background of each frame
    def init():
        im1.set_data(np.random.random((5,5)))
        return [im1]

    # animation function.  This is called sequentially
    def animate(i):
        im1.set_array(data[i, spatial_idx[0], spatial_idx[1], spatial_idx[2]])
        return [im1]

    anim = animation.FuncAnimation(fig,animate, init_func=init,
                                frames = data.shape[0],
                                interval = 100, repeat = False) # in ms)
    anim.save(f'{eval_dir}/Animate_invivo_{title}_{fps}fps.gif', fps=fps)


def plot_slices_over_time3(gt_cube,lr_cube,  mask_cube, rel_error_cube, comparison_lst, comparison_name,timepoints, idxs,min_v, max_v,exclude_rel_error = True, save_as = "Frame_comparison.png", figsize = (30,20)):
    # assert len(timepoints) == gt_cube.shape[0] # timepoints must match size of first dimension of HR

    def row_based_idx(num_rows, num_cols, idx):
        return np.arange(1, num_rows*num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx-1]

    
    T = 3 + len(comparison_lst)
    N = len(timepoints)
    if exclude_rel_error: T -= 1

    # fig = plt.figure(figsize=(10,10))
    fig, axes = plt.subplots(nrows=T, ncols=N, constrained_layout=True, figsize=figsize)

    i = 1
    #idxs = get_indices(timepoints, axis, idx)
    gt_cube = gt_cube[idxs]
    mask_cube = mask_cube[idxs]
    
    # pred_cube = pred_cube[idxs]
    #lr = lr[idxs]

    # min_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.01)
    # max_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.99)
    if not exclude_rel_error:
        rel_error_slices =rel_error_cube[idxs]#[get_slice(rel_error_cube, t, axis, idx) for t in timepoints]
        min_rel_error = np.min(np.array(rel_error_slices))
        max_rel_error = np.max(np.array(rel_error_slices))
    for j,t in enumerate(timepoints):
        
        gt_slice = gt_cube[j]
        # pred_slice = pred_cube[j]

        lr_slice = np.zeros_like(gt_slice)
        if t%2 == 0: lr_slice = lr_cube[idxs][j]#get_slice(lr_cube, t//2, axis=axis, slice_idx=idx )
        plt.subplot(T, N, row_based_idx(T, N, i))

        if t%2 == 0:
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            plt.xticks([])
            plt.yticks([])
        if i == 1: plt.ylabel("LR", fontsize = 'small')
            
        plt.title('frame '+ str(t))
        plt.xticks([])
        plt.yticks([])
        # plt.axis('off')
        

        i +=1
        plt.subplot(T, N, row_based_idx(T, N, i))
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
        if i == 2: plt.ylabel("HR", fontsize = 'small')
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
            im = plt.imshow(comp[idxs][j], vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel(name, fontsize = 'small')
            plt.xticks([])
            plt.yticks([])

        if not exclude_rel_error:
            i +=1
            plt.subplot(T, N, row_based_idx(T, N, i))
            re_img = plt.imshow(rel_error_cube[idxs][j],vmin=min_rel_error, vmax=max_rel_error, cmap='viridis',aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel("abs. error", fontsize = 'small')
            plt.xticks([])
            plt.yticks([])
            if t == timepoints[-1]:
                plt.colorbar(re_img, ax = axes[-1], aspect = 10, label = 'abs. error ')

        
        i +=1
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.colorbar(im, ax=axes.ravel().tolist(), aspect = 50, label = 'velocity (m/s)')
    plt.savefig(save_as,bbox_inches='tight' )



if __name__ == "__main__":
    # for one network evluation on multiple models
    if True:
        # set directories 
        input_dir = 'Temporal4DFlowNet/data/PIA/THORAX/'
        res_dir = 'Temporal4DFlowNet/results/in_vivo/THORAX'
        eval_dir = 'Temporal4DFlowNet/results/in_vivo/THORAX/plots4'

        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir)

        dict_results = defaultdict(list)
        cases = ['P01', 'P02', 'P03', 'P04', 'P05'] 
        for c in cases:
            print('-------------------', c, '-------------------')
            in_vivo = f'{input_dir}/{c}/h5/{c}.h5'
            in_vivo_upsampled = f'{res_dir}/{c}_20230602-1701_8_4_arch_25Frames.h5' 
            name_evaluation = f'THORAX_{c}_{os.path.basename(in_vivo)[:-3]}_ModelII_OriginalMagnitude'

            #set slice index for animation
            idx_slice = np.index_exp[61, :, :]

            data_original = {}
            data_predicted = {}
            vencs = {}
            vel_colnames = ['u', 'v','w']
            mag_colnames = ['mag_u', 'mag_v', 'mag_w']
            venc_colnames = [  'u_max', 'v_max', 'w_max'] #['venc_u', 'venc_v', 'venc_w']#
            vel_plotnames = [r'$V_x$', r'$V_y$', r'$V_z$']
            mag_plotnames = [r'$M_x$', r'$M_y$', r'$M_z$']


            # load in-vivo data
            with h5py.File(in_vivo, mode = 'r' ) as p1:
                data_original['mask'] =  np.asarray(p1['mask']).squeeze()

                for vel, venc in zip(vel_colnames, venc_colnames):
                    vencs[venc] = np.asarray(p1[venc])
                    data_original[vel] = np.asarray(p1[vel], dtype = float).squeeze()#/np.max(vencs[venc]) #TODO change this
                    print('original', vel, data_original[vel].shape)
                    data_original[f'{vel}_fluid'] = np.multiply(data_original[vel], data_original['mask'])
                for mag in mag_colnames:
                    data_original[mag] =  np.asarray(p1[mag]).squeeze()

            

            # load prediction
            with h5py.File(in_vivo_upsampled, mode = 'r' ) as h_pred:
                for vel, venc in zip(vel_colnames, venc_colnames):
                    data_predicted[vel] = np.asarray(h_pred[f'{vel}_combined']) #/np.max(vencs[venc]) 
                    print('prediction', vel, data_predicted[vel].shape)
                    # data_predicted[vel] = data_predicted[vel][:data_1[vel].shape[0]] # TODO delete this, tis is just because it overwrote the file

                    # add information considering only the fluid regions  
                    if data_predicted[vel].shape[0] == data_original[vel].shape[0]:
                        data_predicted[f'{vel}_fluid'] = np.multiply(data_predicted[vel], data_original['mask'])
                        data_predicted['mask'] = data_original['mask']
                        
                    elif  data_predicted[vel].shape[0] == 2*data_original[vel].shape[0]:
                        t, x, y, z = data_original['mask'].shape
                        data_predicted['mask']= np.zeros((2*t,x, y, z ))
                        data_predicted['mask'][::2, :, :, :] = data_original['mask']
                        data_predicted['mask'][1::2, :, :, :] = data_original['mask']
                        data_predicted[f'{vel}_fluid'] = np.multiply(data_predicted[vel], data_predicted['mask'])

            print('Shape of predicted data and original data:', data_predicted['u'].shape, data_original['u'].shape)
            N_frames = data_original['u'].shape[0]
            print("Max val:", np.max(data_original['u']), np.max(data_original['v']), np.max(data_original['w']))
            print("Min val:", np.min(data_original['u']), np.min(data_original['v']), np.min(data_original['w']))

            N_frames_input_data = data_original['u'].shape[0]
            N_frames_pred_data = data_predicted['u'].shape[0]

            super_resolved_prediction = False if N_frames_input_data == N_frames_pred_data else True
            if super_resolved_prediction: print('Evaluation of higher resolved velocity field')
            if super_resolved_prediction: print('Prediction increases temporal resolution of original data by 2x. (super resolved) ..')

            #find lower and higher values to display velocity fields
            min_v = {}
            max_v = {}
            for vel in vel_colnames:
                min_v[vel] = np.quantile(data_original[vel][np.where(data_original['mask'] !=0)].flatten(), 0.01)
                max_v[vel] = np.quantile(data_original[vel][np.where(data_original['mask'] !=0)].flatten(), 0.99)
                print(min_v[vel], max_v[vel])

            max_V = np.max([max_v['u'], max_v['v'], max_v['w']])
            min_V = np.min([min_v['u'], min_v['v'], min_v['w']])
            #-----------------save img slices over time---------------------
            if False:
                time_point = 10
                slice_idx = np.index_exp[time_point, 18, :, :]
                fig, axes = plt.subplots(nrows=2, ncols=3, ) #constrained_layout=True
                for i, (vel, mag, nam_vel, name_mag) in enumerate(zip(vel_colnames, mag_colnames, vel_plotnames, mag_plotnames)):
                    plt.subplot(2, 3, i+1)
                    ax = plt.gca()
                    im1 = plt.imshow(data_original[vel][slice_idx], vmin = min_V, vmax = max_V)
                    plt.axis('off')
                    plt.title(nam_vel)
                    # plt.savefig(f'{eval_dir}/{c}_{vel}_Invivo_Original_Frame{time_point}.png', bbox_inches='tight')
                    
                    if i == 2:
                        # plt.colorbar(im1, ax = axes[0], aspect = 25, label = 'velocity')
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im1, cax=cax, label = 'velocity (m/s)')

                    plt.subplot(2, 3, 4+i)
                    ax = plt.gca()
                    im2 = plt.imshow(data_original[mag][slice_idx], cmap = 'Greys_r')
                    plt.axis('off')
                    plt.title(name_mag)
                    # plt.savefig(f'{eval_dir}/{c}_{mag}_Invivo_Original_Frame{time_point}.png', bbox_inches='tight')
                    if i == 2:
                        # plt.colorbar(im2, ax = axes[-1], aspect = 25, label = 'magnitude')
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im2, cax=cax, label = 'magnitude')

            
                plt.tight_layout()
                plt.savefig(f'{eval_dir}/{c}_SUBPLOT_Invivo_Original_Frame{time_point}.png', bbox_inches='tight')

            

            


            #----------------------------------------------
            magn = np.sqrt(data_original['mag_u']**2 + data_original['mag_v']**2 + data_original['mag_w']**2)
            speed = np.sqrt(data_original['u']**2 + data_original['v']**2 + data_original['w']**2)
            pc_mri = np.multiply(magn, speed)
            data_original['mean_speed'] = calculate_mean_speed(data_original["u_fluid"], data_original["v_fluid"] , data_original["w_fluid"], data_original["mask"])
            data_predicted['mean_speed'] = calculate_mean_speed(data_predicted["u_fluid"], data_predicted["v_fluid"] , data_predicted["w_fluid"], data_predicted["mask"])
            
            peak_flow_frame = np.argmax(data_original['mean_speed'])
            # if peak_flow_frame % 2 == 0: 
            #     eval_peak_flow_frame = peak_flow_frame +1 
            # else: 
            eval_peak_flow_frame = peak_flow_frame # take next frame if peak flow frame included in lr data
            print('MEAN SPEED', np.average(data_original['mean_speed']))
            print('MAX SPEED', np.max(speed))

            if True:
                #-------------mean speed plot---------------------
                plt.figure(figsize=(7, 4))
                step_pred = 0.5 if super_resolved_prediction else 1
                

                frame_range_input = np.arange(0, N_frames_input_data)#np.linspace(0, data_1['u'].shape[0]-1,  data_1['u'].shape[0])
                frame_range_predicted = np.arange(0, N_frames_input_data, step_pred)#np.linspace(0, data_1['u'].shape[0], data_predicted['u'].shape[0])
                print(frame_range_input, frame_range_predicted)
                plt.title('Mean speed')
                plt.plot(frame_range_predicted, data_predicted['mean_speed'], '.-', label = 'prediction', color= KTH_colors['blue100'])
                plt.plot(frame_range_input, data_original['mean_speed'],'--', label = 'noisy input data', color= 'black')
                if not super_resolved_prediction:
                    plt.plot(frame_range_input[::2], data_original['mean_speed'][::2],'.',  label ='sample points',  color= 'black')
                else:
                    plt.plot(frame_range_input, data_original['mean_speed'],'.',  label ='sample points',  color= 'black')
                plt.xlabel("Frame")
                plt.ylabel("Mean speed (cm/s)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'{eval_dir}/Meanspeed_{name_evaluation}.svg')
                plt.show()

                #---------------correlation + k + r^2 plots -------------------
                if not super_resolved_prediction:
                    # correlation
                    peak_flow_frame = np.argmax(data_original['mean_speed'])
                    bounds, core_mask = get_boundaries(data_original['mask'])
                    frame_corr_plot = peak_flow_frame.copy()
                    if frame_corr_plot % 2 == 0: frame_corr_plot +=1 # take next frame if peak flow frame included in lr data
                    plt.figure(figsize=(15, 5))
                    plot_correlation(data_original, data_predicted, bounds=bounds, frame_idx=frame_corr_plot, save_as=f'{eval_dir}/Correlation_frame{frame_corr_plot}_{name_evaluation}.svg')

                    
                    print('Peak flow frame:', peak_flow_frame)
                    frames = data_original['mean_speed'].shape[0]
                    k, r2, k_bounds, r2_bounds = np.zeros(3*frames), np.zeros(3*frames), np.zeros(3*frames), np.zeros(3*frames)
                    bounds_mask = bounds.copy()
                    inner_mask = data_original['mask'] - bounds_mask

                    #calculate k values in core and boundary region
                    for i, vel in enumerate(vel_colnames):
                        for t in range(frames):
                            k[t+i*frames], r2[t+i*frames]  = calculate_k_R2( data_predicted[vel][t], data_original[vel][t], core_mask[t])
                            k_bounds[t+i*frames], r2_bounds[t+i*frames]  = calculate_k_R2( data_predicted[vel][t], data_original[vel][t], bounds[t])

                    #plot k and r^2 values
                    plot_k_r2_vals(frames, k,k_bounds, r2,  r2_bounds, peak_flow_frame, name_evaluation, eval_dir)

                    #print mean k and r^2 values
                    dict_intermediate_results = defaultdict(list)
                
                    for i, vel in enumerate(vel_colnames):
                        for t in range(frames):
                            k, r2 = calculate_k_R2( data_predicted[vel][t], data_original[vel][t], data_original['mask'][t])
                            
                            dict_intermediate_results[f'k_{vel}'].append(k)
                            dict_intermediate_results[f'R2_{vel}'].append(r2)
                    
                    dict_results['Patient'].append(c)
                    for key in dict_intermediate_results.keys():
                        dict_results[key].append(np.mean(dict_intermediate_results[key]))
                        dict_results[f'{key}_std'].append(np.std(dict_intermediate_results[key]))
                        dict_results[f'{key}_peak'].append(dict_intermediate_results[key][eval_peak_flow_frame])
                    
                    print('Eval peak flow frame:', eval_peak_flow_frame, peak_flow_frame)




            #---------create animation------------------------
            if False:
                fps_anim = 10
                fps_pred = fps_anim*2 if super_resolved_prediction else fps_anim

                if not os.path.exists(f'{eval_dir}/Animate_invivo_case00{c}_mag_{fps_anim}fps.gif'):
                    create_temporal_comparison_gif_single(idx_slice, magn, 0, np.quantile(magn, 0.99),         eval_dir, fps = fps_anim , title = f'{c}_mag', colormap='Greys_r' )
                    create_temporal_comparison_gif_single(idx_slice, data_original['mask'], 0, 1,         eval_dir, fps = fps_anim , title = f'{c}_mask', colormap='Greys' )
                    create_temporal_comparison_gif_single(idx_slice, data_original['u'], min_v['u'], max_v['u'],      eval_dir, fps = fps_anim , title = f'{c}_u_gt')
                    create_temporal_comparison_gif_single(idx_slice, data_original['v'], min_v['v'], max_v['v'],      eval_dir, fps = fps_anim , title = f'{c}_v_gt')
                    create_temporal_comparison_gif_single(idx_slice, data_original['w'], min_v['w'], max_v['w'],      eval_dir, fps = fps_anim , title = f'{c}_w_gt')
                    create_temporal_comparison_gif_single(idx_slice, data_original['u_fluid'], min_v['u'], max_v['u'],eval_dir, fps = fps_anim , title = f'{c}_u_gt_fluid')
                    create_temporal_comparison_gif_single(idx_slice, data_original['v_fluid'], min_v['v'], max_v['v'],eval_dir, fps = fps_anim , title = f'{c}_v_gt_fluid')
                    create_temporal_comparison_gif_single(idx_slice, data_original['w_fluid'], min_v['w'], max_v['w'],eval_dir, fps = fps_anim , title = f'{c}_w_gt_fluid')

                create_temporal_comparison_gif_single(idx_slice, data_original['mask'], 0, 1,         eval_dir, fps = fps_anim , title = f'{c}_mask', colormap='Greys' )
                create_temporal_comparison_gif_single(idx_slice, data_predicted['u'], min_v['u'], max_v['u'], eval_dir, fps = fps_pred , title = f'{name_evaluation}_u_pred')
                create_temporal_comparison_gif_single(idx_slice, data_predicted['v'], min_v['v'], max_v['v'], eval_dir, fps = fps_pred , title = f'{name_evaluation}_v_pred')
                create_temporal_comparison_gif_single(idx_slice, data_predicted['w'], min_v['w'], max_v['w'], eval_dir, fps = fps_pred , title = f'{name_evaluation}_w_pred')

        r_dt = pd.DataFrame(dict_results).round(2)
        rearaanged_columns = ['Patient', 'k_u', 'k_u_std', 'k_v', 'k_v_std', 'k_w', 'k_w_std','R2_u', 'R2_u_std', 'R2_v', 'R2_v_std', 'R2_w', 'R2_w_std', 'k_u_peak',  'k_v_peak',  'k_w_peak', 'R2_u_peak',  'R2_v_peak', 'R2_w_peak']
        r_dt = r_dt[rearaanged_columns]

        print(r_dt.to_latex(index=False))
    # comparison of different networks on one dataset
    if False:
        
        patient = 'P01'
        pred_dir = 'Temporal4DFlowNet/results/in_vivo/THORAX'

        invivo_file = f'Temporal4DFlowNet/data/PIA/THORAX/{patient}/h5/{patient}.h5' 
        prediction_files = [f'{patient}_20230405-1417_8_4_arch.h5',f'{patient}_20230405-1417_8_4_arch_25Frames_MaskXMagnitude.h5',f'{patient}_20230405-1417_8_4_arch_25Frames_MaskASMagnitude.h5', f'{patient}_20230602-1701_8_4_arch_25Frames.h5']
        name_prediction = ['MI - Magnitude', r' MI - Mask $\cdot$ Mag.','MI - Mask as Mag.',  'MII Magnitude',  ]
        eval_dir = 'Temporal4DFlowNet/results/in_vivo/THORAX/plots2'
        name_comparison = 'THORAX_DifferentMagnitudeInputs'
        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir)

        frame_start = 1
        frame_end = 6
        idx_slice = np.index_exp[frame_start:frame_end, 18, :, :]
        save_slices_individual = False
        super_resolved_prediction = False
        vel_colnames = ['u', 'v', 'w']
        mag_colnames = ['mag_u', 'mag_v', 'mag_w']
        min_v = {}
        max_v = {}
        in_vivo_original = {}
        in_vivo_predicted = {}


        # load first original data 
        # find vel max and min
        with h5py.File(invivo_file, mode = 'r' ) as p1:
                mask = np.array(p1['mask'])[idx_slice]
                in_vivo_original['mask'] = np.array(p1['mask'])
                for vel in vel_colnames:

                    #save in dictionary
                    
                    in_vivo_original[vel] = np.array(p1[vel])
                    in_vivo_original[f'{vel}_fluid'] = np.multiply(in_vivo_original[vel], in_vivo_original['mask'])

                    slice_vel = np.array(p1[vel])[idx_slice]
                    min_v[vel] = np.quantile(slice_vel.flatten(), 0.01)
                    max_v[vel] = np.quantile(slice_vel.flatten(), 0.99)
                    
                    
                    if save_slices_individual:
                        t_range = np.arange(frame_start, frame_end)
                        for i, t in enumerate(t_range):
                            plt.imshow(slice_vel[i], vmin = min_v[vel], vmax = max_v[vel])
                            plt.axis('off')
                            plt.savefig(f'{eval_dir}/{patient}_{vel}_Invivo_Original_Frame{t}.png', bbox_inches='tight')

                            plt.imshow(np.multiply(mask[i], slice_vel[i]), vmin = min_v[vel], vmax = max_v[vel])
                            plt.axis('off')
                            plt.savefig(f'{eval_dir}/{patient}_{vel}_Invivo_Original_Frame{t}_OnlyFluid.png', bbox_inches='tight')
                        
                in_vivo_original[f'mean_speed'] = calculate_mean_speed(in_vivo_original["u_fluid"], in_vivo_original["v_fluid"] , in_vivo_original["w_fluid"], in_vivo_original["mask"])
                        
                if save_slices_individual:
                    for mag in mag_colnames: 
                        slice_vel = np.array(p1[mag])[idx_slice]
                        t_range = np.arange(frame_start, frame_end)
                        # for i, t in enumerate(t_range):
                        i = 0
                        t = t_range[0]
                        plt.imshow(slice_vel[i], cmap ='Greys_r')
                        plt.axis('off')
                        plt.savefig(f'{eval_dir}/{patient}_{mag}_Invivo_OriginalMagnitude_Frame{t}.png', bbox_inches='tight')
                        plt.show()
                        
                        plt.imshow(mask[i], cmap ='Greys_r')
                        plt.axis('off')
                        plt.savefig(f'{eval_dir}/{patient}_{mag}_Invivo_Mask_Frame{t}.png', bbox_inches='tight')
                        plt.show()

                        plt.imshow(np.multiply(slice_vel, mask)[i], cmap ='Greys_r')
                        plt.axis('off')
                        plt.savefig(f'{eval_dir}/{patient}_{mag}_Invivo_OriginalMagnitudeXMask_Frame{t}.png', bbox_inches='tight')
                        plt.show()


        # load in-vivo data
        for p, name_pred in zip(prediction_files, name_prediction):
            in_vivo = f'{pred_dir}/{p}'
            print('--------------', p, '------------------')
            with h5py.File(in_vivo, mode = 'r' ) as p1:
                for vel in vel_colnames:
                    in_vivo_predicted[f'{vel}_{name_pred}'] = np.array(p1[f'{vel}_combined'])

                    if save_slices_individual:
                        slice_vel = np.array(p1[f'{vel}_combined'])[idx_slice]
                        t_range = np.arange(frame_start, frame_end)
                        for i, t in enumerate(t_range):
                            plt.imshow(slice_vel[i], vmin = min_v[vel], vmax = max_v[vel])
                            plt.axis('off')
                            plt.savefig(f'{eval_dir}/{patient}_{vel}_{name_pred}_Frame{t}.png', bbox_inches='tight')
                            plt.show()

                # calculate mean speed
                print(in_vivo_predicted.keys())
                in_vivo_predicted[f'mean_speed_{name_pred}'] = calculate_mean_speed(in_vivo_predicted[f"u_{name_pred}"], in_vivo_predicted[f"v_{name_pred}"] , in_vivo_predicted[f"w_{name_pred}"], in_vivo_original["mask"])

        # plot frames slices over time for each velocity component
        for vel in vel_colnames:
            plot_slices_over_time3(in_vivo_original[vel],in_vivo_original[vel],  in_vivo_original['mask'], 0, [in_vivo_original[f'{vel}_fluid'], ] + [in_vivo_predicted[f'{vel}_{name_pred}'] for name_pred in name_prediction], ['HR fluid region'] + name_prediction,np.arange(frame_start, frame_end), idx_slice,min_v[vel], max_v[vel],exclude_rel_error = True, save_as = f"{eval_dir}/{patient}_Frame_comparison_{vel}.svg", figsize = (8,8))

        
        #-------------mean speed plot---------------------
        plt.figure(figsize=(7, 4))
        colors = [KTH_colors['blue100'], KTH_colors['lightblue100'],  KTH_colors['grey100'], KTH_colors['pink80'], KTH_colors['grey40'],]#['steelblue', 'darkorange', 'brown', 'orchid', 'darkviolet', 'olivedrab', 'lightcoral', 'maroon', 'yellow', 'seagreen']
        markers = ['solid', ':',  '--','-.',  (0, (3, 1, 1, 1)),'-.', ]
        
        step_pred = 0.5 if super_resolved_prediction else 1
        
        N_frames_input_data = in_vivo_original['u'].shape[0]
        N_frames_pred_data = in_vivo_predicted[f'u_{name_prediction[0]}'].shape[0]

        frame_range_input = np.arange(0, N_frames_input_data)#np.linspace(0, data_1['u'].shape[0]-1,  data_1['u'].shape[0])
        frame_range_predicted = np.arange(0, N_frames_input_data, step_pred)#np.linspace(0, data_1['u'].shape[0], data_predicted['u'].shape[0])
        print(frame_range_input, frame_range_predicted)
        plt.title('Mean speed')
        plt.plot(frame_range_input, in_vivo_original['mean_speed'],'-', label = 'noisy input data', color= 'black')

        for name_pred, c, marker in zip(name_prediction, colors, markers):
            plt.plot(frame_range_predicted, in_vivo_predicted[f'mean_speed_{name_pred}'], '.-', label = name_pred, color= c, linestyle = marker)

            print(f'Mean speed {name_pred}:', np.average(in_vivo_predicted[f'mean_speed_{name_pred}']))
            print(f'Max speed {name_pred}:', np.max(in_vivo_predicted[f'mean_speed_{name_pred}']))
            print(f'Min speed {name_pred}:', np.min(in_vivo_predicted[f'mean_speed_{name_pred}']))
            # print(f'Mean speed {name_pred} in peak flow frame:', in_vivo_predicted[f'mean_speed_{name_pred}'][peak_flow_frame])
            print(f'Mean difference {name_pred}:', np.average(in_vivo_predicted[f'mean_speed_{name_pred}'] - in_vivo_original['mean_speed']))
            print(f'Max difference {name_pred}:', np.max(np.abs(in_vivo_predicted[f'mean_speed_{name_pred}'] - in_vivo_original['mean_speed'])))

        if not super_resolved_prediction:
            plt.plot(frame_range_input[::2], in_vivo_original['mean_speed'][::2],'.',  label ='sample points',  color= 'black')
        else:
            plt.plot(frame_range_input, in_vivo_original['mean_speed'],'.',  label ='sample points',  color= 'black')
        plt.xlabel("Frame")
        plt.ylabel("Mean speed (cm/s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{eval_dir}/{patient}_Meanspeed_{name_comparison}.svg')
        plt.show()

        #------- k value plot and R^2 plots -------------------
        #---------------correlation + k + r^2 plots -------------------
        if not super_resolved_prediction:
            # correlation
            peak_flow_frame = np.argmax(in_vivo_original['mean_speed'])
            frame_corr_plot = peak_flow_frame.copy()
            if frame_corr_plot % 2 == 0: frame_corr_plot +=1 # take next frame if peak flow frame included in lr data
            plt.figure(figsize=(15, 5))
            # plot_correlation(data_original, data_predicted, bounds=bounds, frame_idx=frame_corr_plot, save_as=f'{eval_dir}/Correlation_frame{frame_corr_plot}_{name_evaluation}.svg')
            
            print('Peak flow frame:', peak_flow_frame)
            frames = in_vivo_original['mean_speed'].shape[0]
           
            vel_plotname = [r'$V_x$', r'$V_y$', r'$V_z$']
                
            
            # save each plot separately
            plt.figure(figsize=(5, 5))
            for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
                plt.clf()
                for name_pred, c, marker in zip(name_prediction, colors, markers):
                    k, r2, = np.zeros(frames), np.zeros(frames)
                    for t in range(frames):
                        k[t], r2[t]  = calculate_k_R2( in_vivo_predicted[f'{vel}_{name_pred}'][t], in_vivo_original[vel][t], in_vivo_original['mask'][t])
                    
                    
                    plt.plot(range(frames), k , label = f'{name_pred}', color = c, linestyle = marker)
                    plt.scatter(np.ones(1)*peak_flow_frame, k[peak_flow_frame] , color = KTH_colors['grey80'])
                plt.legend(loc = 'upper right')
                plt.title(title)
                plt.xlabel('frame')
                plt.ylabel('k')
                plt.plot(np.ones(frames), 'k:')
                plt.ylim([0.05, 1.05])
                plt.savefig(f'{eval_dir}/{name_comparison}_k_vals_{vel}_.svg', bbox_inches='tight')
