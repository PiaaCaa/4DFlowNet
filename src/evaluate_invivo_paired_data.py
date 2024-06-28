import numpy as np
import time
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from utils.evaluate_utils import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
from prepare_data.h5functions import save_to_h5
import matplotlib
from utils.colors import *
import matplotlib.animation as animation
plt.rcParams['figure.figsize'] = [10, 8]



if __name__ == "__main__":
    # for one network evluation on multiple invivo datasets
    if True:
        # set directories 
        input_dir = 'data/PairedInvivo/'
        res_dir   = 'results/in_vivo/'
        eval_dir  = 'results/in_vivo/plots/PairedInvivo/20240605-1504'

        plot_animation = True
        plt_meanspeed = False

        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir)
        
        #
        dict_results = defaultdict(list)
        cases = ['SR/Volunteer3_4D_WholeHeart_2mm_40ms/Volunteer3_4D_WholeHeart_2mm_40ms_20240605-1504_50Frames.h5' ]#, 'P03', 'P04', 'P05'] 
        lr_filenames = ['Volunteer3_4D_WholeHeart_2mm_40ms.h5']
        hr_filenames = ['Volunteer3_4D_WholeHeart_2mm_20ms.h5']
        for c,  hr_filename, lr_filename in zip(cases, hr_filenames, lr_filenames):
            print('-------------------', c, '-------------------')

            if not os.path.exists(f'{eval_dir}'):
                os.makedirs(f'{eval_dir}')

            path_invivo_hr = f'{input_dir}/{hr_filename}'
            path_invivo_lr = f'{input_dir}/{lr_filename}'
            in_vivo_upsampled = f'{res_dir}/{c}' 
            name_evaluation = f'PAIRED_INVIVO_{c}_{os.path.basename(path_invivo_hr)[:-3]}'

            #set slice index for animation
            idx_slice = np.index_exp[:, :, 25]

            data_hr = {}
            data_lr = {}
            data_predicted = {}
            vencs = {}
            vel_colnames = ['u', 'v','w']
            mag_colnames = ['mag_u', 'mag_v', 'mag_w']
            venc_colnames = [  'u_max', 'v_max', 'w_max'] #['venc_u', 'venc_v', 'venc_w']#
            vel_plotnames = [r'$V_x$', r'$V_y$', r'$V_z$']
            mag_plotnames = [r'$M_x$', r'$M_y$', r'$M_z$']


            # load HR in-vivo data
            with h5py.File(path_invivo_hr, mode = 'r' ) as p1:

                for vel, venc, mag in zip(vel_colnames, venc_colnames, mag_colnames):
                    vencs[venc] = np.asarray(p1[venc])
                    data_hr[vel] = np.asarray(p1[vel], dtype = float).squeeze().transpose(3, 0, 1, 2)
                    data_hr[mag] =  np.asarray(p1[mag]).squeeze().transpose(3, 0, 1, 2) 
                print('VENCS HR:', p1['u_max'][0], p1['v_max'][0], p1['w_max'][0])
            
            with h5py.File(path_invivo_lr, mode = 'r' ) as p1:
                for vel in vel_colnames:
                    data_lr[vel] = np.asarray(p1[vel], dtype = float).squeeze().transpose(3, 0, 1, 2)
                print('VENCS LR:', p1['u_max'][0], p1['v_max'][0], p1['w_max'][0])
                

            # load prediction
            with h5py.File(in_vivo_upsampled, mode = 'r' ) as h_pred:
                for vel, venc in zip(vel_colnames, venc_colnames):
                    data_predicted[vel] = np.asarray(h_pred[f'{vel}_combined'])

            print('Shape of predicted data and original data:', data_predicted['u'].shape, data_hr['u'].shape)
            N_frames = data_hr['u'].shape[0]

            N_frames_input_data = data_hr['u'].shape[0]
            N_frames_pred_data = data_predicted['u'].shape[0]

            super_resolved_prediction = False if N_frames_input_data == N_frames_pred_data else True
            if super_resolved_prediction: print('Evaluation of higher resolved velocity field')
            if super_resolved_prediction: print('Prediction increases temporal resolution of original data by 2x. (super resolved) ..')

            #find lower and higher values to display velocity fields
            min_v = {}
            max_v = {}
            t, x, y, z = data_hr['u'].shape
            for vel in vel_colnames:
                
                min_v[vel] = np.quantile(data_hr[vel][:, x//4:x//4 + x//2, y//4:y//4 + y//2, z//4:z//4 + z//2 ].flatten(), 0.01)
                max_v[vel] = np.quantile(data_hr[vel][:, x//4:x//4 + x//2, y//4:y//4 + y//2, z//4:z//4 + z//2 ].flatten(), 0.99)
                print(min_v[vel], max_v[vel])

            max_V = np.max([max_v['u'], max_v['v'], max_v['w']])
            min_V = np.min([min_v['u'], min_v['v'], min_v['w']])

            #-----------------save img slices over time---------------------

            #--------------------calculate mean speed --------------------------
            magn = np.sqrt(data_hr['mag_u']**2 + data_hr['mag_v']**2 + data_hr['mag_w']**2)
            speed = np.sqrt(data_hr['u']**2 + data_hr['v']**2 + data_hr['w']**2)
            pc_mri = np.multiply(magn, speed)



            if plot_animation:
                fps_anim = 10
                fps_pred = fps_anim
                if True: 
                    if not os.path.exists(f'{eval_dir}/Animate_invivo_case00{c}_mag_{fps_anim}fps.gif'):
                        animate_data_over_time_gif(idx_slice, magn, 0, np.quantile(magn, 0.99),      fps = fps_anim , save_as = f'{eval_dir}/{os.path.basename(c)}_animate_mag', colormap='Greys_r' )
                        animate_data_over_time_gif(idx_slice, data_hr['u'], min_v['u'], max_v['u'],  fps = fps_anim , save_as = f'{eval_dir}/{os.path.basename(c)}_animate_u_gt')
                        animate_data_over_time_gif(idx_slice, data_hr['v'], min_v['v'], max_v['v'],  fps = fps_anim , save_as = f'{eval_dir}/{os.path.basename(c)}_animate_v_gt')
                        animate_data_over_time_gif(idx_slice, data_hr['w'], min_v['w'], max_v['w'],  fps = fps_anim , save_as = f'{eval_dir}/{os.path.basename(c)}_animate_w_gt')
                        animate_data_over_time_gif(idx_slice, data_predicted['u'], min_v['u'], max_v['u'], fps = fps_pred , save_as = f'{eval_dir}/{os.path.basename(c)}_animate_u_pred')
                        animate_data_over_time_gif(idx_slice, data_predicted['v'], min_v['v'], max_v['v'], fps = fps_pred , save_as = f'{eval_dir}/{os.path.basename(c)}_animate_v_pred')
                        animate_data_over_time_gif(idx_slice, data_predicted['w'], min_v['w'], max_v['w'], fps = fps_pred , save_as = f'{eval_dir}/{os.path.basename(c)}_animate_w_pred')
                        animate_data_over_time_gif(idx_slice, data_lr['u'], min_v['u'], max_v['u'],  fps = fps_anim//2 , save_as = f'{eval_dir}/{os.path.basename(c)}_animate_u_lr')
                        animate_data_over_time_gif(idx_slice, data_lr['v'], min_v['v'], max_v['v'],  fps = fps_anim//2 , save_as = f'{eval_dir}/{os.path.basename(c)}_animate_v_lr')
                        animate_data_over_time_gif(idx_slice, data_lr['w'], min_v['w'], max_v['w'],  fps = fps_anim//2 , save_as = f'{eval_dir}/{os.path.basename(c)}_animate_w_lr')


                if False: 
                    animate_data_over_time_gif(idx_slice, data_hr['mask'], 0, 1,         eval_dir, fps = fps_anim , title = f'{c}_mask', colormap='Greys' )
                    animate_data_over_time_gif(idx_slice, data_predicted['u'], min_v['u'], max_v['u'], eval_dir, fps = fps_pred , title = f'{name_evaluation}_u_pred')
                    animate_data_over_time_gif(idx_slice, data_predicted['v'], min_v['v'], max_v['v'], eval_dir, fps = fps_pred , title = f'{name_evaluation}_v_pred')
                    animate_data_over_time_gif(idx_slice, data_predicted['w'], min_v['w'], max_v['w'], eval_dir, fps = fps_pred , title = f'{name_evaluation}_w_pred')

                #     animate_invivo_HR_pred(idx, v_orig, v_gt_fluid, v_pred, min_v, max_v, save_as, fps = 10)
                # animate_invivo_HR_pred(idx_slice, data_original['u'][::2], hr, pred, vel,min_v, max_v, save_as, fps = 10)


    #-------------------------------------------------------------------------------------------------------------------------
