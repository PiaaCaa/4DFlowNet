import tensorflow as tf
import numpy as np
import time
import os
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from matplotlib import pyplot as plt
import h5py
from prepare_data.visualize_utils import generate_gif_volume
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.figsize'] = [10, 8]
from utils.evaluate_utils import *
from collections import defaultdict

#TODO: make this nicer, by e.g. dount double directory creation

def load_data(model_name, set_name, data_model, step,dynamic_mask_used, ending_file= ''):
    vel_colnames=['u', 'v', 'w']
    #directories
    gt_dir = 'data/CARDIAC'
    result_dir = f'results/Temporal4DFlowNet_{model_name}'
    eval_dir = f'{result_dir}/plots'
    lr_dir = 'data/CARDIAC'
    offset = False

    inbetween_string = ''
    add_offset = ''
    if dynamic_mask_used:
        inbetween_string = '_dynamic'
    if offset:
        print('LR is now sampled with offset of 1')
        add_offset = '_offset1'
        offset_val = 1
    else:
        offset_val = 0

    #/home/pcallmer/Temporal4DFlowNet/data/CARDIAC/M4_2mm_step2_invivoP02_magnitude_adapted_noisy.h5
    #filenames
    gt_filename = f'M{data_model}_2mm_step{step}_static{inbetween_string}.h5'
    lr_filename = f'M{data_model}_2mm_step{step}_static{inbetween_string}_noise.h5' #_noise
    # gt_filename = f'M{data_model}_2mm_step{step}_invivoP02_magnitude.h5'
    # lr_filename = f'M{data_model}_2mm_step{step}_invivoP02_magnitude_noisy.h5'
    # gt_filename = f'M{data_model}_2mm_step{step}_invivoP01_magnitude.h5'
    # lr_filename = f'M{data_model}_2mm_step{step}_invivoP01_magnitude_noisy.h5'

    result_filename = f'{set_name}set_result_model{data_model}_2mm_step{step}_{model_name[-4::]}_temporal{ending_file}{add_offset}.h5' #_newpadding
    evaluation_filename = f'eval_rel_err_{data_model}_2mm_step{step}_{model_name[-4::]}_temporal.h5'

    print(gt_filename, lr_filename)

    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    #Params for evalation
    save_relative_error_file= False

    # Setting up
    gt_filepath = '{}/{}'.format(gt_dir, gt_filename)
    res_filepath = '{}/{}'.format(result_dir, result_filename)
    lr_filepath = '{}/{}'.format(lr_dir, lr_filename)
    

    if save_relative_error_file:
        assert(not os.path.exists(f'{result_dir}/{evaluation_filename}')) #STOP if relative error file is already created

    
    gt = {}
    lr = {}
    pred = {}

    with h5py.File(res_filepath, mode = 'r' ) as h_pred:
        with h5py.File(gt_filepath, mode = 'r' ) as h_gt:
            with h5py.File(lr_filepath, mode = 'r' ) as h_lr:
                
                gt["mask"] = np.asarray(h_gt["mask"]).squeeze()
                gt["mask"][np.where(gt["mask"] !=0)] = 1
                if len(gt['mask'].shape) == 4 : # check for dynamical mask, otherwise create one
                    temporal_mask = gt['mask'].copy()
                else:
                    print('Create static temporal mask for model')
                    temporal_mask = create_dynamic_mask(gt["mask"], h_gt['u'].shape[0])
                gt['mask'] = temporal_mask.copy()
                lr['mask'] = temporal_mask[offset_val::2, :, :, :].copy()
                print(gt['mask'].shape)

                # adapt dimension
                for vel in vel_colnames:
                    
                    gt[vel] = np.asarray(h_gt[vel])
                    pred[vel] = np.asarray(h_pred[f'{vel}_combined'])   
                    lr[vel] = np.asarray(h_lr[vel])[offset_val::2, :, :, :]       

                    print('pred shape', pred[vel].shape)
                    # add information considering only the fluid regions  
                    pred[f'{vel}_fluid'] =np.multiply(pred[vel], temporal_mask)
                    lr[f'{vel}_fluid'] =  np.multiply(lr[vel], lr['mask'])
                    gt[f'{vel}_fluid'] =  np.multiply(gt[vel], temporal_mask)

                    
                #include speed calculations
                gt['speed']   = np.sqrt(gt["u"]**2 + gt["v"]**2 + gt["w"]**2)
                lr['speed']   = np.sqrt(lr["u"]**2 + lr["v"]**2 + lr["w"]**2)
                pred['speed'] = np.sqrt(pred["u"]**2 + pred["v"]**2 + pred["w"]**2)

                gt['speed_fluid']   = np.multiply(gt['speed'], temporal_mask)
                lr['speed_fluid']   = np.multiply(lr['speed'], lr['mask'])
                pred['speed_fluid'] = np.multiply(pred['speed'], temporal_mask)


    return lr, gt, pred, temporal_mask, eval_dir

def load_interpolation(data_model, step, lr, gt, use_dynamical_mask):
    vel_colnames=['u', 'v', 'w']
    interpolate_NN = {}
    interpolate_linear = {}
    interpolate_cubic = {}


    inbetween_string = ''
    if use_dynamical_mask:
        inbetween_string = '_dynamic'

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



#TODO add extra ending for file

if __name__ == "__main__":


    # Define directories and filenames
    model_name = '20240226-2026' 
    set_name = 'Validation'               
    data_model= '1'
    step = 2
    use_dynamical_mask = True
    add_ending = ''
    load_interpolation_files = False

    # directories
    gt_dir = 'data/CARDIAC'
    result_dir = f'results/Temporal4DFlowNet_{model_name}'
    eval_dir = f'{result_dir}/plots'
    lr_dir = 'data/CARDIAC'
    model_dir = '/models'

    # filenames
    # gt_filename = f'M{data_model}_2mm_step{step}_static_dynamic.h5'
    # lr_filename = f'M{data_model}_2mm_step{step}_static_dynamic_noise.h5'
    gt_filename = f'M{data_model}_2mm_step2_flowermagn_boxavg_HRfct.h5'
    lr_filename = f'M{data_model}_2mm_step2_flowermagn_boxavg_LRfct_noise.h5'

    result_filename = f'{set_name}set_result_model{data_model}_2mm_step{step}_{model_name[-4::]}_temporal.h5'#newpadding.h5'
    evaluation_filename = f'eval_rel_err_{data_model}_2mm_step{step}_{model_name[-4::]}_temporal.h5'
    model_filename = f'Temporal4DFlowNet_{model_name}/Temporal4DFlowNet-best.h5'


    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    
    # Params for evalation
    save_relative_error_file= False

    # Setting up
    gt_filepath  = '{}/{}'.format(gt_dir, gt_filename)
    res_filepath = '{}/{}'.format(result_dir, result_filename)
    lr_filepath  = '{}/{}'.format(lr_dir, lr_filename)
    model_path   = '{}/{}'.format(model_dir, model_filename)

    if save_relative_error_file:
        assert(not os.path.exists(f'{result_dir}/{evaluation_filename}')) #STOP if relative error file is already created

    vel_colnames=['u', 'v', 'w']
    # gt = {}
    # lr = {}
    # pred = {}
    # dt = {}

    # #load predictions 
    # with h5py.File(res_filepath, mode = 'r' ) as h_pred:
    #     with h5py.File(gt_filepath, mode = 'r' ) as h_gt:
    #         with h5py.File(lr_filepath, mode = 'r' ) as h_lr:
                
    #             gt["mask"] = np.asarray(h_gt["mask"])
    #             gt["mask"][np.where(gt["mask"] !=0)] = 1
    #             temporal_mask = create_temporal_mask(gt["mask"], h_gt['u'].shape[0])

    #             # adapt dimension
    #             for vel in vel_colnames:
                    
    #                 gt[vel] = np.asarray(h_gt[vel])
    #                 pred[vel] = np.asarray(h_pred[f'{vel}_combined']) # TODO chnaged this with new combination of all axis 
    #                 lr[vel] = np.asarray(h_lr[vel])[::2, :, :, :] #TODO: this chnaged with the new loading modules
    #                 #transpose for temporal resolution
    #                 #pred[vel] = pred[vel].transpose(1, 0, 2, 3) #TODO changed for new csv file

    #                 pred[f'{vel}_fluid'] = np.multiply(pred[vel], temporal_mask)
    #                 lr[f'{vel}_fluid'] = np.multiply(lr[vel], temporal_mask[::2, :, :, :])
    #                 gt[f'{vel}_fluid'] = np.multiply(gt[vel], temporal_mask)
    

    lr, gt, pred, temporal_mask, eval_dir = load_data(model_name, set_name, data_model, step, use_dynamical_mask, ending_file=add_ending)
    if load_interpolation_files: interpolate_linear, interpolate_cubic, interpolate_NN = load_interpolation(data_model, step,lr, gt, use_dynamical_mask)

    # check that dimension fits
    assert(gt["u"].shape == pred["u"].shape)  ,str(pred["u"].shape) + str(gt["u"].shape) # dimensions need to be the same
    
    #calculate velocity values in 1% and 99% quantile for plotting without noise
    min_v = {}
    max_v = {}
    for vel in vel_colnames:
        min_v[vel] = np.quantile(gt[vel][np.where(temporal_mask !=0)].flatten(), 0.01)
        max_v[vel] = np.quantile(gt[vel][np.where(temporal_mask !=0)].flatten(), 0.99)


    
    T_peak_flow = np.unravel_index(np.argmax(gt["u"]), shape =gt["u"].shape)[0]
    print("Peak flow frame", T_peak_flow)
    if T_peak_flow%2==0: T_peak_flow +=1
    #mask_diff, mask_pred = compare_masks(gt["u"], gt["v"] , gt["w"], gt["mask"])
    
    #calculate relative error
    rel_error = calculate_relative_error_normalized(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
    error_pointwise, error_absolut = calculate_pointwise_error(pred["u"], pred["v"], pred["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
    if load_interpolation_files:
        rel_error_lin_interpolation =   calculate_relative_error_normalized(interpolate_linear["u"], interpolate_linear["v"], interpolate_linear["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])
        rel_error_cubic_interpolation = calculate_relative_error_normalized(interpolate_cubic["u"], interpolate_cubic["v"], interpolate_cubic["w"], gt["u"], gt["v"] , gt["w"], gt["mask"])

    error_pointwise_cap = error_pointwise.copy()
    error_pointwise_cap[np.where(error_pointwise_cap>1)] = 1
    for vel in vel_colnames:
        error_absolut[f'{vel}_fluid'] = np.multiply(error_absolut[vel], gt["mask"])

    # mean speed of gt and prediction
    mean_speed_gt =                 calculate_mean_speed(gt["u_fluid"], gt["v_fluid"] , gt["w_fluid"], gt["mask"])
    mean_speed_pred =               calculate_mean_speed(pred["u_fluid"], pred["v_fluid"] , pred["w_fluid"], gt["mask"])
    if load_interpolation_files:
        mean_speed_lin_interpolation =  calculate_mean_speed(interpolate_linear["u_fluid"], interpolate_linear["v_fluid"] ,interpolate_linear["w_fluid"], gt["mask"])
        mean_speed_cubic_interpolation = calculate_mean_speed(interpolate_cubic["u_fluid"], interpolate_cubic["v_fluid"] , interpolate_cubic["w_fluid"], gt["mask"])

    #speed 
    if load_interpolation_files:
        interpolate_linear['speed'] = np.sqrt(interpolate_linear["u"]**2 + interpolate_linear["v"]**2 + interpolate_linear["w"]**2)
        interpolate_cubic['speed'] = np.sqrt(interpolate_cubic["u"]**2 + interpolate_cubic["v"]**2 + interpolate_cubic["w"]**2)
        interpolate_linear['speed_fluid'] = np.multiply(interpolate_linear['speed'], gt['mask'])
        interpolate_cubic['speed_fluid'] = np.multiply(interpolate_cubic['speed'], gt['mask'])


    # dt["u"] = calculate_temporal_derivative(gt["u"], timestep=1)
    # dt["v"] = calculate_temporal_derivative(gt["v"], timestep=1)
    # dt["w"] = calculate_temporal_derivative(gt["w"], timestep=1)

    diastole_end = 25

    bounds, core_mask = get_boundaries(gt["mask"])
    bounds_boolmask = bounds.astype(bool)
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
            k_bounds, r2_bounds  = calculate_k_R2( pred[vel][t], gt[vel][t], bounds[t])
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

    print("Plot example time frames..")
    if load_interpolation_files:
        show_timeframes(gt["u"], lr["u"], pred["u"],gt["mask"],error_pointwise_cap ,[interpolate_linear["u"], interpolate_cubic["u"]], ["linear", "cubic"] ,timepoints=[4, 5, 6], axis=0, idx = 22,min_v = min_v["u"],max_v =max_v["u"], save_as=f'{eval_dir}/{set_name}_M{data_model}_Qual_frame_examples_VX.png')
        show_timeframes(gt["v"], lr["v"], pred["v"],gt["mask"],error_pointwise_cap ,[interpolate_linear["v"], interpolate_cubic["v"]], ["linear", "cubic"] ,timepoints=[4, 5, 6], axis=0, idx = 22,min_v = min_v["v"],max_v =max_v["v"], save_as=f'{eval_dir}/{set_name}_M{data_model}_Qual_frame_examples_VY.png')
        show_timeframes(gt["w"], lr["w"], pred["w"],gt["mask"],error_pointwise_cap ,[interpolate_linear["w"], interpolate_cubic["w"]], ["linear", "cubic"] ,timepoints=[4, 5, 6], axis=0, idx = 22,min_v = min_v["w"],max_v =max_v["w"], save_as=f'{eval_dir}/{set_name}_M{data_model}_Qual_frame_examples_VZ.png')
    else:
        show_timeframes(gt["u"], lr["u"], pred["u"],gt["mask"],error_pointwise_cap ,[], [] ,timepoints=[4, 5, 6], axis=0, idx = 22,min_v = min_v["u"],max_v =max_v["u"], save_as=f'{eval_dir}/{set_name}_M{data_model}_Qual_frame_examples_VX.png')
        show_timeframes(gt["v"], lr["v"], pred["v"],gt["mask"],error_pointwise_cap ,[], [] ,timepoints=[4, 5, 6], axis=0, idx = 22,min_v = min_v["v"],max_v =max_v["v"], save_as=f'{eval_dir}/{eval_dir}/{set_name}_M{data_model}_Qual_frame_examples_VY.png')
        show_timeframes(gt["w"], lr["w"], pred["w"],gt["mask"],error_pointwise_cap ,[], [] ,timepoints=[4, 5, 6], axis=0, idx = 22,min_v = min_v["w"],max_v =max_v["w"], save_as=f'{eval_dir}/{eval_dir}/{set_name}_M{data_model}_Qual_frame_examples_VZ.png')
  
    plt.clf()

    # show_temporal_development_line(gt["u"], interpolate_linear["u"], pred["u"],gt["mask"], axis=3, indices=(20,20), save_as=f'{eval_dir}/{set_name}_temporal_development.png')
    plt.clf()
    #evaluate where the higest shift (temporal derivative) is for each frame
    #show_quiver(gt["u"][4, :, :, :], gt["v"][4, :, :, :], gt["w"][4, :, :, :],gt["mask"], save_as=f'{result_dir}/plots/test_quiver.png')

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
    plot_correlation(gt, pred, bounds, frame_idx = T_peak_flow, save_as=f'{result_dir}/plots/{set_name}_M{data_model}_correlation_pred_frame{T_peak_flow}')
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

            

