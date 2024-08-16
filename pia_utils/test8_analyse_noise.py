import numpy as np 
import h5py

def noise_ratio(vel_orig, vel_noise):
    """
    Compute the noise ratio between the original velocity field and the noisy one
    """
    eps = 1e-6
    return np.abs(vel_orig) / (np.abs(vel_orig - vel_noise) + eps)



if __name__ == '__main__':

    data_models = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    invivo_magn = ['P01', 'P04', 'P03', 'P02', 'P05', 'P03']

    data_dir = 'data/CARDIAC'


    for model, imagn in zip(data_models, invivo_magn): 
        
        print(f'--------TESTING model {model}--------')
        hr_filename =  f'{model}_2mm_step2_cs_invivo{imagn}_hr.h5'
        lr_filename =  f'{model}_2mm_step2_cs_invivo{imagn}_lr_lessnoise.h5'

        # Load the lr file
        with h5py.File(f'{data_dir}/{lr_filename}', 'r') as f:
            lr_u = np.array(f['u'])[2::]  
            lr_v = np.array(f['v'])[2::]
            lr_w = np.array(f['w'])[2::]
            mag = np.array(f['mag_u'])[2::]
            mask_lr = np.array(f['mask'])[2::]

            if 'u_max' in f.keys():
                venc_u = np.unique(np.array(f['u_max']))
                venc_v = np.unique(np.array(f['v_max']))
                venc_w = np.unique(np.array(f['w_max']))
            else:
                venc_u = np.unique(np.array(f['venc_u']))
                venc_v = np.unique(np.array(f['venc_v']))
                venc_w = np.unique(np.array(f['venc_w']))

        # Load the hr file
        with h5py.File(f'{data_dir}/{hr_filename}', 'r') as f:
            hr_u = np.array(f['u'])[4::]
            hr_v = np.array(f['v'])[4::]
            hr_w = np.array(f['w'])[4::]
            mask = np.array(f['mask'])[4::]


        min_u = np.quantile(hr_u[np.where(mask !=0)].flatten(), 0.01)
        max_u = np.quantile(hr_u[np.where(mask !=0)].flatten(), 0.99)

        u_noise_ratio = noise_ratio(hr_u[::2], lr_u)
        v_noise_ratio = noise_ratio(hr_v[::2], lr_v)
        w_noise_ratio = noise_ratio(hr_w[::2], lr_w)

        #noise
        avg_noise_u = np.mean(u_noise_ratio, where = mask_lr !=0)
        avg_noise_v = np.mean(v_noise_ratio, where = mask_lr !=0)
        avg_noise_w = np.mean(w_noise_ratio, where = mask_lr !=0)

        min_noise_u = np.min(u_noise_ratio[np.where(mask_lr  !=0)])
        min_noise_v = np.min(v_noise_ratio[np.where(mask_lr  !=0)])
        min_noise_w = np.min(w_noise_ratio[np.where(mask_lr  !=0)])

        abs_diff_u = np.abs(hr_u[::2] - lr_u)[np.where(u_noise_ratio<1)]
        abs_diff_v = np.abs(hr_v[::2] - lr_v)[np.where(v_noise_ratio<1)]
        abs_diff_w = np.abs(hr_w[::2] - lr_w)[np.where(w_noise_ratio<1)]


        # magnitude within mask
        mag = mask_lr*mag

        print('VENC u/v/w: ', venc_u, venc_v, venc_w)
        print('NOISE RATIO u/v/w: ', avg_noise_u, avg_noise_v, avg_noise_w)
        print('MIN NOISE RATIO: ', min_noise_u, min_noise_v, min_noise_w)
        print('ABS DIFF: ', abs_diff_u.mean(),  abs_diff_v.mean(),  abs_diff_w.mean(), )
        print('MAGNITUDE avg, std, max, min: ', np.mean(mag, where=mask_lr !=0), np.std(mag, where=mask_lr !=0), np.max(mag), np.min(mag))


        





