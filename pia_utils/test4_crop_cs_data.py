
import numpy as np
import h5py
import os
from prepare_data import h5functions


if __name__ == '__main__':
    print('Create new dataset by cropping first two frames in LR and four frames in HR to not include t sampling artifacts..')

    data_dir = 'data/CARDIAC'

    lr_models =   ['M4_2mm_step2_cs_invivoP02_lr_lessnoise.h5', 'M1_2mm_step2_cs_invivoP01_lr_lessnoise.h5', 
                  'M2_2mm_step2_cs_invivoP04_lr_lessnoise.h5', 'M3_2mm_step2_cs_invivoP03_lr_lessnoise.h5', 
                  'M5_2mm_step2_cs_invivoP05_lr_lessnoise.h5', 'M6_2mm_step2_cs_invivoP03_lr_lessnoise.h5']

    hr_models =   ['M4_2mm_step2_cs_invivoP02_hr.h5', 'M1_2mm_step2_cs_invivoP01_hr.h5',
                    'M2_2mm_step2_cs_invivoP04_hr.h5', 'M3_2mm_step2_cs_invivoP03_hr.h5',
                    'M5_2mm_step2_cs_invivoP05_hr.h5', 'M6_2mm_step2_cs_invivoP03_hr.h5']

    
    # m = 'M1'
    for file_lr, file_hr in zip(lr_models, hr_models):
        print('Cropping models', file_lr, file_hr)
        crop_colnames_hr = ['u', 'v', 'w', 'mask', 'mag_u', 'mag_v', 'mag_w', 'u_max', 'v_max', 'w_max']
        crop_colnames_lr = ['u', 'v', 'w', 'mask', 'mag_u', 'mag_v', 'mag_w', 'venc_u', 'venc_v', 'venc_w']

        
        output_lr = f'{file_hr[:-3]}' + '_cropped.h5'
        output_hr = f'{file_lr[:-3]}' + '_cropped.h5'

        # 1.  HR data
        with h5py.File(os.path.join(data_dir, file_hr), 'r') as f:
            for key in crop_colnames_hr:
                h5functions.save_to_h5(f'{data_dir}/{output_hr}', key, np.array(f[key])[4::], expand_dims = False)
                print('HR ',key, ' new shape: ' , np.array(f[key])[4::].shape)
            
            for key in f.keys():
                if key not in crop_colnames_hr:
                    print('Saving:', key)
                    h5functions.save_to_h5(f'{data_dir}/{output_hr}', key, np.array(f[key]), expand_dims = False)
        
        # 1.  LR data
        with h5py.File(os.path.join(data_dir, file_lr), 'r') as f:
            for key in crop_colnames_lr:
                h5functions.save_to_h5(f'{data_dir}/{output_lr}', key, np.array(f[key])[2::], expand_dims = False)
                print(np.array(f[key])[2::].shape)
            
            for key in f.keys():
                if key not in crop_colnames_lr:
                    print('Saving:', key)
                    h5functions.save_to_h5(f'{data_dir}/{output_lr}', key, np.array(f[key]), expand_dims = False)
            

