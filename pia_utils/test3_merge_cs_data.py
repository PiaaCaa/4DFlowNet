
import numpy as np
import h5py
import os
from prepare_data import h5functions


if __name__ == '__main__':


    
    # m = 'M1'
    for m in ['M6']:
        vel_colnames = ['u', 'v', 'w', 'mask']
        mag_colnames = ['mag_u', 'mag_v', 'mag_w']

        data_dir = 'data/CARDIAC'
        path_cloudmagn = 'data/cloud_magn_data_4D_spatial_rotated_M1-M6.h5'

        orig_hr = f'{data_dir}/{m}_2mm_step2_static_dynamic.h5'
        orig_lr = f'{data_dir}/{m}_2mm_step2_cloudmagnRot_toeger_LRfct_noise.h5'

        hr_cs_path = f'{data_dir}/{m}_2mm_step2_dynamic_cs_20ms.h5'
        lr_cs_path = f'{data_dir}/{m}_2mm_step2_dynamic_cs_40ms_noise.h5'
        

        # 1.  HR data
        
        h5functions.merge_data_to_h5(hr_cs_path, orig_hr)

        # now delete magnitude data from the HR file
        h5functions.delete_data_from_h5(hr_cs_path, mag_colnames)

        # add the cloudmagn data to the HR file
        with h5py.File(hr_cs_path, 'a') as f:
            with h5py.File(path_cloudmagn, 'r') as f_cloud:
                for mag_col in mag_colnames:
                    f.create_dataset(mag_col, data=f_cloud[m])

            print(f'HR data saved to {hr_cs_path}, now contains keys: {list(f.keys())}')

        
        # 2.  LR data
        with h5py.File(lr_cs_path, mode='a') as input_h5:
            with h5py.File(orig_lr, mode='r') as toadd_h5:
                for key in toadd_h5.keys():
                    if key not in input_h5.keys():
                        print('Adding key', key)
                        dataset = np.array(toadd_h5[key]).squeeze()

                        # convert float64 to float32 to save space
                        if dataset.dtype == 'float64':
                            dataset = np.array(dataset, dtype='float32')

                        if dataset.shape[0] > input_h5['u'].shape[0]:
                            print('Downsampling', key, 'from', dataset.shape[0], 'to', input_h5['u'].shape[0])
                            dataset = dataset[::2]

                        input_h5.create_dataset(key, data=dataset)

        # now delete magnitude data from the LR file
        h5functions.delete_data_from_h5(lr_cs_path, mag_colnames)

        # add the cloudmagn data to the LR file
        with h5py.File(lr_cs_path, 'a') as f:
            with h5py.File(path_cloudmagn, 'r') as f_cloud:
                for mag_col in mag_colnames:
                    magn_data = np.array(f_cloud[m])
                    if magn_data.shape[0] > f['u'].shape[0]:
                        magn_data = magn_data[::2]
                        print('Downsampling', mag_col, 'from', f_cloud[m].shape[0], 'to', magn_data.shape[0])

                    f.create_dataset(mag_col, data=magn_data)

            print(f'LR data saved to {lr_cs_path}, now contains keys: {list(f.keys())}')

        



