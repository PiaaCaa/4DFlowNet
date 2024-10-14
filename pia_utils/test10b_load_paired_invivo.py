import numpy as np 
import h5py


if __name__ == '__main__':

    invivo_filenames = ['v3_wholeheart_25mm_40ms.h5', 'v3_wholeheart_25mm_20ms.h5',
                        'v4_wholeheart_25mm_40ms.h5', 
                        'v5_wholeheart_25mm_40ms.h5', 'v5_wholeheart_25mm_20ms.h5',
                        'v6_wholeheart_25mm_40ms.h5', 'v6_wholeheart_25mm_20ms.h5',
                        'v7_wholeheart_25mm_40ms.h5', 'v7_wholeheart_25mm_20ms.h5',]

    data_dir = 'data/paired_invivo'


    for invivo_filename in invivo_filenames: 
        
        print(f'--------TESTING model {invivo_filename}--------')

        # Load the lr file
        with h5py.File(f'{data_dir}/{invivo_filename}', 'r') as f:
            print(f.keys())
            # for key in f.keys():
            #     print(key, f[key].shape)

            print('vencs:', np.array(f['u_max'][0]))
            print('vencs:', np.array(f['v_max'][0]))
            print('vencs:', np.array(f['w_max'][0]))
            


        





