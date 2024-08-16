import numpy as np 
import h5py


if __name__ == '__main__':

    invivo_filenames = ['Volunteer3_4D_WholeHeart_2mm_20ms.h5']

    data_dir = 'data/paired_invivo'


    for invivo_filename in invivo_filenames: 
        
        print(f'--------TESTING model {invivo_filename}--------')

        # Load the lr file
        with h5py.File(f'{data_dir}/{invivo_filename}', 'r') as f:
            print(f.keys())
            for key in f.keys():
                print(key, f[key].shape)

            print('vencs:', np.array(f['u_max']))
            print('vencs:', np.array(f['v_max']))
            print('vencs:', np.array(f['w_max']))
            


        





