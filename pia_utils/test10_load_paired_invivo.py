import numpy as np 
import h5py
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':

    # invivo_filenames = ['Volunteer3_4D_WholeHeart_2mm_20ms.h5']
    

    data_dir = '../Temporal4DFlowNet/data/paired_invivo/' #Temporal4DFlowNet/data/paired_invivo/Volunteer3_4D_WholeHeart_2mm_20ms.h5

    # read all filenames from directory
    # invivo_filenames = 
    invivo_filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]


    for invivo_filename in invivo_filenames: 
        
        print(f'--------TESTING model {invivo_filename}--------')


        # Load the lr file
        with h5py.File(f'{data_dir}/{invivo_filename}', 'r+') as f:
            print(f.keys())
            for key in f.keys():
                print(key, f[key].shape)

            if f['u_max'][0] < 100:
                print('Multiplying by 100')
                f['u_max'][()] *= 100
                f['v_max'][()] *= 100
                f['w_max'][()] *= 100

            print('vencs:', np.array(f['u_max']))
            print('vencs:', np.array(f['v_max']))
            print('vencs:', np.array(f['w_max']))
            


        





