import numpy as np 
import h5py


if __name__ == '__main__':

    invivo_magn = ['P01', 'P04', 'P03', 'P02', 'P05', 'P03']

    data_dir = 'data/PIA/THORAX'


    for imodel in invivo_magn: 
        
        print(f'--------TESTING model {imodel}--------')
        invivo_filename =  f'{imodel}/h5/{imodel}.h5'

        # Load the lr file
        with h5py.File(f'{data_dir}/{invivo_filename}', 'r') as f:
            print(f.keys())
            print('venc u :', np.unique(np.array(f['u_max']) ))
            print('venc v :', np.unique(np.array(f['v_max']) ))
            print('venc w :', np.unique(np.array(f['w_max']) ))


        





