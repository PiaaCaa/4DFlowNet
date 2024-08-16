# libraries 
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../src')

# from Network.PatchHandler3D import PatchHandler3D
# functions
import prepare_data.fft_downsampling as fft_fcts
from utils.evaluate_utils import *
from prepare_data.h5functions import *




if __name__ == '__main__':
    
    # this script is there to:
    # 1. Take the new magnitude data, extend it to fit the insilico size by mirroring 
    # 2. load the in-silico model which fits the invivo magnitude 
    # 3. and put this into a new dataformat and save this as h5 file
    # note: test inbetween that magnitude and velocity image are of same size and shape
    mag_colnames  = ['mag_u', 'mag_v', 'mag_w']

    data_dir = 'data/CARDIAC'

    models = [ 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    name_invivomagn = ['M1_2mm_step2_cs_invivoP01_lr.h5', 
                       'M2_2mm_step2_cs_invivoP04_lr.h5',
    ]
    # this is now for lr resolution data

    colnames = ['u', 'v', 'w', 'mask', 'mag_u', 'mag_v', 'mag_w']
    venc_colnames = ['u_max', 'v_max', 'w_max']

    for model, p_invivo in zip(models, name_invivomagn): 
        
        path_invivomagn = f'{data_dir}/{p_invivo}'
        path_new = f'{path_invivomagn[:-3]}_50frames.h5'

        new_data = {}
        # load magnitude data
        with h5py.File(path_invivomagn, 'r') as h5:

            print(h5.keys())
            
            for v in colnames:
                print(np.array(h5[v]).shape) 
                data = np.array(h5[v]).squeeze()
                new_data[v] = np.zeros((50, *data.shape[1:]))
                new_data[v][::2, :, :, :] = data
                new_data[v][1::2, :, :, :] = data
            
            for venc in venc_colnames:
                data = np.array(h5[venc]).squeeze()
                new_data[venc] = np.zeros(50)
                new_data[venc][::2] = data
                new_data[venc][1::2] = data

        # save new data
        for v in new_data.keys():
            print('Save new data to:', v, path_new)
            print(new_data[v].shape)
            save_to_h5(path_new,  v, new_data[v], expand_dims = False)





        


   