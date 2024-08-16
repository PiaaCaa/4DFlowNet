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

from utils.colors import *
import prepare_data.cfl as cfl
from prepare_data.h5functions import *




if __name__ == '__main__':
    
    # this script is there to:
    # 1. Take the new magnitude data, extend it to fit the insilico size by mirroring 
    # 2. load the in-silico model which fits the invivo magnitude 
    # 3. and put this into a new dataformat and save this as h5 file
    # note: test inbetween that magnitude and velocity image are of same size and shape
    mag_colnames  = ['mag_u', 'mag_v', 'mag_w']

    data_dir = '/mnt/c/Users/piacal/Code/SuperResolution4DFlowMRI/Temporal4DFlowNet/data/CARDIAC'

    models = [ 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    name_invivomagn = ['M1_2mm_step2_invivoP01_magnitude.h5', 
                       'M2_2mm_step2_invivoP04_magnitude.h5',
                       'M3_2mm_step2_invivoP03_magnitude.h5',
                       'M4_2mm_step2_invivoP02_magnitude.h5',
                       'M5_2mm_step2_invivoP05_magnitude.h5',
                       'M6_2mm_step2_invivoP03_magnitude.h5',
    ]
    # this is now for high resolution data

    # M6_2mm_step2_invivoP05_magnitude
    for model, p_invivo in zip(models, name_invivomagn): 
        
        path_datamodel = f'{data_dir}/{model}_2mm_step2_static_dynamic.h5'
        path_invivomagn = f'{data_dir}/{p_invivo}'
        path_newdata = f'{data_dir}/{model}_2mm_step2_cs_{p_invivo.split("_")[-2]}_hr.h5'

        # get original spatial shape
        with h5py.File(path_datamodel, 'r') as h5:
            _, x, y, z = np.array(h5['u']).shape

        # load magnitude data
        with h5py.File(path_invivomagn, 'r') as h5:
            mag_u = np.array(h5['mag_u'])
            mag_v = np.array(h5['mag_v'])
            mag_w = np.array(h5['mag_w'])

        
        # check if shape is same
        # 1. padding with reflection only on right side (since orgiginal data has been cropped on right side)
        mag_u = np.pad(mag_u, ((0,0), (0, x-mag_u.shape[1]), (0, y-mag_u.shape[2]), (0, z-mag_u.shape[3])), mode='reflect')
        mag_v = np.pad(mag_v, ((0,0), (0, x-mag_v.shape[1]), (0, y-mag_v.shape[2]), (0, z-mag_v.shape[3])), mode='reflect')
        mag_w = np.pad(mag_w, ((0,0), (0, x-mag_w.shape[1]), (0, y-mag_w.shape[2]), (0, z-mag_w.shape[3])), mode='reflect')

        # load velocity data and combine in a new h5 file
        with h5py.File(path_datamodel, 'r') as h5:
            u = np.array(h5['u'])
            v = np.array(h5['v'])
            w = np.array(h5['w'])
            mask = np.array(h5['mask'])
            venc_u = np.max(np.array(h5[f'u_max']))
            venc_v = np.max(np.array(h5[f'v_max']))
            venc_w = np.max(np.array(h5[f'w_max']))
            venc_max = np.max([venc_u, venc_v, venc_w])
            dx = np.array(h5['dx'])

        print('shapes', u.shape, mask.shape, mag_u.shape)

        # check if shape is same
        assert u.shape == mag_u.shape
        assert mask.shape == mag_u.shape
        assert u.shape == mask.shape

        if not os.path.isfile(path_newdata):

            print('Save new data..')
            # save new data
            save_to_h5(path_newdata, 'u', u, expand_dims= False)
            save_to_h5(path_newdata, 'v', v, expand_dims= False)
            save_to_h5(path_newdata, 'w', w, expand_dims= False)
            
            save_to_h5(path_newdata, 'mag_u', mag_u, expand_dims= False)
            save_to_h5(path_newdata, 'mag_v', mag_v, expand_dims= False)
            save_to_h5(path_newdata, 'mag_w', mag_w, expand_dims= False)

            # data from original model
            save_to_h5(path_newdata, 'mask',  mask, expand_dims= False)
            save_to_h5(path_newdata, 'venc_max', venc_max, expand_dims= True)
            save_to_h5(path_newdata, 'u_max', np.repeat(venc_max, u.shape[0]), expand_dims= True)
            save_to_h5(path_newdata, 'v_max', np.repeat(venc_max, u.shape[0]), expand_dims= True)
            save_to_h5(path_newdata, 'w_max', np.repeat(venc_max, u.shape[0]), expand_dims= True)
            save_to_h5(path_newdata, 'dx', dx, expand_dims= False)

            print(f'Saved data under {path_newdata}')
        
        else:
            print(f'File {path_newdata} already exists!')
