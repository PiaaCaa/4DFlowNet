import numpy as np
import tensorflow as tf
import time
import h5py
from Network.PatchHandler3D import PatchHandler3D
from Network.PatchHandler3D_temporal import PatchHandler4D

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes


def check_compatibility(datapair):
    '''
    Test function, which checks compatibility of generated low ress and highres patches
    Be aware: Only useful when comparing patches with no added noise
    '''

    def simple_temporal_downsampling(hr_data, downsample =2):
        assert(len(hr_data.shape) == 4), hr_data.shape # assume that data is of form b,  t,  w, d
        if downsample ==2:
            lr_data = hr_data[:, ::2,  :, :]
            return lr_data
        else:
            print("Only implemented for downsampling by 2, please implement if needed.")
    

    lr_u, lr_v, lr_w = datapair[:3]
    hr_u, hr_v, hr_w = datapair[6:9]

    hr_u_downsampled = simple_temporal_downsampling(np.asarray(hr_u)[:, :, :, 0], downsample=2)
    hr_v_downsampled = simple_temporal_downsampling(np.asarray(hr_v)[:, :, :, 0], downsample=2)
    hr_w_downsampled = simple_temporal_downsampling(np.asarray(hr_w)[:, :, :, 0], downsample=2)

    # check u 
    if np.linalg.norm(np.asarray(lr_u)[:, :, :, 0] - hr_u_downsampled) != 0:
        print("LR u is not compatible with downsampled high res image!! ")
        print('Norm of difference: ', np.linalg.norm(np.asarray(lr_u)[:, :, :, 0] - hr_u_downsampled))
        

    #check v
    if np.linalg.norm(np.asarray(lr_v)[:, :, :, 0] - hr_v_downsampled) != 0:
        print("LR v is not compatible with downsampled high res image!! ")
        print('Norm of difference: ', np.linalg.norm(np.asarray(lr_v)[:-1] - hr_v_downsampled))
    
    #check w
    if np.linalg.norm(np.asarray(lr_w)[:, :, :, 0] - hr_w_downsampled) != 0:
        print("LR w is not compatible with downsampled high res image!! ")
        print('Norm of difference: ', np.linalg.norm(np.asarray(lr_w)[:, :, :, 0] - hr_w_downsampled))



if __name__ == "__main__":
    data_dir = 'Temporal4DFlowNet/data/CARDIAC'
    
    # ---- Patch index files ----
    training_file = '{}/Temporal10MODEL1_2_no_noise.csv'.format(data_dir)
   
    # Hyperparameters optimisation variables
    epochs =  1
    batch_size =2

    patch_size = 10
    res_increase = 2
    

    # Load data file and indexes
    trainset = load_indexes(training_file)
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    z = PatchHandler4D(data_dir, patch_size, res_increase, batch_size)
    trainset = z.initialize_dataset(trainset, shuffle=True, n_parallel=2)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        start_time = time.time()
        for i, data_pairs in enumerate(trainset):
            start_loop = time.time()
            
            a = data_pairs
            #check if datapairs align
            check_compatibility(a)
            message = f"Iteration {i+1}   - batch {time.time()-start_loop:.4f} sec {time.time()-start_time:.1f} secs"
            print(f"\r{message}", end='')
            
    print("\nDone")
    