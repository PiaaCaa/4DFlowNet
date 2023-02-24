import numpy as np
import tensorflow as tf
import time
import h5py
# from Network.PatchHandler3D import PatchHandler3D
from Network.PatchHandler3D_temporal import PatchHandler4D, PatchHandler4D_all_axis
from matplotlib import pyplot as plt

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes


def check_compatibility(datapair):
    '''
    Test function, which checks compatibility of generated low ress and highres patches
    Be aware: Norm of difference is only zero when comparing patches with no added noise
    '''

    def simple_temporal_downsampling(hr_data, downsample =2):
        if len(hr_data.shape)==3:
            hr_data = np.expand_dims(hr_data, 0)
            print("Expand dims to:", hr_data.shape)
        assert(len(hr_data.shape) == 4), hr_data.shape # assume that data is of form b, t,  w, d
        if downsample ==2:
            lr_data = hr_data[:, ::2,  :, :]
            return lr_data
        else:
            print("Only implemented for downsampling by 2, please implement if needed.")

    lr_u, lr_v, lr_w = np.asarray(datapair[:3]).squeeze()
    hr_u, hr_v, hr_w = np.asarray(datapair[6:9]).squeeze()
    mask = np.asarray(datapair[10])
    #temporal_mask[np.where(temporal_mask !=0)] = 1

    #downsample
    mask = mask[:, ::2, :, :]
    hr_u_downsampled = simple_temporal_downsampling(hr_u, downsample=2)
    hr_v_downsampled = simple_temporal_downsampling(hr_v, downsample=2)
    hr_w_downsampled = simple_temporal_downsampling(hr_w, downsample=2)

    hr_u_mask = np.ones_like(hr_u_downsampled)
    tol = 1e-8
    tol = 1e-8
    hr_u_mask[np.where(np.abs(hr_u_downsampled) < tol)] = 0
    
    hr_u_downsampled = np.multiply(hr_u_downsampled, mask )
    hr_v_downsampled = np.multiply(hr_v_downsampled, mask )
    hr_w_downsampled = np.multiply(hr_w_downsampled, mask )

    lr_u = np.multiply(lr_u, mask )
    lr_v = np.multiply(lr_v, mask )
    lr_w = np.multiply(lr_w, mask )

    
    overlap_masks = np.count_nonzero(hr_u_mask-mask)
    if overlap_masks > 0 :
        print("Check compatibility of masks! Difference:", overlap_masks, '/', np.prod(list(hr_u_downsampled.shape)), 'i.e. ', 100*overlap_masks/np.prod(np.array(hr_u_downsampled.shape)) ,"%") 
    
    # check u 
    # just consider values above 1, since noise is added on data
    if np.linalg.norm(lr_u - hr_u_downsampled) >= 1 :
        print("LR u is not compatible with downsampled high res image!! ")
        print('Norm of difference: ', np.linalg.norm(lr_u - hr_u_downsampled))
        print("Count nonzero", np.count_nonzero(lr_u - hr_u_downsampled))
        print('Norm of difference: ', np.linalg.norm(lr_u - hr_u_downsampled))

    #check v
    if np.linalg.norm(lr_v - hr_v_downsampled) >= 1 :
        print("LR v is not compatible with downsampled high res image!! ")
        print('Norm of difference: ', np.linalg.norm(lr_v - hr_v_downsampled))
    
    #check w
    if np.linalg.norm(np.asarray(lr_w) - hr_w_downsampled)>= 1 :
        print("LR w is not compatible with downsampled high res image!! ")
        print('Norm of difference: ', np.linalg.norm(lr_w - hr_w_downsampled))

    
    return lr_u, hr_u_mask, mask



if __name__ == "__main__":
    data_dir = 'Temporal4DFlowNet/data/CARDIAC'
    
    # ---- Patch index files ----
    training_file = '{}/Temporal14MODEL1_2mm_step2_all_axis.csv'.format(data_dir)
   
    # Hyperparameters optimisation variables
    epochs =  1
    batch_size = 10

    patch_size = 14
    res_increase = 2
    

    # Load data file and indexes
    trainset = load_indexes(training_file)
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    z = PatchHandler4D_all_axis(data_dir, patch_size, res_increase, batch_size)
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
            print(' ________________________')
            
    print("\nDone")
    