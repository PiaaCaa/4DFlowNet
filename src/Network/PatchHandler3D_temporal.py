import tensorflow as tf
import numpy as np
import h5py
import os
import time

class PatchHandler4D():
    # constructor
    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6):
        self.patch_size = patch_size
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames = ['venc_u','venc_v','venc_w']
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'

    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        ds = ds.prefetch(self.batch_size)
        
        return ds
    
    def load_data_using_patch_index(self, indexes):
        return tf.py_function(func=self.load_patches_from_index_file, 
            # U-LR, HR, MAG, V-LR, HR, MAG, W-LR, HR, MAG, venc, MASK
            inp=[indexes], 
                Tout=[tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32])

    def load_patches_from_index_file(self, indexes):
        # Do typecasting, we need to make sure everything has the correct data type
        # Solution for tf2: https://stackoverflow.com/questions/56122670/how-to-get-string-value-out-of-tf-tensor-which-dtype-is-string
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))
        
        #read out attributes from csvline
        idx = int(indexes[2])
        x_start, y_start, z_start = int(indexes[3]), int(indexes[4]), int(indexes[5])
        is_rotate = int(indexes[6])
        rotation_plane = int(indexes[7])
        rotation_degree_idx = int(indexes[8])

        patch_size = self.patch_size
        hr_patch_size = self.patch_size * self.res_increase
        
        # ============ get the patch ============ 
        #TODO  format is (t, h, w, d) such that [x_start, idx, y_start, z_start]
        # NO INCREASE IN Y AND Z
        patch_t_index  = np.index_exp[x_start:x_start+patch_size,idx, y_start:y_start+patch_size, z_start:z_start+patch_size]
        hr_t_patch_index = np.index_exp[x_start*self.res_increase :x_start*self.res_increase +hr_patch_size,idx, y_start:y_start+patch_size, z_start:z_start+patch_size]
        mask_t_index = np.index_exp[x_start*self.res_increase :x_start*self.res_increase +hr_patch_size,idx  ,y_start:y_start+patch_size, z_start:z_start+patch_size ]
        # mask_index = np.index_exp[0, x_start*self.res_increase :x_start*self.res_increase +hr_patch_size ,y_start*self.res_increase :y_start*self.res_increase +hr_patch_size , z_start*self.res_increase :z_start*self.res_increase +hr_patch_size ]
        u_patch, u_hr_patch, mag_u_patch, v_patch, v_hr_patch, mag_v_patch, w_patch, w_hr_patch, mag_w_patch, venc, mask_patch = self.load_vectorfield(hd5path, lr_hd5path, idx, mask_t_index, patch_t_index, hr_t_patch_index)
        
   
        # Expand dims (for InputLayer)
        return u_patch[...,tf.newaxis], v_patch[...,tf.newaxis], w_patch[...,tf.newaxis], \
                    mag_u_patch[...,tf.newaxis], mag_v_patch[...,tf.newaxis], mag_w_patch[...,tf.newaxis], \
                    u_hr_patch[...,tf.newaxis], v_hr_patch[...,tf.newaxis], w_hr_patch[...,tf.newaxis], \
                    venc, mask_patch
                    
    
    def create_temporal_mask(self, mask, n_frames):
        '''
        from static mask create temporal mask of shape (n_frames, h, w, d)
        '''
        assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
        return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)

    def load_vectorfield(self, hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components
            Also returns the global venc and HiRes mask
        '''
        hires_images = []
        lowres_images = []
        mag_images = []
        vencs = []
        global_venc = 0

        # Load the U, V, W component of HR, LR, and MAG
        with h5py.File(hd5path, 'r') as hl:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                w_hr = hl.get(self.hr_colnames[i])[hr_patch_index]
                # add them to the list
                hires_images.append(w_hr)

            # We only have 1 mask for all the objects in 1 file
            try:
                mask = hl.get(self.mask_colname)[mask_index] # Mask value [0 .. 1]
            except:
                #TODO this probably has to be chnaged if structure changes
                mask_temp = self.create_temporal_mask(np.asarray(hl.get(self.mask_colname)).squeeze(),  hl.get(self.hr_colnames[i]).shape[0])
                mask = mask_temp[mask_index] 
            mask = (mask >= self.mask_threshold) * 1.
            print("mask shape:", hl.get(self.mask_colname).shape)
            
        with h5py.File(lr_hd5path, 'r') as hl:
            for i in range(len(self.lr_colnames)):
                w = hl.get(self.lr_colnames[i])[patch_index]
                mag_w = hl.get(self.mag_colnames[i])[patch_index]
                #TODO change venc
                w_venc = hl.get(self.venc_colnames[i])[:]

                # add them to the list
                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)
        
        global_venc = np.max(vencs)

        # Convert to numpy array
        hires_images = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)
        
        # Normalize the values 
        hires_images = self._normalize(hires_images, global_venc) # Velocity normalized to -1 .. 1
        lowres_images = self._normalize(lowres_images, global_venc)
        mag_images = mag_images / 4095. # Magnitude 0 .. 1

        # U-LR, HR, MAG, V-LR, HR, MAG, w-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'),\
            global_venc.astype('float32'), mask.astype('float32')
    
    def _normalize(self, u, venc):
        return u / venc


class PatchHandler4D_all_axis():
    # constructor
    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6):
        self.patch_size = patch_size
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames =  ['u_max', 'v_max', 'w_max']# ['venc_u','venc_v','venc_w'] #
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'

    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        ds = ds.prefetch(self.batch_size)
        
        return ds
    
    def load_data_using_patch_index(self, indexes):
        return tf.py_function(func=self.load_patches_from_index_file, 
            # U-LR, HR, MAG, V-LR, HR, MAG, W-LR, HR, MAG, venc, MASK
            inp=[indexes], 
                Tout=[tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32])

    def load_patches_from_index_file(self, indexes):
        # t = time.time()
        # Do typecasting, we need to make sure everything has the correct data type
        # Solution for tf2: https://stackoverflow.com/questions/56122670/how-to-get-string-value-out-of-tf-tensor-which-dtype-is-string
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))
        
        #read out attributes from csvline
        axis= int(indexes[2])
        idx = int(indexes[3])
        start_t, start_1, start_2 = int(indexes[4]), int(indexes[5]), int(indexes[6])
        step_t = int(indexes[7])
        reverse = int(indexes[8]) # 1 for no reverse, -1 for reverse order, only reverse te first spatial component
       
        patch_size = self.patch_size
        
        # if step is 1, the loaded LR data is already downsampled
        if step_t == 1:
            start_t_lr = start_t
            hr_patch_size = int(patch_size*self.res_increase)
            lr_patch_size = patch_size
            start_t_hr = int(start_t*self.res_increase)
        else:
            start_t_lr = start_t
            hr_patch_size = self.patch_size * step_t    
            lr_patch_size = hr_patch_size
            start_t_hr = start_t


        # ============ get the patch ============ 
        if axis == 0 :
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t,   idx, start_1:start_1+patch_size, start_2:start_2+patch_size]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          idx, start_1:start_1+patch_size, start_2:start_2+patch_size]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          idx, start_1:start_1+patch_size, start_2:start_2+patch_size]
        elif axis == 1:
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t, start_1:start_1+patch_size,idx, start_2:start_2+patch_size]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,        start_1:start_1+patch_size,idx, start_2:start_2+patch_size]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,        start_1:start_1+patch_size,idx, start_2:start_2+patch_size]
        elif axis == 2:
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t, start_1:start_1+patch_size, start_2:start_2+patch_size, idx]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,        start_1:start_1+patch_size, start_2:start_2+patch_size, idx]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,        start_1:start_1+patch_size, start_2:start_2+patch_size, idx]

        
        u_patch, u_hr_patch, mag_u_patch, v_patch, v_hr_patch, mag_v_patch, w_patch, w_hr_patch, mag_w_patch, venc, mask_patch = self.load_vectorfield(hd5path, lr_hd5path, idx, mask_t_index, patch_t_index, hr_t_patch_index, reverse)
        
        # Expand dims (for InputLayer)
        return u_patch[...,tf.newaxis], v_patch[...,tf.newaxis], w_patch[...,tf.newaxis], \
                    mag_u_patch[...,tf.newaxis], mag_v_patch[...,tf.newaxis], mag_w_patch[...,tf.newaxis], \
                    u_hr_patch[...,tf.newaxis], v_hr_patch[...,tf.newaxis], w_hr_patch[...,tf.newaxis], \
                    venc, mask_patch
                    

    def create_temporal_mask(self, mask, n_frames):
        '''
        from static mask create temporal mask of shape (n_frames, h, w, d)
        '''
        assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
        return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)

    def load_vectorfield(self, hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index, reverse):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components
            Also returns the global venc and HiRes mask
        '''
        
        hires_images = []
        lowres_images = []
        mag_images = []
        vencs = []
        global_venc = 0

        # Load the U, V, W component of HR, LR, and MAG
        with h5py.File(hd5path, 'r') as hl:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                w_hr = hl.get(self.hr_colnames[i])[hr_patch_index]
                # add them to the list
                hires_images.append(w_hr)

            # We only have 1 mask for all the objects in 1 file
            try:
                mask = hl.get(self.mask_colname)[mask_index] # Mask value [0 .. 1]
            except:
                print('Temporal static mask created')
                mask_temp = self.create_temporal_mask(np.asarray(hl.get(self.mask_colname)).squeeze(),  hl.get(self.hr_colnames[i]).shape[0])
                mask = mask_temp[mask_index] 
            mask = (mask >= self.mask_threshold) * 1.
            # print("mask shape:", hl.get(self.mask_colname).shape)
        
        with h5py.File(lr_hd5path, 'r') as hl:
            
            for i in range(len(self.lr_colnames)):
                w = hl.get(self.lr_colnames[i])[patch_index]
                mag_w = hl.get(self.mag_colnames[i])[patch_index]
                w_venc = np.array(hl.get(self.venc_colnames[i])).squeeze()
                            
                # add them to the list
                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)
        
        global_venc = np.max(vencs)
        
        # Convert to numpy array
        hires_images = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)

        if reverse: #reverse images shape now (3, t, x, y) (or other combinations of x, y, z)
            hires_images = hires_images[:, :, ::-1, :] 
            lowres_images = lowres_images[:, :, ::-1, :] 
            mag_images = mag_images[:, :, ::-1, :] 
            mask = mask[:, ::-1,:] # mask shape (t, x, y)
            
        
        # Normalize the values 
        hires_images = self._normalize(hires_images, global_venc) # Velocity normalized to -1 .. 1
        lowres_images = self._normalize(lowres_images, global_venc)
        mag_images = mag_images / 4095. # Magnitude 0 .. 1
        
        # U-LR, HR, MAG, V-LR, HR, MAG, w-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'),\
            global_venc.astype('float32'), mask.astype('float32')
    
    def _normalize(self, u, venc):
        return u / venc


class PatchHandler4D_extended_data_augmentation():
    # constructor
    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6):
        self.patch_size = patch_size # this will be overwritten in augmenttaion file
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames =  ['u_max', 'v_max', 'w_max']# ['venc_u','venc_v','venc_w'] #
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'
        self.colname2number = {'u':0, 'v':1, 'w':2}

    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        ds = ds.prefetch(self.batch_size)
        
        return ds
    
    def load_data_using_patch_index(self, indexes):
        return tf.py_function(func=self.load_patches_from_index_file, 
            # U-LR, HR, MAG, V-LR, HR, MAG, W-LR, HR, MAG, venc, MASK
            inp=[indexes], 
                Tout=[tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32])

    def load_patches_from_index_file(self, indexes):
        # t = time.time()
        # Do typecasting, we need to make sure everything has the correct data type
        # Solution for tf2: https://stackoverflow.com/questions/56122670/how-to-get-string-value-out-of-tf-tensor-which-dtype-is-string
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))
        
        # TODO atapt this for new attributes
        #read out attributes from csvline
        axis= int(indexes[2])
        idx = int(indexes[3])
        start_t, start_1, start_2 = int(indexes[4]), int(indexes[5]), int(indexes[6])
        step_t = int(indexes[7])
        s_patchsize = int(indexes[8])
        t_patchsize = int(indexes[9])
        flip_1 = int(indexes[10])
        flip_2 = int(indexes[11])
        rot_angle = int(indexes[12])
        sign_u = int(indexes[13])
        sign_v = int(indexes[14])
        sign_w = int(indexes[15])
        swap_u = bytes.decode(indexes[16].numpy())#str(indexes[16])
        swap_v = bytes.decode(indexes[17].numpy())#str(indexes[17])
        swap_w = bytes.decode(indexes[18].numpy())#str(indexes[18])
        coverage = float(indexes[19])

        # print(f'CSV FILE:, idx: {idx}, start_t: {start_t}, start_1: {start_1}, start_2: {start_2}, step_t: {step_t}, temporal_patch_size: {t_patchsize}, flip_1: {flip_1}, flip_2: {flip_2}, rot_angle: {rot_angle}, sign_u: {sign_u}, sign_v: {sign_v}, sign_w: {sign_w}, swap_u: {swap_u}, swap_v: {swap_v}, swap_w: {swap_w}, coverage: {coverage}')
        
        # if step is 1, the loaded LR data is already downsampled
        if step_t == 1:
            # start_t_lr = int(start_t//self.res_increase)
            # hr_patch_size = int(t_patchsize*self.res_increase)
            # lr_patch_size = t_patchsize
            # start_t_hr = start_t
            # updated loading for temporal patches
            start_t_lr = start_t
            hr_patch_size = int(t_patchsize*self.res_increase)
            lr_patch_size = t_patchsize
            start_t_hr = int(start_t*self.res_increase)
        else:
            start_t_lr = start_t
            hr_patch_size = t_patchsize * step_t    #self.res_increase
            lr_patch_size = hr_patch_size
            start_t_hr = start_t

        # print('LR from:', start_t_lr , 'to', start_t_lr+lr_patch_size)
        # print('HR from: ', start_t_hr, start_t + hr_patch_size)

        # ============ get the patch ============ 
        if axis == 0 :
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t,   idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
        elif axis == 1:
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t, start_1:start_1+s_patchsize,idx, start_2:start_2+s_patchsize]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize,idx, start_2:start_2+s_patchsize]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize,idx, start_2:start_2+s_patchsize]
        elif axis == 2:
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]

        
        u_patch, u_hr_patch, mag_u_patch, v_patch, v_hr_patch, mag_v_patch, w_patch, w_hr_patch, mag_w_patch, venc, mask_patch = self.load_vectorfield(hd5path, lr_hd5path, mask_t_index, patch_t_index, hr_t_patch_index, 
                                                                                                                                                       flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w)
        
        # Expand dims (for InputLayer)
        return u_patch[...,tf.newaxis], v_patch[...,tf.newaxis], w_patch[...,tf.newaxis], \
                    mag_u_patch[...,tf.newaxis], mag_v_patch[...,tf.newaxis], mag_w_patch[...,tf.newaxis], \
                    u_hr_patch[...,tf.newaxis], v_hr_patch[...,tf.newaxis], w_hr_patch[...,tf.newaxis], \
                    venc, mask_patch
                    

    def create_temporal_mask(self, mask, n_frames):
        '''
        from static mask create temporal mask of shape (n_frames, h, w, d)
        '''
        assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
        return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)

    def load_vectorfield(self, hd5path, lr_hd5path,  mask_index, patch_index, hr_patch_index, 
                         flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components
            Also returns the global venc and HiRes mask
        '''
        
        hires_images = []
        lowres_images = []
        mag_images = []
        vencs = []
        global_venc = 0

        if os.path.exists(hd5path) == False:
            print(f'File {hd5path} does not exist')
        if os.path.exists(lr_hd5path) == False:
            print(f'File {lr_hd5path} does not exist')

        # Load the U, V, W component of HR, LR, and MAG
        with h5py.File(hd5path, 'r') as hl:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                w_hr = hl.get(self.hr_colnames[i])[hr_patch_index]
                # add them to the list
                hires_images.append(w_hr)

            # We only have 1 mask for all the objects in 1 file
            try:
                mask = hl.get(self.mask_colname)[mask_index] # Mask value [0 .. 1]
            except:
                print('Temporal static mask created')
                mask_temp = self.create_temporal_mask(np.asarray(hl.get(self.mask_colname)).squeeze(),  hl.get(self.hr_colnames[i]).shape[0])
                mask = mask_temp[mask_index] 
            mask = (mask >= self.mask_threshold) * 1.
            # print("mask shape:", hl.get(self.mask_colname).shape)
        
        with h5py.File(lr_hd5path, 'r') as hl:
            for i in range(len(self.lr_colnames)):
                w = hl.get(self.lr_colnames[i])[patch_index]
                mag_w = hl.get(self.mag_colnames[i])[patch_index]
                w_venc = np.array(hl.get(self.venc_colnames[i])).squeeze()
                            
                # add them to the list
                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)
        
        global_venc = np.max(vencs)
        
        # Convert to numpy array
        hires_images = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)

        if flip_1 ==1 : #reverse images shape now (3, t, x, y) (or other combinations of x, y, z)
            hires_images = hires_images[:, :, ::-1, :] 
            lowres_images = lowres_images[:, :, ::-1, :] 
            mag_images = mag_images[:, :, ::-1, :] 
            mask = mask[:, ::-1,:] # mask shape (t, x, y)
        if flip_2 ==1 :
            hires_images = hires_images[:, :, :, ::-1] 
            lowres_images = lowres_images[:, :, :, ::-1] 
            mag_images = mag_images[:, :, :, ::-1] 
            mask = mask[:, :, ::-1]
        if rot_angle !=0:
            if rot_angle == 90:
                k=1
            elif rot_angle == 180:
                k=2
            elif rot_angle == 270:
                k=3
            hires_images = np.rot90(hires_images, k=k, axes=(2, 3))
            lowres_images = np.rot90(lowres_images, k=k, axes=(2, 3))
            mag_images = np.rot90(mag_images, k=k, axes=(2, 3))
            mask = np.rot90(mask, k=k, axes=(1, 2))
        if sign_u == -1:
            hires_images[0] = -hires_images[0]
            lowres_images[0] = -lowres_images[0]
        if sign_v == -1:
            hires_images[1] = -hires_images[1]
            lowres_images[1] = -lowres_images[1]
        if sign_w == -1:
            hires_images[2] = -hires_images[2]
            lowres_images[2] = -lowres_images[2]
        
        if swap_u != 'u' and swap_v != 'v' :
            # Store the original values in a temporary variable
            temp_images_hr = [
                hires_images[self.colname2number[swap_u]],
                hires_images[self.colname2number[swap_v]],
                hires_images[self.colname2number[swap_w]]
            ]
            temp_images_lr = [
                lowres_images[self.colname2number[swap_u]],
                lowres_images[self.colname2number[swap_v]],
                lowres_images[self.colname2number[swap_w]]
            ]
            # Assign the temporary values back to hires_images
            hires_images[0], hires_images[1], hires_images[2] = temp_images_hr[0], temp_images_hr[1], temp_images_hr[2]
            lowres_images[0], lowres_images[1], lowres_images[2] = temp_images_lr[0], temp_images_lr[1], temp_images_lr[2]

            temp_images_hr = None
            temp_images_lr = None

        
        # Normalize the values 
        hires_images = self._normalize(hires_images, global_venc) # Velocity normalized to -1 .. 1
        lowres_images = self._normalize(lowres_images, global_venc)
        mag_images = mag_images / 4095. # Magnitude 0 .. 1
        
        # U-LR, HR, MAG, V-LR, HR, MAG, w-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'),\
            global_venc.astype('float32'), mask.astype('float32')
    
    def _normalize(self, u, venc):
        return u / venc


