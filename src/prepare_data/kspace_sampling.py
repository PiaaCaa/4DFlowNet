import numpy as np 
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5functions
import fft_downsampling as fft_fcts


# --------this code is copied from Alexander Fyrdahl:------
def biot_savart_simulation(segments, locations):
  """Reference : Esin Y, Alpaslan F, MRI image enhancement using Biot-Savart law at 3 tesla. Turk J Elec Eng & Comp Sci
  """

  num_coil_segments = segments.shape[0] - 1
  if num_coil_segments < 1:
      raise ValueError('Insufficient coil segments specified')

  if segments.shape[1] == 2:
      segments = np.hstack((segments, np.zeros((num_coil_segments, 1))))

  sensitivity_contribution = np.zeros((locations.shape[0], 3))

  segment_start = segments[0, :]
  for segment_index in range(num_coil_segments):
      segment_end = segment_start
      segment_start = segments[segment_index + 1, :]
      unit_segment_vector = (segment_end - segment_start) / np.linalg.norm(segment_end - segment_start)

      vector_u = -locations + segment_end
      vector_v = locations - segment_start

      cos_alpha = np.dot(vector_u, unit_segment_vector) / np.linalg.norm(vector_u, axis=1)
      cos_beta = np.dot(vector_v, unit_segment_vector) / np.linalg.norm(vector_v, axis=1)
      sin_beta = np.sin(np.arccos(cos_beta))

      sensitivity_magnitudes = (cos_alpha + cos_beta) / (np.linalg.norm(vector_v, axis=1) / sin_beta)

      cross_product_matrix = np.cross(np.identity(3), unit_segment_vector)
      normalized_sensitivity_directions = np.dot(cross_product_matrix, vector_v.T).T / np.linalg.norm(np.dot(cross_product_matrix, vector_v.T).T, axis=1)[:, np.newaxis]

      sensitivity_contribution += normalized_sensitivity_directions * sensitivity_magnitudes[:, np.newaxis]

  return np.linalg.norm(sensitivity_contribution, axis=1)

def define_coils(radius, center, pos, axis, segments=21):
  theta = np.linspace(0, 2 * np.pi, segments)
  if axis == 'x':
      x = np.full_like(theta, center[0] + pos)
      y = center[1] + radius * np.cos(theta)
      z = center[2] + radius * np.sin(theta)
  elif axis == 'y':
      x = center[0] + radius * np.cos(theta)
      y = np.full_like(theta, center[1] + pos)
      z = center[2] + radius * np.sin(theta)
  else:
      x = center[0] + radius * np.cos(theta)
      y = center[1] + radius * np.sin(theta)
      z = np.full_like(theta, center[2] + pos)
  return np.column_stack((x, y, z))

def compute_mri_coil_sensitivity(segments, locations, volume_shape):
  sensitivities = biot_savart_simulation(segments, locations)
  coil_image = np.zeros(volume_shape)
  coil_image[locations[:, 0], locations[:, 1], locations[:, 2]] = sensitivities
  return coil_image

def create_sphere_phantom(volume_shape=(192, 192, 192), radius=48):
  z, y, x = np.indices(volume_shape)
  center = np.array(volume_shape) // 2
  distance_from_center = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
  return distance_from_center <= radius


# --------End copy code from Alexander Fyrdahl:------

def adjust_image_size(image, new_shape):
    """
    Adjust the size of the image to the new shape, assumes 4D image
    """
    old_shape = image.shape
    
    padding = []

    # pad the image
    for i in range(len(new_shape)):
        # diff positive for padding and negative for cropping
        diff = new_shape[i] - old_shape[i]
        
        if diff > 0:
            # pad the image
            pad_before = diff // 2
            pad_after = diff - pad_before
            padding.append((pad_before, pad_after))
        else:
            # no adjustment needed
            padding.append((0, 0))

        #cropping
        if diff < 0:
            t_mid = int(old_shape[i] // 2)
            cropl = int(np.floor(abs(new_shape[i]) / 2))
            cropr = int(np.ceil(abs(new_shape[i]) / 2))
            if i == 0:
                image = image[t_mid - cropl:t_mid + cropr, :, :, :]
            elif i == 1:
                image = image[:, t_mid - cropl:t_mid + cropr, :, :]
            elif i == 2:
                image = image[:, :, t_mid - cropl:t_mid + cropr, :]
            elif i == 3:
                image = image[:, :, :, t_mid - cropl:t_mid + cropr]

    # pad the image
    new_image = np.pad(image, padding, mode='constant', constant_values=0)

    print(f"Adjusted image size from {old_shape} to {new_image.shape}")
    return new_image


def k_space_sampling_timeseries(path_kmask, path_order, path_datamodel, save_as, batchsize = 5000):

    # make batches to process data
    batchsize = 5000
    
    order = sio.loadmat(path_order)
    phs = order['phs'].squeeze()
    phs_max = np.max(phs)
    set_sampling = order['set'].squeeze()
    print( phs.shape)

    # load CFD data 
    with h5py.File(path_datamodel, mode = 'r' ) as p1:
            data = {}
            data['mask'] = np.asarray(p1['mask']).squeeze()
            for vel in ['u', 'v', 'w']:
                data[vel] = np.asarray(p1[vel])
                data[f'venc_{vel}'] = np.asarray(p1[f'{vel}_max']).squeeze()
    data['magnitude'] = np.ones_like(data['u'])

    # get shape of kspacemask
    with h5py.File(path_kmask, mode = 'r' ) as p1:
        _, xk, yk, zk = np.asarray(p1['mask'][1:3, :, :, :], dtype = np.int8).shape
    
    plt.imshow(data['u'][10, :, :, 30])
    plt.title('Original velocity image')
    plt.show()

    # crop data to the same size as kspacemask
    for vel in ['u', 'v', 'w', 'mask', 'magnitude']:
        data[vel] = adjust_image_size(data[vel], (data[vel].shape[0], xk, yk, zk))
    
    data['magnitude'] = np.ones_like(data['u'])
    
    print(data['u'].shape, data['mask'].shape)
    plt.imshow(data['u'][10, :, :, 60])
    plt.title('Cropped velocity image')
    plt.show()

    # interpolate data to maximum number of frames of phds
    t_range_orig = np.linspace(0, 1, data[vel].shape[0])
    t_range_interp = np.linspace(0, 1, phs_max)

    interpolated_data = {}
    for vel in ['u', 'v', 'w']:
        # use sinc interpolation
        interpolated_data[vel] = temporal_sinc_interpolation_ndarray(data[vel], t_range_orig, t_range_interp)
        interpolated_data[f'venc_{vel}'] = temporal_sinc_interpolation(data[f'venc_{vel}'], t_range_orig, t_range_interp)
        
    interpolated_data['mask'] = np.ceil(temporal_sinc_interpolation_ndarray(data['mask'], t_range_orig, t_range_interp))
    interpolated_data['magnitude'] = temporal_sinc_interpolation_ndarray(data['magnitude'], t_range_orig, t_range_interp)

    plt.imshow(data['u'][10, :, :, 60])
    plt.imshow(interpolated_data['u'][10, :, :, 60])
    plt.title('Interpolated velocity image with sinc')
    plt.show()

    # convert data to k-space
    kspace_data = {}
    for vel in ['u', 'v', 'w']:
        kspace_data[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.complex64)
        for t in range(interpolated_data[vel].shape[0]):
            kspace_data[vel][t] = fft_fcts.velocity_img_to_centered_kspace(interpolated_data[vel][t], interpolated_data['magnitude'][t], venc = interpolated_data[f'venc_{vel}'][t])

    #example
    #reconstruct example
    ex_reconstructed, _ = fft_fcts.centered_kspace_to_velocity_img(kspace_data['u'][10], interpolated_data['magnitude'][10], venc = interpolated_data[f'venc_u'][10])
    plt.imshow(ex_reconstructed[ :, :, 60])
    plt.title('Reconstructed velocity image from fft')
    plt.show()

    # now sample as given in phs
    kspace_data_sampled = {}
    for vel in ['u', 'v', 'w']:
        kspace_data_sampled[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.complex64)

    # total size
    total_size = order['phs'].squeeze().shape[0] 

    #--------------------rewrite this-------------------
    # for b in tqdm(range(0, total_size, batchsize)):

    #     indices = np.where(phs[b:b+batchsize] == 1 & set_sampling[b:b+batchsize] == 1)
    #     # load only batches for computational purposes
    #     with h5py.File(path_kmask, mode = 'r' ) as p1:
    #         print(b, b+batchsize)
    #         kspacemask = np.asarray(p1['mask'][b:b+batchsize:4, :, :, :], dtype = np.int8)
        
    #     phs_batch = phs[b:b+batchsize:4]
        
    #     print(kspacemask.shape, phs_batch.shape)


    #     # now sample as given in phs
    #     for vel in ['u', 'v', 'w']:
    #         print(kspace_data_sampled[vel].shape, kspacemask.shape, kspace_data[vel].shape, phs.shape)
    #         for i, ph in enumerate(phs_batch):
    #             # kspace_data_sampled[vel][ph -1, :, :, :] += np.multiply(kspace_data[vel][ph -1, :, :, :], kspacemask[i, :, :, :])

    #             masked_kspace = np.zeros_like(kspace_data[vel][ph -1, :, :, :]) 
    #             masked_kspace[np.where(kspacemask[i, :, :, :] != 0)] = kspace_data[vel][ph -1, :, :, :][np.where(kspacemask[i, :, :, :] != 0)]
    #             kspace_data_sampled[vel][ph -1, :, :, :] += masked_kspace

    #---------------------------------------------------
    for b in tqdm(range(0, total_size, batchsize)):
        # iterate over number of max number of phs

        phs_batch = phs[b:b+batchsize]

        # iterate over total number of phs
        for segm in range(phs_max):
            indices = np.where(np.logical_and(set_sampling[b:b+batchsize] == 1, phs_batch == segm+1))
            
            # reduce phsbatch
            phs_batch_red = phs_batch[indices]

            with h5py.File(path_kmask, mode = 'r' ) as p1:
                kspacemask = np.asarray(p1['mask'][b:b+batchsize, :, :, :], dtype = np.int8)
                kspacemask = kspacemask[indices[0], :, :, :] # reduce

            print("n segm", segm, kspacemask.shape, phs_batch_red.shape)
            k_space_redsum = np.sum(kspacemask, axis = 0)

            # now sample as given in phs
            for vel in ['u', 'v', 'w']:
                kspace_data_sampled[vel][segm] += np.multiply(kspace_data[vel][segm], k_space_redsum)

    
    # convert back to velocity image
    velocity_data_sampled = {}
    for vel in ['u', 'v', 'w']:
        velocity_data_sampled[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.float32)
        for t in range(phs_max):
            velocity_data_sampled[vel][t], _ = fft_fcts.centered_kspace_to_velocity_img(kspace_data_sampled[vel][t], interpolated_data['magnitude'][t], venc = interpolated_data[f'venc_{vel}'][t])


    plt.imshow(velocity_data_sampled['u'][10, :, :, 60])
    plt.show()

    # save to h5

    for vel in ['u', 'v', 'w']:
        h5functions.save_to_h5(save_as, vel, velocity_data_sampled[vel], expand_dims=False)
        h5functions.save_to_h5(save_as, f'venc_{vel}', interpolated_data[f'venc_{vel}'], expand_dims=False)


def k_space_sampling_static(path_kmask, path_order, path_datamodel, save_as):
    
    order = sio.loadmat(path_order)
    phs = order['phs'].squeeze()
    phs_max = np.max(phs)
    set_sampling = order['set'].squeeze()
    print( phs.shape)

    # load CFD data 
    with h5py.File(path_datamodel, mode = 'r' ) as p1:
            data = {}
            data['mask'] = np.asarray(p1['mask']).squeeze()
            for vel in ['u', 'v', 'w']:
                data[vel] = np.asarray(p1[vel])
                data[f'venc_{vel}'] = np.asarray(p1[f'{vel}_max']).squeeze()
    data['magnitude'] = np.ones_like(data['u'])

    with h5py.File(path_kmask, mode = 'r' ) as p1:
            _, xk, yk, zk = np.asarray(p1['mask'][1:3, :, :, :], dtype = np.int8).shape
    
    plt.imshow(data['u'][10, :, :, 30])
    plt.title('Original velocity image')
    # plt.show()

    # crop data to the same size as kspacemask
    for vel in ['u', 'v', 'w', 'mask', 'magnitude']:
        data[vel] = adjust_image_size(data[vel], (data[vel].shape[0], xk, yk, zk))

    midx = int(data['u'].shape[1]//2)
    #TODO delete this when croping works!!
    for vel in ['u', 'v', 'w', 'mask']:
        data[vel] = data[vel][:, midx-int(np.floor(xk/2)):midx+int(np.ceil(xk/2)), :, :]
        # print(data['u'].shape, data['mask'].shape)
    
    data['magnitude'] = np.ones_like(data['u'])
        # data[vel] = np.pad(data[vel], ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    print(data['u'].shape, data['mask'].shape)
    plt.imshow(data['u'][10, :, :, 60])
    plt.title('Cropped velocity image')
    # plt.show()

    # interpolate data to maximum number of frames of phds
    t_range_orig = np.linspace(0, 1, data[vel].shape[0])
    t_range_interp = np.linspace(0, 1, phs_max)

    interpolated_data = {}
    for vel in ['u', 'v', 'w']:
        # use sinc interpolation
        interpolated_data[vel] = temporal_sinc_interpolation_ndarray(data[vel], t_range_orig, t_range_interp)
        interpolated_data[f'venc_{vel}'] = temporal_sinc_interpolation(data[f'venc_{vel}'], t_range_orig, t_range_interp)
        
    interpolated_data['mask'] = np.ceil(temporal_sinc_interpolation_ndarray(data['mask'], t_range_orig, t_range_interp))
    interpolated_data['magnitude'] = temporal_sinc_interpolation_ndarray(data['magnitude'], t_range_orig, t_range_interp)

    plt.imshow(data['u'][10, :, :, 60])

    plt.imshow(interpolated_data['u'][10, :, :, 60])
    plt.title('Interpolated velocity image with sinc')
    # plt.show()

    # convert data to k-space
    # kspace_data = {}
    # for vel in ['u', 'v', 'w']:
    #     kspace_data[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.complex64)
    #     for t in range(interpolated_data[vel].shape[0]):
    #         kspace_data[vel][t] = fft_fcts.velocity_img_to_kspace(interpolated_data[vel][t], interpolated_data['magnitude'][t], venc = interpolated_data[f'venc_{vel}'][t])

    #-----------------now use only one k-space set----------------------------
    print('shapes: ', interpolated_data['u'].shape, interpolated_data['magnitude'].shape, interpolated_data[f'venc_u'].shape)
    kspace_data = {}
    kspace_data['u'] = fft_fcts.velocity_img_to_centered_kspace(interpolated_data['u'][0, :, :, :], interpolated_data['magnitude'][0, :, :, :], venc = interpolated_data[f'venc_u'][0])
    kspace_data['v'] = fft_fcts.velocity_img_to_centered_kspace(interpolated_data['v'][0, :, :, :], interpolated_data['magnitude'][0, :, :, :], venc = interpolated_data[f'venc_v'][0])
    kspace_data['w'] = fft_fcts.velocity_img_to_centered_kspace(interpolated_data['w'][0, :, :, :], interpolated_data['magnitude'][0, :, :, :], venc = interpolated_data[f'venc_w'][0])

    interpolated_data['u'] = interpolated_data['u'][0, :, :, :]
    interpolated_data['v'] = interpolated_data['v'][0, :, :, :]
    interpolated_data['w'] = interpolated_data['w'][0, :, :, :]
    interpolated_data['venc_u'] = interpolated_data['venc_u'][0]
    interpolated_data['venc_v'] = interpolated_data['venc_v'][0]
    interpolated_data['venc_w'] = interpolated_data['venc_w'][0]
    interpolated_data['magnitude'] = np.ones_like(interpolated_data['u'])


    #example
    #reconstruct example
    ex_reconstructed, _ = fft_fcts.centered_kspace_to_velocity_img(kspace_data['u'], interpolated_data['magnitude'], venc = interpolated_data[f'venc_u'])
    plt.imshow(ex_reconstructed[ :, :, 60])
    plt.title('Reconstructed velocity image from fft')
    plt.show()

    # now sample as given in phs
    kspace_data_sampled = {}
    for vel in ['u', 'v', 'w']:
        kspace_data_sampled[vel] = np.zeros((xk, yk, zk), dtype = np.complex64)

    # total size
    total_size = order['phs'].squeeze().shape[0] 

    # find all corresponding indices
    indices = np.where(np.logical_and(set_sampling == 1, phs == 1))

    with h5py.File(path_kmask, mode = 'r' ) as p1:
        k_space_mask_static = np.asarray(p1['mask'][indices[0]], dtype = np.int8)

    print(k_space_mask_static.shape)

    plt.imshow(np.abs(np.sum((k_space_mask_static[:, :, :, 60]), axis = 0)))
    plt.show()

    k_space_redsum = np.sum(k_space_mask_static, axis = 0)
    print(np.unique(k_space_redsum))

    for vel in ['u', 'v', 'w']:
        kspace_data_sampled[vel] = np.zeros((xk, yk, zk), dtype = np.complex64)
        kspace_data_sampled[vel] = np.multiply(kspace_data[vel], k_space_redsum)

    plt_x, plt_y, plt_z = np.where(k_space_redsum != 0)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(plt_x, plt_y, plt_z, c = 'r', marker = 'o')
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(kspace_data['u'][:, :, 60]))
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(kspace_data_sampled['u'][:, :, 60]))
    plt.show()


    # convert back to velocity image
    velocity_data_sampled = {}
    for vel in ['u', 'v', 'w']:
        velocity_data_sampled[vel], _ = fft_fcts.centered_kspace_to_velocity_img(kspace_data_sampled[vel], interpolated_data['magnitude'], venc = interpolated_data[f'venc_{vel}'])

    plt.imshow(velocity_data_sampled['u'][:, :, 60])
    plt.show()

    # save to h5
    save_as = 'results/interpolation/M1_2mm_step1_static_dynamic_kspace_sampled5_static3.h5'
    for vel in ['u', 'v', 'w']:
        h5functions.save_to_h5(save_as, vel, velocity_data_sampled[vel], expand_dims=False)
        # h5functions.save_to_h5(save_as, f'venc_{vel}', interpolated_data[f'venc_{vel}'], expand_dims=False)

    exit()
    # for b in tqdm(range(0, total_size, batchsize)):

    #     # load only batches for computational purposes
    #     with h5py.File(path_kmask, mode = 'r' ) as p1:
    #         print(b, b+batchsize)
    #         kspacemask = np.asarray(p1['mask'][b:b+batchsize:4, :, :, :], dtype = np.int8)
        
    #     phs_batch = phs[b:b+batchsize:4]
        
    #     print(kspacemask.shape, phs_batch.shape)

    #     tk, xk, yk, zk = kspacemask.shape

    #     # now sample as given in phs
    #     for vel in ['u', 'v', 'w']:
    #         print(kspace_data_sampled[vel].shape, kspacemask.shape, kspace_data[vel].shape, phs.shape)
    #         for i, ph in enumerate(phs_batch):
    #             # kspace_data_sampled[vel][ph -1, :, :, :] += np.multiply(kspace_data[vel][ph -1, :, :, :], kspacemask[i, :, :, :])

    #             masked_kspace = np.zeros_like(kspace_data[vel][ph -1, :, :, :]) 
    #             masked_kspace[np.where(kspacemask[i, :, :, :] != 0)] = kspace_data[vel][ph -1, :, :, :][np.where(kspacemask[i, :, :, :] != 0)]
    #             kspace_data_sampled[vel][ph -1, :, :, :] += masked_kspace

        
    # # convert back to velocity image
    # velocity_data_sampled = {}
    # for vel in ['u', 'v', 'w']:
    #     velocity_data_sampled[vel] = np.zeros((phs_max, xk, yk, zk), dtype = np.float32)
    #     for t in range(phs_max):
    #         velocity_data_sampled[vel][t], _ = fft_fcts.kspace_to_velocity_img(kspace_data_sampled[vel][t], interpolated_data['magnitude'][t], venc = interpolated_data[f'venc_{vel}'][t])


    plt.imshow(velocity_data_sampled['u'][10, :, :, 60])
    plt.show()

    # save to h5
    save_as = 'results/interpolation/M1_2mm_step1_static_dynamic_kspace_sampled5.h5'
    for vel in ['u', 'v', 'w']:
        h5functions.save_to_h5(save_as, vel, velocity_data_sampled[vel], expand_dims=False)
        h5functions.save_to_h5(save_as, f'venc_{vel}', interpolated_data[f'venc_{vel}'], expand_dims=False)


if __name__ == '__main__':
    # Define datasets
    path_kmask = 'data/kspacemask.h5'
    path_order = 'data/order.mat'
    path_datamodel = 'data/CARDIAC/M2_2mm_step1_static_dynamic.h5'
    save_as = 'results/interpolation/M2_2mm_step1_static_kspace_sampled_coilsens2.h5'

    # 1. Use coil sensitivity matrix on CFD data
    with h5py.File(path_datamodel, mode = 'r' ) as p1: 
        spatial_res = p1['u'].shape[1:]
        static_mask_vel = np.array(p1['mask']).squeeze()[0]
    
    phantom_center = [x//2 for x in spatial_res]
    coil_radius = 30  # Radius of the coil circle
    coil_offset = 75  # Distance from the center of the phantom

    # Define the coil locations
    coil_axis = ['x' , 'y', 'x', 'y', 'z', 'z']
    coil_location = [coil_offset ,  coil_offset , -coil_offset, -coil_offset, coil_offset, -coil_offset]
    coils = [define_coils(coil_radius, phantom_center, pos, ax) for pos, ax in zip(coil_location, coil_axis)]

    # Visualize the coils
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.title.set_text('Coil locations in 3D space')

    for idx, coil in enumerate(coils):
        ax.plot(coil[:, 0], coil[:, 1], coil[:, 2], label=f'Coil {idx+1}')
    
    mask_x, mask_y, mask_z = np.where(static_mask_vel)
    ax.scatter(mask_x, mask_y, mask_z, c = 'gray', marker = 'o', label= 'Phantom',alpha=0.2 )

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(0, spatial_res[0]+20)
    ax.set_ylim(0, spatial_res[1]+20)
    ax.set_zlim(0, spatial_res[2]+20)
    plt.legend()
    plt.show()

    # Generate noisy coil images
    noise_level = 2.5
    fig, ax = plt.subplots(1, len(coil_axis))
    coil_images = np.zeros((spatial_res[0], spatial_res[1], spatial_res[2], len(coil_axis)), dtype=np.complex128)
    for idx, coil in enumerate(coils):
        sens = compute_mri_coil_sensitivity(coil, np.argwhere(static_mask_vel), spatial_res)
        sens = sens.reshape(spatial_res)
        coil_image = static_mask_vel * sens
        # coil_image_ft = np.fft.fftn(coil_image)
        # real_noise = np.random.normal(0, noise_level, coil_image.shape)
        # imag_noise = np.random.normal(0, noise_level, coil_image.shape)
        # coil_image_ft_noisy = coil_image_ft + (real_noise + 1j * imag_noise)
        # coil_image = np.fft.ifftn(coil_image_ft_noisy)
        coil_images[:,:,:,idx] = coil_image
        ax[idx].imshow(abs(coil_image[:, :, 40].squeeze()))
        ax[idx].title.set_text(f'Coil {idx+1}')
    plt.show()

    sum_coil_images = np.sum(coil_images, axis = -1)
    
    data = {}
    with h5py.File(path_datamodel, mode = 'r' ) as p1:
        for vel in ['u', 'v', 'w']:
            data[vel] = np.asarray(p1[vel])[0]
            data[f'venc_{vel}'] = np.asarray(p1[f'{vel}_max']).squeeze()[0]
            data[f'{vel}_complex'] = np.multiply(np.ones_like(data[vel]), np.exp(1j*(data[vel]/data[f'venc_{vel}']*np.pi)))

    data_sensitive = {}
    for vel in ['u', 'v', 'w']:
        data_sensitive[f'{vel}_complex'] = np.multiply(data[f'{vel}_complex'], sum_coil_images)
        #TODO normalize afterwards? 

        data_sensitive[vel] = np.angle(data_sensitive[f'{vel}_complex'])/ np.pi * data[f'venc_{vel}']

    #new_phase / math.pi * venc

    plt.imshow(data_sensitive['u'][ :, :, 30])
    plt.show()

    # h5functions.save_to_h5(save_as, 'u_csens', data_sensitive['u'], expand_dims=False)
    # h5functions.save_to_h5(save_as, 'v_csens', data_sensitive['v'], expand_dims=False)
    # h5functions.save_to_h5(save_as, 'w_csens', data_sensitive['w'], expand_dims=False)
    h5functions.save_to_h5(save_as, 'sum_coil_images', sum_coil_images, expand_dims=False)
    # h5functions.save_to_h5(save_as, 'u', data['u'], expand_dims=False)

    # 2. Use k-space mask on CFD data
    # k_space_sampling_timeseries(path_kmask, path_order, path_datamodel, save_as, batchsize = 5000)
    # k_space_sampling_static(path_kmask, path_order, path_datamodel, save_as)

    # 3. Reconstruct undersampled k-space with compressed sensing (CS)


    # 4. Compare to original data