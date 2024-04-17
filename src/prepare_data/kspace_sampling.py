import numpy as np 
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
# from prepare_data import h5functions
import h5functions
import fft_downsampling as fft_fcts
import cfl 
import os
import sys
import argparse

# --------this code is copied from Alexander Fyrdahl:------
def biot_savart_simulation(segments, locations):
  """Reference : Esin Y, Alpaslan F, MRI image enhancement using Biot-Savart law at 3 tesla. Turk J Elec Eng & Comp Sci
  """

  eps = 1e-10  
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
      unit_segment_vector = (segment_end - segment_start) / (np.linalg.norm(segment_end - segment_start))

      vector_u = -locations + segment_end
      vector_v = locations - segment_start

      cos_alpha = np.dot(vector_u, unit_segment_vector) / (np.linalg.norm(vector_u, axis=1)+eps)
      cos_beta = np.dot(vector_v, unit_segment_vector) / (np.linalg.norm(vector_v, axis=1)+eps)
      sin_beta = np.sin(np.arccos(cos_beta))

      sensitivity_magnitudes = (cos_alpha + cos_beta) / ((np.linalg.norm(vector_v, axis=1) / sin_beta) +eps)

      cross_product_matrix = np.cross(np.identity(3), unit_segment_vector)
      normalized_sensitivity_directions = np.dot(cross_product_matrix, vector_v.T).T / (np.linalg.norm(np.dot(cross_product_matrix, vector_v.T).T, axis=1)[:, np.newaxis]+eps)

      sensitivity_contribution += normalized_sensitivity_directions * sensitivity_magnitudes[:, np.newaxis]

  return np.linalg.norm(sensitivity_contribution, axis=1)



def define_coils(radius, center, pos, axis, segments=21):
    """
    Define the coordinates of coils in a cylindrical arrangement.

    Parameters:
    radius (float): The radius of the cylindrical arrangement.
    center (tuple): The center coordinates of the cylindrical arrangement (x, y, z).
    pos (float): The position of the coils along the specified axis.
    axis (str): The axis along which the coils are positioned ('x', 'y', or 'z').
    segments (int, optional): The number of segments in the cylindrical arrangement. Default is 21.

    Returns:
    numpy.ndarray: An array of shape (segments, 3) containing the coordinates of the coils.
    """

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
  print('Coil sensitivity max:', np.max(coil_image), np.min(coil_image))
  return coil_image

# --------End copy code from Alexander Fyrdahl------

def fibonacci_sphere(samples=1000, r= 1):
    """
    Create a fibonacci sphere with a given number of samples and radius.
    """

    points = []
    tangent = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = (1 - (i / float(samples - 1)) * 2) # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)*r  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y*r, z))
        tangent.append((-r*np.cos(theta)*np.sin(phi), 0, r*np.cos(theta)*np.cos(phi)))
    
    return points, tangent

def define_circle_on_sphere(point, tangent, n, radius):
    """
    Define a circle on a sphere with a given radius and center point where the circle is tangent to the sphere.
    Input:

    point: center point of the circle
    tangent: tangent vector to the sphere
    n: number of points on the circle
    radius: radius of the circle

    Output:
    points_c: array of points on the circle (n, 3)
    """
    
    angles = np.linspace(0, 2*np.pi, n)
    normal = np.array(point)/np.linalg.norm(np.array(point)) #assumption: sphere is centered around origin

    tangent1 = np.array(tangent) 
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)
    tangent2 /= np.linalg.norm(tangent2)

    points_c = []
    for a in angles:
        points_c.append((radius*np.cos(a)*tangent1[0]+ radius*np.sin(a)*(tangent2[0]) + point[0], 
                         radius*np.cos(a)*tangent1[1]+ radius*np.sin(a)*(tangent2[1]) + point[1],
                         radius*np.cos(a)*tangent1[2]+ radius*np.sin(a)*(tangent2[2]) + point[2]))
    return np.array(points_c)


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
            cropr = int(np.floor(abs(new_shape[i]) / 2))
            cropl = int(np.ceil(abs(new_shape[i]) / 2))
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



def vel_to_phase_norm(vel):
    print('Normalizing velocity data between -2pi and 0..')
    print('Velocity min', np.min(vel), ' max ', np.max(vel))
    return (vel-np.min(vel))/(np.max(vel) - np.min(vel)) * np.pi - np.pi




def k_space_sampling_timeseries_vel_new(path_order,vel_c,set_):

    #assume velocity image is already complex

    # load order data    
    order = sio.loadmat(path_order, squeeze_me=True)
    N_frames = np.max(order['phs'])
    Nset     = np.max(order['set'])
    assert set_ <= Nset, 'Set number is larger than the maximum set number'

    # get spatial shape of kspacemask
    X = order['NCol']//2
    Y = order['NLin']
    Z = order['NPar']    

    # reshape data to the same size as kspacemask
    vel_c  = adjust_image_size(vel_c, (vel_c.shape[0], X, Y, Z))
    
    print('Transforming velocity data into centered k space ..')
    # complex_img = np.multiply(data_magn, np.exp(1j * vel_to_phase_norm(vel_u)))
    data_ksp = fft_fcts.complex_image_to_centered_kspace(vel_c)
    # data_ksp = fft_fcts.velocity_img_to_centered_kspace(data_vel, data_magn, venc = venc)
    # TODO use min and max values for normalization for reconstruction

    # t_range of mr k-space mask and cfd data
    mr_range  = np.linspace(0, 1, N_frames)
    cfd_range = np.linspace(0, 1, vel_c.shape[0]) 
    
    # Free some memory
    vel_c = None

    print('Sample k space according to order ..')
    sampled_kspace = np.zeros((N_frames, X, Y, Z), dtype = np.complex64)
    for lin, par, phs, set_sample in zip(order['lin'], order['par'],  order['phs'], order['set']):
        if set_sample != set_: continue
        t_idx = np.argmin(np.abs(cfd_range - mr_range[phs-1]))
        sampled_kspace[phs-1, :, lin-1, par-1] = data_ksp[t_idx, :, lin-1, par-1]

    return sampled_kspace, data_ksp

def compute_coil_sensitivity_imgs(coils,  static_mask):
    """
    Compute coil sensitvity for each coil
    """
    print('Calculate coil sensitivity matrices..')
    spatial_res = static_mask.shape
    coil_images = np.zeros((spatial_res[0], spatial_res[1], spatial_res[2], len(coils)), dtype=np.complex128)

    # Compute coil sensitivity maps
    for idx, coil in enumerate(coils):
        coil_images[:,:,:,idx] =  compute_mri_coil_sensitivity(coil, np.argwhere(static_mask), spatial_res).reshape(spatial_res)

    return coil_images

def normalize_coil_sensitivity(coil_images):
    """
    Normalize coil sensitivity images such that the sum of the absolute values of the coil images is 1
    """
    print('Normalize coil sensitivity images..')
    N_coils = coil_images.shape[-1]
    norm_coil_images = np.zeros_like(coil_images)
    for idx in range(N_coils):
        norm_coil_images[:,:,:,idx] = coil_images[:,:,:,idx] / np.sum(np.abs((coil_images[:,:,:,idx])))
    return norm_coil_images


def transform_cfl_format(data):
    """
    Assumption that data is of shape (t, x, y, z, c)
    Transform data to bart cfl format (x, y, z, c, 1, 1, 1, 1, 1, 1, t)
    """
    assert len(data.shape) == 5, 'Data should be of shape (t, x, y, z, c)'
    print('Convert from shape', data.shape, 'to shape', data.transpose(1, 2, 3, 4, 0)[:, :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis,np.newaxis, :].shape)
    return data.transpose(1, 2, 3, 4, 0)[:, :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis,np.newaxis, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process kspace sampling arguments')
    parser.add_argument('-m', '--model', help='Model name (e.g., M1, M2, M3, M4, M5)', required=True)
    parser.add_argument('-v', '--velocity', help='Velocity value (e.g., u, v, w)', choices=['u', 'v', 'w'], required=True)
    args = parser.parse_args()

    if not args.model:
        print("Error: Model name argument is required. Use -m <modelname>")
        sys.exit(1)
    if not args.velocity:
        print("Error: Velocity value argument is required. Use -v <velocity>")
        sys.exit(1)
    
    model_name = args.model
    velocity = args.velocity


    data_dir = '/mnt/c/Users/piacal/Code/SuperResolution4DFlowMRI/Temporal4DFlowNet/data'

    # Define datasets
    path_kmask = f'{data_dir}/kspacemask.h5'
    path_order = f'{data_dir}/order_2mm_20ms.mat'
    path_datamodel = f'{data_dir}/CARDIAC/{model_name}_2mm_step2_static_dynamic.h5'
    save_as = 'results/kspacesampling'
    vel_colnames = [velocity] # this can be set to ['u', 'v', 'w'] to sample all velocities but be aware of memory usage
    x_k, y_k, z_k = 192,126,104
    t_res = 50

    add_noise = False
    save_state = False
    save_cfl = True

    save_coil_sens = f'{save_as}/coil_sensitivity16_sphere'

    # 1. Use coil sensitivity matrix on CFD data
    with h5py.File(path_datamodel, mode = 'r' ) as p1: 
        spatial_res = p1['u'].shape[1:]
        mask = np.array(p1['mask']).squeeze()


    if os.path.isfile(save_coil_sens + '.hdr'):
        print('Load existing coil sensitivity images with ', save_coil_sens )
        coil_images = cfl.readcfl(save_coil_sens).squeeze()
        if len(coil_images.shape) == 5:
            coil_images = coil_images.transpose(4, 0, 1, 2, 3)

        n_coils = coil_images.shape[-1]
        
    else:

        #second approach 
        n_coils = 9
        sphere_radius = 192//2 +10  # similar to coil offset
        coil_radius = 30
        segments = 25
        # put coil centers on a sphere with equally spaced
        s_points, tangents = fibonacci_sphere(n_coils, sphere_radius)
        coils = [define_circle_on_sphere(point, tangent, segments, coil_radius) for point, tangent in zip(s_points, tangents)]

        # center point of phantom
        mask_x, mask_y, mask_z = np.where(mask[0, :, :, :])
        cp_phantom = (np.sum(mask_x)/len(mask_x), np.sum(mask_y)/len(mask_y), np.sum(mask_z)/len(mask_z))

        # adjust coil points around center point of phantom
        for coil in coils:
            coil += np.array(cp_phantom)

        # Visualize the coils
        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.title.set_text('Coil locations in 3D space')

            for idx, coil in enumerate(coils):
                ax.plot(coil[:, 0], coil[:, 1], coil[:, 2], label=f'Coil {idx+1}')

            ax.scatter(mask_x, mask_y, mask_z, c = 'gray', marker = 'o', label= 'Phantom',alpha=0.2 )
            ax.scatter(cp_phantom[0], cp_phantom[1],cp_phantom[2], color='red')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            plt.legend()
            plt.show()
        
        coil_images = compute_coil_sensitivity_imgs(coils,  np.ones((x_k, y_k, z_k)))
        print('coil images', coil_images.shape, coil_images.dtype, coil_images.sum(), coil_images.max(), coil_images.min(), np.sum(np.abs(np.imag(coil_images))))

        # Normalize coil sensitivity images
        coil_images = normalize_coil_sensitivity(coil_images)
        # bart normalize

        print('Save coil sensitivity images..')
        
        if save_cfl:
            cfl.writecfl(f'{save_as}/{save_coil_sens}', transform_cfl_format(np.expand_dims(coil_images, 0)))
            # h5functions.save_to_h5(f'{save_as}/coil_sensitivity6.h5', 'coil_sensitivty', coil_images, expand_dims=False)
            
        coil_images = adjust_image_size(coil_images, (spatial_res[0],spatial_res[1] ,spatial_res[2], n_coils))
        
        magn = mask.copy()

        data = {}
        data_c_sens = {}
        print('Add coil sensitivity to velocity data..')
        with h5py.File(path_datamodel, mode = 'r' ) as p1:
            for vel in vel_colnames:
                data[f'venc_{vel}'] = np.asarray(p1[f'{vel}_max']).squeeze()
                
                # generate coil sensitivity maps
                venc_max = np.max(data[f'venc_{vel}'])
                print('Venc max', venc_max)

                complex_img = np.multiply(magn, np.exp(1j * vel_to_phase_norm(np.asarray(p1[vel]).squeeze())))

                if add_noise:
                    targetSNRdb = np.random.randint(140,170) / 10
                    img_fft = fft_fcts.complex_image_to_centered_kspace(complex_img)
                    img_fft = fft_fcts.add_complex_signal_noise(img_fft, targetSNRdb)
                    complex_img = fft_fcts.centered_kspace_to_complex_img(img_fft)

                    img_fft = None
                    
                # resulting shape (T, X, Y, Z, C)
                data_c_sens[vel] = coil_images[np.newaxis, :, :, :, :] * complex_img[..., np.newaxis] 
        
        # Free up memory
        coil_images = None
        complex_img = None

        # 2. Use k-space mask on CFD data
        for vel in vel_colnames: 
            #make a new k-space for every coil
            k_space_sampled_u = np.zeros((t_res, x_k, y_k, z_k, n_coils), dtype = np.complex64)

            for c in range(n_coils):
                k_space_sampled_u[:, :, :, :, c], ksp_u = k_space_sampling_timeseries_vel_new(path_order, data_c_sens[vel][:, :, :, :, c], set_=1)

            #save
            print('Save k-space sampled data..')
            if save_cfl:
                cfl.writecfl(f'{save_as}/{vel}_kspace{model_name}_hr', transform_cfl_format(k_space_sampled_u))
                print(f'Saved file under {save_as}/{vel}_kspace{model_name}_hr')
                # cfl.writecfl(save_as + f'/{vel}_kspace_17_nonsparse', transform_cfl_format(ksp_u[:, :, :, :, np.newaxis]))

            if False:
                print('Save reconstructions of k-space sampled data without compressed sensing')
                h5functions.save_to_h5(f'{save_as}/{vel}_reconstructed_kspacesampled_sens16.h5','u kspace' , ksp_u, expand_dims=False)

                vel_u_recon, _ = fft_fcts.centered_kspace_to_velocity_img(ksp_u, magn, venc = np.max(data[f'venc_{vel}']))
                h5functions.save_to_h5(f'{save_as}/{vel}_reconstructed_kspacesampled_sens16.h5','u vel not sparse coil 1' , vel_u_recon, expand_dims=False)
                
                # make reconstruction to check result
                vel_u, _ = fft_fcts.centered_kspace_to_velocity_img(k_space_sampled_u[:, :, :, :, 0], magn, venc = np.max(data[f'venc_{vel}']))
                h5functions.save_to_h5(f'{save_as}/{vel}_reconstructed_kspacesampled_sens16.h5','u vel sparse sample' , vel_u, expand_dims=False)

    # bart pics -d5 -D --wavelet haar -R W:7:0:0.0015 -R W:1024:0:0.0075 results/kspacesampling/u_kspace16 results/kspacesampling/coil_sensitivity16_sphere results/kspacesampling/output_test16
    # bart pics -d5 -D --wavelet haar -R W:7:0:0.0015 -R W:1024:0:0.0075 results/kspacesampling/u_kspace16 results/kspacesampling/coil_sensitivity_ones_large_int results/kspacesampling/output_test16
    # bart pics -d5 -D --wavelet haar -R W:7:0:0.0015 -R W:1024:0:0.0075 results/kspacesampling/u_kspace16_small results/kspacesampling/coil_sensitivity_small_ones_nonnorm_9coils results/kspacesampling/output_test16_small

    
    # 3. Reconstruct undersampled k-space with compressed sensing (CS) - save as clf file
    # save undersampled kspace file

    # save sensitivity file

    # 4. Compare to original data