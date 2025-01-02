import os
import sys
import argparse
import math
import numpy as np 
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
import gc
import h5functions
import fft_downsampling as fft_fcts
sys.path.append(os.getenv("BART_TOOLBOX_PATH") + "/python")
from bart import bart, cfl


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
    Aim is to have equally spaced points on the sphere.
    Code is adapted from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    samples: number of points on the sphere
    r: radius of the sphere

    Output:
    points: array of points on the sphere (samples, 3)
    tangent: array of a tangent vector at each point (samples, 3)

    """

    points = []
    tangent = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2 # y goes from 1 to -1
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


def adjust_image_size_centered(image, new_shape):
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

        # cropping
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
    print('Normalizing velocity data between -pi and 0..')
    print('Velocity min', np.min(vel), ' max ', np.max(vel))
    return (vel-np.min(vel))/(np.max(vel) - np.min(vel)) * np.pi - np.pi


def ksp_sampling_timeseries(path_order,data_ksp,sset):

    # load order data    
    order = sio.loadmat(path_order, squeeze_me=True)
    N_frames = np.max(order['phs'])
    Nset     = np.max(order['set'])
    assert sset <= Nset, 'Set number is larger than the maximum set number'

    # get spatial shape of kspacemask
    X = data_ksp.shape[1]
    Y = order['NLin']
    Z = order['NPar']    

    # keep count of phase no. for temporal averaging
    count_phase = np.zeros(N_frames, dtype=np.int16)
    sampling_factor = data_ksp.shape[0]//N_frames

    print('Sample k space according to order ..')
    sampled_kspace = np.zeros((N_frames, X, Y, Z), dtype = np.complex64)
    for lin, par, phs, set_sample in zip(order['lin'], order['par'],  order['phs'], order['set']):
        if set_sample != sset: continue

        t_idx = int((phs-1)*sampling_factor + (count_phase[phs-1] % sampling_factor))
        sampled_kspace[phs-1, :, lin-1, par-1] = data_ksp[t_idx, :, lin-1, par-1]
        count_phase[phs-1] += 1 

    return sampled_kspace

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
    parser.add_argument('-r', '--resolution', help='High or low resolution mode', choices=['lr', 'hr'])
    args = parser.parse_args()

    if not args.model:
        print("Error: Model name argument is required. Use -m <modelname>")
        sys.exit(1)
    if not args.velocity:
        print("Error: Velocity value argument is required. Use -v <velocity>")
        sys.exit(1)
    
    model_name = args.model
    vel = args.velocity
    res = args.resolution

    settings_lr = {'sampling_mask': 'order_2mm_40ms.mat', 't_res': 25, 'noise': True}
    settings_hr = {'sampling_mask': 'order_2mm_20ms.mat', 't_res': 50, 'noise': False}

    if res == 'lr':
        settings = settings_lr
    elif res == 'hr':
        settings = settings_hr


    # Directories
    data_dir = '/mnt/c/Users/piacal/Code/SuperResolution4DFlowMRI/Temporal4DFlowNet/data'
    save_dir = 'results/kspacesampling'
    path_order = f"{data_dir}/{settings['sampling_mask']}"
    path_datamodel = f'{data_dir}/CARDIAC/{model_name}_2mm_step1_static_dynamic.h5'

    load_external_magnitude = False # set to True if magnitude data is loaded from external file, e.g. include invivo magnitude
    path_magnitude = f'{data_dir}/CARDIAC/{model_name}_2mm_step2_cs_invivoP01_hr.h5'

    # Settings
    x_k, y_k, z_k = 72,126,104    # k-space mask size
    n_coils = 8

    add_noise  = settings['noise']
    save_state = False
    save_cfl   = True

    path_csm = f'{save_dir}/csm/csm_64_126_resized'
    path_ksp = f'{save_dir}/final/ksp/{vel}_kspace_{model_name}_{res}_magn80'
    path_cs  = f'{save_dir}/final/output_{vel}_{model_name}_{res}_magn80' 
    path_h5  = f'{save_dir}/final/output_{model_name}_{res}_magn80' 

    velocity_to_set = {'u':1, 'v':2, 'w':3}

    # Load coil sensitivty images, otherwise generate new images
    if os.path.isfile(path_csm + '.hdr'):
        print('Load existing coil sensitivity images with ', path_csm )
        coil_images = cfl.readcfl(path_csm).squeeze().astype(np.complex64)

    else:
        max_dim = np.max([x_k, y_k, z_k])
        coil_images = bart(1, f'bart phantom -S 64 -x {max_dim} -3 --coil HEAD_3D_64CH {path_csm}')

        if save_cfl:
            print('Save coil sensitivity images..')
            cfl.writecfl(path_csm, coil_images)
        
        coil_images = coil_images.squeeze()

    # resize, normalize and pick random n_coils from 64 coils
    random_idxs = np.random.randint(0, coil_images.shape[-1], n_coils)
    cropped_csm = bart(1, f'resize -c 0 {x_k} 1 {y_k} 2 {z_k} 3 {n_coils}', coil_images[:, :, :, random_idxs])
    coil_images = bart(1, 'normalize 8' , cropped_csm)

    print('Add coil sensitivity to velocity data..')
    with h5py.File(path_datamodel, mode = 'r' ) as p1:
        spatial_res = p1['u'].shape[1:]
        mask = np.array(p1['mask']).squeeze().astype(np.int8)

        if not load_external_magnitude:
            magn = np.array(p1['mag_u']).squeeze()*80
        
        venc_u = np.max(np.array(p1[f'u_max']))
        venc_v = np.max(np.array(p1[f'v_max']))
        venc_w = np.max(np.array(p1[f'w_max']))
        venc_max = np.max([venc_u, venc_v, venc_w])

        velocity = adjust_image_size_centered(np.asarray(p1[vel]), (np.asarray(p1[vel].shape[0]), *coil_images.shape[:3]))
        magn     = adjust_image_size_centered(magn, (mask.shape[0], *coil_images.shape[:3]))

        # normalize velocity to [0, 2pi]
        phase    = (velocity/venc_max)*np.pi + np.pi

        # load velocity data and convert to complex image
        complex_img = np.multiply(magn, np.exp(1j * phase)).astype(np.complex64)

        # create coil sensitivity images for each coil
        # (T, X, Y, Z) * (X, Y, Z, C) - resulting shape (T, X, Y, Z, C)
        vel_csm = coil_images[np.newaxis, :, :, :, :] * complex_img[..., np.newaxis] 
    
    # Free up memory
    del complex_img 
    del velocity 
    del phase
    gc.collect()

    # 2. Use k-space mask on CFD data for every coil
    ksp_sampled = np.zeros((settings['t_res'] , x_k, y_k, z_k, n_coils), dtype = np.complex64)
    targetSNRdbs = []

    for c in range(n_coils):
        img_fft = fft_fcts.complex_image_to_centered_kspace(vel_csm[:, :, :, :, c])

        # add noise level with SNR variation for every coil
        if add_noise:
            targetSNRdb = np.random.randint(140,170) / 10
            img_fft = fft_fcts.add_complex_signal_noise(img_fft, targetSNRdb)
            targetSNRdbs.append(targetSNRdb)

        ksp_sampled[:, :, :, :, c] = ksp_sampling_timeseries(path_order, img_fft, sset=velocity_to_set[vel])

    # save
    print('Save k-space sampled data..')
    if save_cfl:
        cfl.writecfl(path_ksp, transform_cfl_format(ksp_sampled))
        print(f'Saved file under {path_ksp}')

    if save_state:
        print('Save reconstructions of k-space sampled data without compressed sensing')
        h5functions.save_to_h5(f'{path_ksp}.h5', f'{vel} kspace' , img_fft, expand_dims=False)

        # this just saves the last coil sensitivty map reconstruction
        complex_img = fft_fcts.centered_kspace_to_complex_img(img_fft)
        h5functions.save_to_h5(f'{path_ksp}_reconstructed.h5',f'{vel} reconstructed' , np.angle(complex_img), expand_dims=False)
        h5functions.save_to_h5(f'{path_ksp}_reconstructed.h5',f'{vel} mag reconstructed' , np.abs(complex_img), expand_dims=False)
            
    # 3. Reconstruct undersampled k-space with compressed sensing (CS) - save as clf file
    print('Run compressed sensing on bart ..')

    cs_result = bart(1, f'pics -d5 -e -S --lowmem-stack 8 --fista --wavelet haar -R W:1024:0:0.0075 --fista_pqr 0.05:0.5:4 -i20', transform_cfl_format(ksp_sampled), coil_images)
    # cs_result = bart(1, f'pics -d5 -e -S --lowmem-stack 8 -l2 0.0001 --pridu -P 1e-6 --wavelet haar -R W:1024:0:0.0075 -i 10', transform_cfl_format(ksp_sampled), coil_images)
    # cs_result = bart(1, f'pics -d5 -e -S --lowmem-stack 8 --fista -i 30 --wavelet haar -R W:1024:0:0.0075 ', transform_cfl_format(ksp_sampled), coil_images)
    print(f'Saving compressed sensing results at {path_cs}.')
    cfl.writecfl(path_cs, cs_result)
    
    # 4. Convert and save as h5 file

    # crop to smaller image size
    cs_result = cs_result.squeeze().transpose(3, 0, 1, 2)
    cs_result = adjust_image_size_centered(cs_result, (settings['t_res'], *spatial_res))
    
    # convert to phase and velocity data [m/s]
    # output of compressed sensing is between [-pi, pi]
    vel_cs = np.angle(cs_result)/np.pi*venc_max
    magn_cs = np.abs(cs_result)

    # set non-fluid region to zero for high-resolution data
    if res == 'hr':
        vel_cs = np.multiply(vel_cs, mask[::2])

    # save as h5 file
    h5functions.save_to_h5(f'{path_h5}.h5', vel, vel_cs, expand_dims=False)
    h5functions.save_to_h5(f'{path_h5}.h5', f'mag_{vel}', magn_cs,  expand_dims=False)
    h5functions.save_to_h5(f'{path_h5}.h5', 'venc_max', venc_max, expand_dims=True)
    h5functions.save_to_h5(f'{path_h5}.h5', 'mask', mask, expand_dims=False)
    if res == 'lr':
        h5functions.save_to_h5(f'{path_h5}.h5', 'targetSNRdb', np.array(targetSNRdbs), expand_dims=True)