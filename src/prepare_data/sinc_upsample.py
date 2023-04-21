import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import h5functions

def plot_images(u, v, w, u_mag, v_mag, w_mag):
    print(u.shape)
    # slice_to_show = np.index_exp[:, 4, :]
    slice_to_show = np.index_exp[16, :, :]


    plot_phasemag(1, 'U', u, u_mag, slice_to_show)
    plot_phasemag(4, 'V', v, v_mag, slice_to_show)
    plot_phasemag(7, 'W', w, w_mag, slice_to_show)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def plot_phasemag(starting_idx, velocity_name, u, u_mag, slice_to_show):
    plt.subplot(330+starting_idx),plt.imshow(u[slice_to_show], cmap = 'jet')
    plt.colorbar()
    plt.title(velocity_name)

    plt.subplot(330+starting_idx+1),plt.imshow(u_mag[slice_to_show], cmap = 'gnuplot2', clim=(0, 150))
    plt.colorbar()
    plt.title('{} Mag'.format(velocity_name))

    u_reshape = np.reshape(u, [-1])
    plt.subplot(330+starting_idx+2),plt.hist(u_reshape, bins=np.arange(min(u_reshape), max(u_reshape) + 1, step=0.2))


def pad_with_zero(f, upsample_rate):
    half_x = f.shape[0] // 2
    half_y = f.shape[1] // 2
    half_z = f.shape[2] // 2
    
    # shift it to make it easier to crop, otherwise we need to concat half left and half right
    new_kspace = np.fft.fftshift(f)
    new_kspace = np.pad(new_kspace, ((half_x, half_x), (half_y, half_y), (half_z, half_z)), 'constant')
    # shift it back to original freq domain
    new_kspace = np.fft.fftshift(new_kspace)
     
    return new_kspace

def pad_with_zero_temp(f,upsample_rate ):
    half_t = f.shape[0]//2
    
    # shift it to make it easier to crop, otherwise we need to concat half left and half right
    new_kspace = np.fft.fftshift(f)
    new_kspace = np.pad(new_kspace, ((half_t, half_t), (0, 0), (0, 0), (0, 0)), 'constant')
    # shift it back to original freq domain
    new_kspace = np.fft.fftshift(new_kspace)
     
    return new_kspace


def rectangular_crop3d(f, crop_ratio):
    half_x = f.shape[0] // 2
    half_y = f.shape[1] // 2
    half_z = f.shape[2] // 2
    
    x_crop = int(half_x * crop_ratio)
    y_crop = int(half_y * crop_ratio)
    z_crop = int(half_z * crop_ratio)

    # shift it to make it easier to crop, otherwise we need to concat half left and half right
    new_kspace = np.fft.fftshift(f)
    new_kspace = new_kspace[half_x-x_crop:half_x+x_crop, half_y-y_crop:half_y+y_crop, half_z-z_crop : half_z+z_crop]
    # shift it back to original freq domain
    new_kspace = np.fft.fftshift(new_kspace)
     
    return new_kspace

def sinc_interpolation(complex_img, upsample_rate, temporal_upsampling):
    imgfft = np.fft.fftn(complex_img)

    if imgfft.ndim == 3:
        imgfft = pad_with_zero(imgfft, upsample_rate)
    if imgfft.ndim == 4 and temporal_upsampling:
        imgfft = pad_with_zero_temp(imgfft, upsample_rate)
    
    shifted_mag  = 20*np.log(np.fft.fftshift(np.abs(imgfft)))

    # inverse fft to image domain
    new_complex_img = np.fft.ifftn(imgfft)

    return new_complex_img, shifted_mag

def downsample_complex_img(complex_img, crop_ratio):
    imgfft = np.fft.fftn(complex_img)

    if imgfft.ndim == 3:
        imgfft = rectangular_crop3d(imgfft, crop_ratio)

    shifted_mag  = 20*np.log(np.fft.fftshift(np.abs(imgfft)))

    # inverse fft to image domain
    new_complex_img = np.fft.ifftn(imgfft)

    return new_complex_img, shifted_mag

def fft_upsample(velocity_img, mag_image, venc, upsample_rate, temporal_upsampling):
    # convert to phase
    phase_image = velocity_img / venc * math.pi
  
    complex_img = np.multiply(mag_image, np.exp(1j*phase_image))
    
    # -----------------------------------------------------------
    new_complex_img, shifted_freqmag = sinc_interpolation(complex_img, upsample_rate, temporal_upsampling)
    # -----------------------------------------------------------

    # Get the MAGnitude and rescale
    new_mag = np.abs(new_complex_img)
    # new_mag = rescale_magnitude_on_ratio(new_mag, mag_image)

    # Get the PHASE
    new_phase = np.angle(new_complex_img)
    
    # Get the velocity image
    new_velocity_img = new_phase / math.pi * venc

    return new_velocity_img, new_mag

def rescale_magnitude_on_ratio(new_mag, old_mag):
    old_mag_flat = np.reshape(old_mag, [-1])
    new_mag_flat = np.reshape(new_mag, [-1])

    rescale_ratio = new_mag_flat.shape[0] / old_mag_flat.shape[0]

    return new_mag * rescale_ratio
    
def fft_downsample(velocity_img, mag_image, venc, crop_ratio):
    # convert to phase
    phase_image = velocity_img / venc * math.pi
  
    complex_img = np.multiply(mag_image, np.exp(1j*phase_image))
    
    # -----------------------------------------------------------
    new_complex_img, shifted_freqmag = downsample_complex_img(complex_img, crop_ratio)
    # -----------------------------------------------------------

    # Get the MAGnitude and rescale
    new_mag = np.abs(new_complex_img)
    new_mag = rescale_magnitude_on_ratio(new_mag, mag_image)

    # Get the PHASE
    new_phase = np.angle(new_complex_img)
    
    # Get the velocity image
    new_velocity_img = new_phase / math.pi * venc

    return new_velocity_img, new_mag

if __name__ == '__main__':


    base_path = 'Temporal4DFlowNet/data/CARDIAC'
    # Put your path to Hires Dataset
    input_filepath  =  f'{base_path}/M4_2mm_step2_static_dynamic_noise.h5'

    output_filename = f'{base_path}/M4_2mm_step2_static_dynamic_noise_sinc.h5' 
    
    temporal_upsampling = True
    
    # setting the seeds for both random and np random, so we get the same random order on dataset everytime
    upsample_rate = 2

    # Load the mask once
    with h5py.File(input_filepath, mode = 'r' ) as hf:        
        data_count = len(hf.get("u"))

    
    for idx in range(1): # TODO changes here since we want to work with 4D data
        print("Processing data row", idx)
        
        # Create the magnitude based on the possible values
        ## This is a part of augmentation to make sure we have varying magnitude
        
        # Load the velocity U V W from H5
        with h5py.File(input_filepath, mode = 'r' ) as hf:
            print(hf.keys())
            mask = np.asarray(hf['mask'][0])

            lr_u = np.asarray(hf['u'])[::2] #downsample noisy image
            lr_v = np.asarray(hf['v'])[::2] #downsample noisy image
            lr_w = np.asarray(hf['w'])[::2] #downsample noisy image
        
            mag_u = np.asarray(hf['mag_u'])[::2]
            mag_v = np.asarray(hf['mag_v'])[::2]
            mag_w = np.asarray(hf['mag_w'])[::2]

            venc_u = np.max(np.asarray(hf['venc_u'])[::2])
            venc_v = np.max(np.asarray(hf['venc_v'])[::2])
            venc_w = np.max(np.asarray(hf['venc_w'])[::2])

           
        if upsample_rate > 1:
            print("Upsampling...")
            # Upsample
            hr_u, hrmag_u = fft_upsample(lr_u, mag_u, venc_u, upsample_rate, temporal_upsampling)
            hr_v, hrmag_v = fft_upsample(lr_v, mag_v, venc_v, upsample_rate, temporal_upsampling)
            hr_w, hrmag_w = fft_upsample(lr_w, mag_w, venc_w, upsample_rate, temporal_upsampling)

            # VIsual check for the noise distribution
            # plot_images(hr_u, hr_v, hr_w, hrmag_u, hrmag_v, hrmag_w)

            # Save the upsampled images
            h5functions.save_to_h5(output_filename, "u", hr_u)
            h5functions.save_to_h5(output_filename, "v", hr_v)
            h5functions.save_to_h5(output_filename, "w", hr_w)
            print('Upsampled shape: ', hr_u.shape, hr_v.shape, hr_w.shape)

            # h5utils.save_to_h5(output_filename, "mag_u_sinc", hrmag_u)
            # h5utils.save_to_h5(output_filename, "mag_v_sinc", hrmag_v)
            # h5utils.save_to_h5(output_filename, "mag_w_sinc", hrmag_w)
        elif upsample_rate < 1:
            print("Downsampling...")
            hr_u, hrmag_u = fft_downsample(lr_u, mag_u, venc_u, upsample_rate)
            hr_v, hrmag_v = fft_downsample(lr_v, mag_v, venc_v, upsample_rate)
            hr_w, hrmag_w = fft_downsample(lr_w, mag_w, venc_w, upsample_rate)

            h5utils.save_to_h5(output_filename, "lr_u", hr_u)
            h5utils.save_to_h5(output_filename, "lr_v", hr_v)
            h5utils.save_to_h5(output_filename, "lr_w", hr_w)

            h5utils.save_to_h5(output_filename, "mag_u", hrmag_u)
            h5utils.save_to_h5(output_filename, "mag_v", hrmag_v)
            h5utils.save_to_h5(output_filename, "mag_w", hrmag_w)
            
            h5utils.save_to_h5(output_filename, "venc_u", venc_u)
            h5utils.save_to_h5(output_filename, "venc_v", venc_v)
            h5utils.save_to_h5(output_filename, "venc_w", venc_w)
        else:
            print("Upsample rate", upsample_rate, "not supported")

    print("Done!")


