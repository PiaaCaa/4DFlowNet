import math
import numpy as np

def rectangular_crop3d(f, crop_ratio):
    """
    Crop a 3D array in the frequency domain using a rectangular shape in the k-space center.

    Args:
        f (ndarray): The input 3D array in the frequency domain.
        crop_ratio (float): The ratio of the crop size to the original size.

    Returns:
        ndarray: The cropped 3D array in the frequency domain.
    """

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


def add_complex_signal_noise(imgfft, targetSNRdb, add_complex_noise=False):
    """
        Add gaussian noise to real and imaginary signal
        The sigma is assumed to be the same (Gudbjartsson et al. 1995)

        SNRdb = 10 log SNR
        SNRdb / 10 = log SNR
        SNR = 10 ^ (SNRdb/10)
        
        Pn = Pn / SNR
        Pn = variance
        
        Relation of std and Pn is taken from Matlab Communication Toolbox, awgn.m

        For complex signals, we can use the equation above.
        If we do it in real and imaginary signal, half the variance is in real and other half in in imaginary.
        
        https://www.researchgate.net/post/How_can_I_add_complex_white_Gaussian_to_the_signal_with_given_signal_to_noise_ratio
        "The square of the signal magnitude is proportional to power or energy of the signal.
        SNR is the ratio of this power to the variance of the noise (assuming zero-mean additive WGN).
        Half the variance is in the I channel, and half is in the Q channel.  "

    """    
    # adding noise on the real and complex image
    # print("--------------Adding Gauss noise to COMPLEX signal----------------")

    # Deconstruct the complex numbers into real and imaginary
    mag_signal = np.abs(imgfft)
    
    signal_power = np.mean((mag_signal) ** 2)

    logSNR = targetSNRdb / 10
    snr = 10 ** logSNR
    print(f'SNR is {snr:.1f} with targetsnrdb is {targetSNRdb}')

    noise_power = signal_power / snr

    if add_complex_noise:
        
        sigma  = np.sqrt(noise_power)
        # print('Target SNR ', targetSNRdb, "db, sigma(complex)", sigma, "shape:" ,imgfft.shape)

        # add the noise to the complex signal directly
        gauss_noise = np.random.normal(0, sigma, imgfft.shape)
        imgfft += gauss_noise
    else:
        # Add the noise to real and imaginary separately
        sigma  = np.sqrt(noise_power/2) 
        # print('Target SNR ', targetSNRdb, "db, sigma (real/imj)", sigma)
        
        real_signal = np.real(imgfft)
        imj_signal = np.imag(imgfft)
        
        real_noise = np.random.normal(0, sigma, real_signal.shape)
        imj_noise  = np.random.normal(0, sigma, imj_signal.shape)
        
        # add the noise to both components
        real_signal += real_noise
        imj_signal  += imj_noise
        
        # reconstruct the image back to complex numbers
        imgfft = real_signal + 1j * imj_signal

    return imgfft


def rescale_magnitude_on_ratio(new_mag, old_mag):
    old_mag_flat = np.reshape(old_mag, [-1])
    new_mag_flat = np.reshape(new_mag, [-1])

    rescale_ratio = new_mag_flat.shape[0] / old_mag_flat.shape[0]

    return new_mag * rescale_ratio


def velocity_img_to_kspace(vel_img, mag_image, venc):
    """
    Convert velocity image to k-space representation.

    Parameters:
    vel_img (ndarray): Velocity image.
    mag_image (ndarray): Magnitude image.
    venc (float): Velocity encoding value.

    Returns:
    ndarray: K-space representation of the velocity image.
    """

    # Convert to phase
    phase_image = vel_img / venc * math.pi

    # Convert to complex image
    complex_img = np.multiply(mag_image, np.exp(1j * phase_image))

    # Perform FFT
    imgfft = np.fft.fftn(complex_img)

    return imgfft


def velocity_img_to_centered_kspace(vel_img, mag_image, venc, normalize_0_2pi = False):
    """
    Convert velocity image to centered k-space, i.e. including shift

    Args:
        vel_img (ndarray): Velocity image.
        mag_image (ndarray): Magnitude image.
        venc (float): Velocity encoding scale.

    Returns:
        ndarray: Centered k-space representation of the velocity image.
    """
    if len(vel_img.shape) == 3:
        axes = (0, 1, 2)
    elif len(vel_img.shape) == 4:
        axes = (1, 2, 3)
    else:
        print("Error: Unsupported number of dimensions, please extend function to ", len(vel_img.shape), " dimensions.")

    # convert to phase
    if normalize_0_2pi:
        phase_image = vel_img / venc * np.pi + np.pi 
    else:
        phase_image = vel_img / venc * np.pi 

    # convert to complex image
    complex_img = np.multiply(mag_image, np.exp(1j*phase_image))

    # ifftshift
    complex_img = np.fft.ifftshift(complex_img, axes=axes)

    # fft
    imgfft = np.fft.fftn(complex_img, axes = axes)

    # shift img to center
    imgfft = np.fft.fftshift(imgfft, axes=axes)

    return imgfft

def complex_image_to_centered_kspace(complex_img):

    if len(complex_img.shape) == 3:
        axes = (0, 1, 2)
    elif len(complex_img.shape) == 4:
        axes = (1, 2, 3)
    else:
        print("Error: Unsupported number of dimensions, please extend function to ", len(complex_img.shape), " dimensions.")

    # ifftshift
    complex_img = np.fft.ifftshift(complex_img, axes=axes)

    # fft
    imgfft = np.fft.fftn(complex_img, axes = axes)

    # shift img to center
    imgfft = np.fft.fftshift(imgfft, axes=axes).astype(np.complex64)
    return imgfft

def centered_kspace_to_complex_img(imgfft):
    if len(imgfft.shape) == 3:
        axes = (0, 1, 2)
    elif len(imgfft.shape) == 4:
        axes = (1, 2, 3)
    else:
        print("Error: Unsupported number of dimensions, please extend function to ", len(imgfft.shape), " dimensions.")

    # ifftshift
    imgfft = np.fft.ifftshift(imgfft, axes=axes)

    # ifft
    complex_img = np.fft.ifftn(imgfft, axes = axes)

    # shift img to center
    complex_img = np.fft.fftshift(complex_img, axes=axes)
    return complex_img




def centered_kspace_to_velocity_img(imgfft, mag_image, venc, normalized_0_2pi=False):
    """
    Convert centered k-space data to velocity image.

    Args:
        imgfft (ndarray): The centered k-space data.
        mag_image (ndarray): The magnitude image used for rescaling.
        venc (float): The velocity encoding scale.

    Returns:
        ndarray: The velocity image.
        ndarray: The rescaled magnitude image.
    """
    if len(imgfft.shape) == 3:
        axes = (0, 1, 2)
    elif len(imgfft.shape) == 4:
        axes = (1, 2, 3)
    else:
        print("Error: Unsupported number of dimensions, please extend function to ", len(imgfft.shape), " dimensions.")

    # shift img to center
    imgfft = np.fft.ifftshift(imgfft, axes=axes)
    imgfft = np.fft.ifftn(imgfft, axes=axes)
    new_complex_img = np.fft.fftshift(imgfft, axes=axes)
    
    # Get the Magnitude and rescale
    new_mag = np.abs(new_complex_img)
    new_mag = rescale_magnitude_on_ratio(new_mag, mag_image)

    # Get the PHASE
    new_phase = np.angle(new_complex_img)

    # Get the velocity image
    #Note changes to scaling betwen 0 and 2*pi
    if normalized_0_2pi:
        new_velocity_img = (new_phase- np.pi) / (np.pi) * venc
    else:
        new_velocity_img = new_phase / math.pi * venc
    
    return new_velocity_img, new_mag

def kspace_to_velocity_img(imgfft, mag_image, venc):
    """
    Converts k-space data to velocity image.

    Parameters:
    imgfft (ndarray): The input k-space data.
    mag_image (ndarray): The magnitude image used for rescaling.
    venc (float): The velocity encoding scale.

    Returns:
    ndarray: The velocity image.
    ndarray: The rescaled magnitude image.
    """

    new_complex_img = np.fft.ifftn(imgfft)

    # Get the MAGnitude and rescale
    new_mag = np.abs(new_complex_img)
    new_mag = rescale_magnitude_on_ratio(new_mag, mag_image)

    # Get the PHASE
    new_phase = np.angle(new_complex_img)

    # Get the velocity image
    new_velocity_img = new_phase / math.pi * venc

    return new_velocity_img, new_mag

def whole_script_TBD(vel_img, mag_image, venc, targetSNRdb):

    # convert to phase
    phase_image = vel_img / venc * math.pi

    # convert to complex image
    complex_img = np.multiply(mag_image, np.exp(1j*phase_image))

    # fft 
    imgfft = np.fft.fftn(complex_img)
    shifted_mag  = 20*np.log(np.fft.fftshift(np.abs(imgfft)))

    # add noise on freq domain
    imgfft = add_complex_signal_noise(imgfft, targetSNRdb)

    # inverse fft to image domain
    new_complex_img = np.fft.ifftn(imgfft)

    # Get the MAGnitude and rescale
    new_mag = np.abs(new_complex_img)
    new_mag = rescale_magnitude_on_ratio(new_mag, mag_image)

    # Get the PHASE
    new_phase = np.angle(new_complex_img)

    # Get the velocity image
    new_velocity_img = new_phase / math.pi * venc

    return new_velocity_img, new_mag

def noise_and_downsampling(vel_img, mag_image, venc, targetSNRdb, add_noise=True, spatial_crop_ratio=1.0):
    """
    Apply noise and downsampling to the velocity image.

    Args:
        vel_img (ndarray): The velocity image.
        mag_image (ndarray): The magnitude image.
        venc (float): The velocity encoding value.
        targetSNRdb (float): The target signal-to-noise ratio in decibels.
        add_noise (bool, optional): Whether to add noise to the frequency domain. Defaults to True.
        spatial_crop_ratio (float, optional): The ratio for rectangular cropping in the spatial domain. Defaults to 1.0.

    Returns:
        ndarray: The new velocity image after noise and downsampling.
        ndarray: The new magnitude image after noise and downsampling.
    """
    
    # convert to kspace
    imgfft = velocity_img_to_kspace(vel_img, mag_image, venc)

    # downsample the kspace by rectangular cropping
    if spatial_crop_ratio < 1.0:
        print("Downsampling by rectangular cropping")
        imgfft = rectangular_crop3d(imgfft, spatial_crop_ratio)

    # add noise on freq domain
    if add_noise:
        print("Adding noise")
        shifted_mag = 20 * np.log(np.fft.fftshift(np.abs(imgfft)))
        imgfft = add_complex_signal_noise(imgfft, targetSNRdb)

    # inverse fft to image domain
    new_velocity_img, new_mag = kspace_to_velocity_img(imgfft, mag_image, venc)

    return new_velocity_img, new_mag

