import time
import math
import numpy as np
from matplotlib import pyplot as plt

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


def add_complex_signal_noise(imgfft, targetSNRdb):
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
    add_complex_noise =True
    # adding noise on the real and complex image
    # print("--------------Adding Gauss noise to COMPLEX signal----------------")

    # Deconstruct the complex numbers into real and imaginary
    mag_signal = np.abs(imgfft)
    
    signal_power = np.mean((mag_signal) ** 2)

    logSNR = targetSNRdb / 10
    snr = 10 ** logSNR

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
        real_signal = real_signal + real_noise
        imj_signal  = imj_signal + imj_noise
        
        # reconstruct the image back to complex numbers
        imgfft = real_signal + 1j * imj_signal

    return imgfft

def downsample_complex_img_TBD(complex_img, crop_ratio, targetSNRdb, temporal_downsampling = False):
    imgfft = np.fft.fftn(complex_img)
    
    if not temporal_downsampling:
        imgfft = rectangular_crop3d(imgfft, crop_ratio)
    
    shifted_mag  = 20*np.log(np.fft.fftshift(np.abs(imgfft)))

    # add noise on freq domain
    imgfft = add_complex_signal_noise(imgfft, targetSNRdb)

    # inverse fft to image domain
    new_complex_img = np.fft.ifftn(imgfft)

    return new_complex_img, shifted_mag


def rescale_magnitude_on_ratio(new_mag, old_mag):
    old_mag_flat = np.reshape(old_mag, [-1])
    new_mag_flat = np.reshape(new_mag, [-1])

    rescale_ratio = new_mag_flat.shape[0] / old_mag_flat.shape[0]

    return new_mag * rescale_ratio
    
def downsample_phase_img_TBD(velocity_img, mag_image, venc, crop_ratio, targetSNRdb, temporal_downsampling = False):
    # convert to phase
    phase_image = velocity_img / venc * math.pi

    complex_img = np.multiply(mag_image, np.exp(1j*phase_image))
    # -----------------------------------------------------------
    new_complex_img, shifted_freqmag = downsample_complex_img(complex_img, crop_ratio, targetSNRdb, temporal_downsampling)
    # -----------------------------------------------------------

    # Get the MAGnitude and rescale
    new_mag = np.abs(new_complex_img)
    new_mag = rescale_magnitude_on_ratio(new_mag, mag_image)

    # Get the PHASE
    new_phase = np.angle(new_complex_img)

    # Get the velocity image
    new_velocity_img = new_phase / math.pi * venc

    return new_velocity_img, new_mag

def velocity_img_to_kspace(vel_img,mag_image, venc):

    # convert to phase
    phase_image = vel_img / venc * math.pi

    # convert to complex image
    complex_img = np.multiply(mag_image, np.exp(1j*phase_image))

    # fft
    imgfft = np.fft.fftn(complex_img)

    return imgfft

def velocity_img_to_centered_kspace(vel_img,mag_image, venc):
    # convert to phase
    phase_image = vel_img / venc * math.pi

    # convert to complex image
    complex_img = np.multiply(mag_image, np.exp(1j*phase_image))

    # ifftshift
    complex_img = np.fft.ifftshift(complex_img)

    # fft
    imgfft = np.fft.fftn(complex_img)

    # shift img to center
    imgfft = np.fft.fftshift(imgfft)

    return imgfft

def centered_kspace_to_velocity_img(imgfft, mag_image, venc):

    # shift img to center
    imgfft = np.fft.ifftshift(imgfft)

    new_complex_img = np.fft.ifftn(imgfft)

    new_complex_img = np.fft.fftshift(new_complex_img)
    
    # Get the MAGnitude and rescale
    new_mag = np.abs(new_complex_img)
    new_mag = rescale_magnitude_on_ratio(new_mag, mag_image)

    # Get the PHASE
    new_phase = np.angle(new_complex_img)

    # Get the velocity image
    new_velocity_img = new_phase / math.pi * venc
    

    return new_velocity_img, new_mag

def kspace_to_velocity_img(imgfft, mag_image, venc):
    
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

def noise_and_downsampling(vel_img, mag_image, venc, targetSNRdb, add_noise = True, spatial_crop_ratio = 1.0):

    # convert to kspace
    imgfft = velocity_img_to_kspace(vel_img,mag_image, venc)

    # downsample the kspace by rectangular cropping
    if spatial_crop_ratio < 1.0:
        print("Downsampling by rectangular cropping")
        imgfft = rectangular_crop3d(imgfft, spatial_crop_ratio)

    # add noise on freq domain
    if add_noise:
        print("Adding noise")
        shifted_mag  = 20*np.log(np.fft.fftshift(np.abs(imgfft)))
        imgfft = add_complex_signal_noise(imgfft, targetSNRdb)

    # inverse fft to image domain
    new_velocity_img, new_mag = kspace_to_velocity_img(imgfft, mag_image, venc)

    return new_velocity_img, new_mag



# -------------test function----------------------------------------------
# TODO put this in a test file
def test_transform():
    """test if going from kspace and back to image domain is lossless"""
    # Create a random velocity image
    vel_img = np.random.rand(100,100,100)
    mag_image = np.random.rand(100,100,100)
    venc = 100

    # convert to kspace
    imgfft = velocity_img_to_kspace(vel_img,mag_image, venc)
    # inverse fft to image domain
    new_velocity_img, new_mag = kspace_to_velocity_img(imgfft, mag_image, venc)

    # check if the original and new are the same
    assert np.allclose(vel_img, new_velocity_img) # check if the velocity image is the same
    assert np.allclose(mag_image, new_mag) # check if the magnitude image is the same

    print("Test 1 passed")

    print("----Test 2----")

    # Create a random velocity image
    vel_img = np.arange(0, 100*100*100).reshape((100,100,100))
    mag_image = np.random.rand(100,100,100)
    mask = np.zeros_like(mag_image)
    mask[25:75,15:34,25:75] = 1

    vel_img = vel_img * mask

    venc = 100

    new_vel_img, _ = noise_and_downsampling(vel_img, mag_image, venc, 17, add_noise = False, spatial_crop_ratio = 1.0)
    fig = plt.subplot(121)
    plt.subplot(121)
    plt.imshow(new_vel_img[:,:,50])
    plt.subplot(122)
    plt.imshow(vel_img[:,:,50])
    plt.show()
    assert np.allclose(vel_img, new_vel_img) # check if the velocity image is the same

    print("Test 2 passed")


# test_transform()