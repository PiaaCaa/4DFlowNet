from prepare_data.fft_downsampling import *

# -------------test function----------------------------------------------

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