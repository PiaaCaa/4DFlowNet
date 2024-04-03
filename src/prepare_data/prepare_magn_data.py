import numpy as np
import os
from PIL import Image
import h5py
import h5functions 
from matplotlib import pyplot as plt
import scipy.ndimage
from scipy.ndimage import map_coordinates
# from skimage.draw import random_shapes

def convert_rgb_to_gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_jpg_image(file_path):
    img = Image.open(file_path)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def get_paths(directory, ending = '.jpg'):
    "return all files with given ending in the directory and subdirectories"
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ending):
                file_paths.append(os.path.join(root, file))
    return file_paths

def generate_random_4D_mag_images(directory, save_path, in_silico_model_size):

    # get all file paths
    file_paths = get_paths(directory)

    # in_silico_model_size = {'M1': (50, 72, 70, 76), 'M2': (50, 84, 60, 96), 'M3': (50, 72, 82, 84), 'M4': (50, 62, 58, 84) }
    models = list(in_silico_model_size.keys())#['M1', 'M2', 'M3', 'M4']
    new_mag_data = {}

    # set onto first model
    model = models[0]
    t, x, y, z = in_silico_model_size[model]
    new_mag_data[model] = np.zeros((t, x, y, z))

    if len(file_paths) == 0:
        print("No files found in the directory")
        print(os.getcwd())
        exit(0)

    magn_ranges = np.asarray([60, 80, 120, 180, 240]) # in px values [0-4095]

    i = 0
    # loop through all files
    for f, file_path in enumerate(file_paths):
        i = i % (new_mag_data[model].shape[-1] - 1) # filled until z axis is full

        #check if limit is reached (i.e. same input shape) and get new model
        if i == 0 and f != 0:
            if model == models[-1]:
                break
            model = models[models.index(model) + 1]
            t, x, y, z = in_silico_model_size[model]
            new_mag_data[model] = np.zeros((t, x, y, z))
        
        #-------load and convert image-----

        # load the image
        img = load_jpg_image(file_path)

        # convert the image to grayscale
        img_gray = convert_rgb_to_gray(img)

        img_x, img_y = img_gray.shape

        #normalize the image
        img_gray = img_gray / 255

        # set to magnitude range of mri data
        mri_magn = np.random.choice(magn_ranges)

        img_gray = img_gray * mri_magn
        
        # skip too small images
        if img_x < x or img_y < y:
            continue
        
        #---------use same image for multiple frames-------------
        for frame in range(t):
            
            x_rand = np.random.randint(0, img_x-x)
            y_rand = np.random.randint(0, img_y-y)

            # crop the image in random position
            img_gray_resized = img_gray[x_rand:x_rand+x, y_rand:y_rand+y]

            # add the image to the dictionary
            new_mag_data[model][frame, :, :, i] = img_gray_resized
        
        i += 1
    
    for model in models:
        print(model, new_mag_data[model].shape)
        # save the data
        h5functions.save_to_h5(save_path, model, new_mag_data[model], expand_dims=False)


def generate_static_4D_mag_images(directory, save_path, in_silico_model_size):

    # get all file paths
    file_paths = get_paths(directory)

    models = list(in_silico_model_size.keys())

    if len(file_paths) == 0:
        print("No files found in the directory")
        print(os.getcwd())
        exit(0)

    #choose random image for each model
    random_paths = [np.random.choice(file_paths) for _ in range(len(models))]

    magn_ranges = np.asarray([60, 80, 120, 180, 240]) # in px values [0-4095]

    # loop through all models
    for m, model in enumerate(models):
        t, x, y, z = in_silico_model_size[model]

        #load one magn image for each model
        file_path = random_paths[m]
        
        #-------load and convert image-----

        # load the image
        img = load_jpg_image(file_path)

        # convert the image to grayscale
        img_gray = convert_rgb_to_gray(img)

        img_x, img_y = img_gray.shape

        # normalize 
        img_gray = img_gray / 255

        # set to magnitude range of mri data
        mri_magn = np.random.choice(magn_ranges)

        img_gray = img_gray * mri_magn
        
        # image is too small
        if img_x < x or img_y < y:
            print(f"Image {os.path.basename(file_path)} too small")
        
        # crop image randomly
        x_rand = np.random.randint(0, img_x-x)
        y_rand = np.random.randint(0, img_y-y)

        # crop the image in random position
        img_gray_resized = img_gray[x_rand:x_rand+x, y_rand:y_rand+y]

        #------stack along z and t axis for 3D construction--------

         # stack in 3D along z axis
        img_gray_resized = np.repeat(img_gray_resized[:, :, None], z, axis=-1)

        print(img_gray_resized.shape)

        # use same image in time for temporal coherency
        img_gray_resized = np.repeat(img_gray_resized[None, :, :, :], t, axis=0)

        print(img_gray_resized.shape)

        assert((t, x, y, z) == img_gray_resized.shape) #shape of new magnitude should match datamodel shape

        #-------save--------
        print(f'------- Magnitude to correspomding Model {model} saved to {save_path} -------')
        h5functions.save_to_h5(save_path, model, img_gray_resized, expand_dims=False)

def rotate_2Dimage_in_3D(image, angle, axis):
    "rotate the image by the given angle"
    return scipy.ndimage.rotate(image, angle, axis, reshape=False)

def fully_rotate_2Dimage_in_3D_with_interpolation(img):
    "rotate the image by the given angle create a cylindical image in 3D space"
    " Code is from Alexander Fyrdahl"
    img2D = np.tile(img[:, :, np.newaxis], (1, 1, img.shape[1]))
    x, y, z = np.mgrid[0:img2D.shape[0], 0:img2D.shape[1], 0:img2D.shape[2]]
    xc = x - img2D.shape[0] / 2
    zc = z - img2D.shape[2] / 2
    r = np.sqrt(xc**2 + zc**2)
    angle = 2 * np.pi
    xRot = r * np.cos(angle) + img2D.shape[0] / 2
    zRot = r * np.sin(angle) + img2D.shape[2] / 2
    coords = np.array([xRot.flatten(), y.flatten(), zRot.flatten()])
    img3D = map_coordinates(img2D, coords, order=1, mode='constant', cval=0).reshape(img2D.shape)
    return img3D


if __name__ == "__main__":


    # path to the directory with the images
    directory = 'data/clouds'

    save_path = 'data/cloud_magn_data_4D_spatial_rotated_M1-M6.h5'

    # get all file paths
    file_paths = get_paths(directory, ending='.JPEG')

    in_silico_model_size = {'M1': (50, 72, 70, 76), 'M2': (50, 84, 60, 96), 'M3': (50, 72, 82, 84), 'M4': (50, 62, 58, 84), 'M5': (50, 70, 70, 71), 'M6': (50, 70, 70, 70)}
    models = list(in_silico_model_size.keys())

    new_mag_data = {}

    if len(file_paths) == 0:
        print("No files found in the directory")
        print(os.getcwd())
        exit(0)

    #choose random image for each model
    random_paths = [np.random.choice(file_paths) for _ in range(len(models))]

    # random_paths = [f'{directory}/n09247410_1208.JPEG', f'{directory}/n09247410_2033.JPEG', f'{directory}/n09247410_3183.JPEG', f'{directory}/n09247410_3507.JPEG']

    magn_ranges = np.asarray([60, 80, 120, 180, 240]) # in px values [0-4095]

    # loop through all models
    for m, model in enumerate(models):
        t, x, y, z = in_silico_model_size[model]

        #load one magn image for each model
        file_path = random_paths[m]
        
        #-------load and convert image-----

        # load the image
        img = load_jpg_image(file_path)

        # convert the image to grayscale
        img_gray = convert_rgb_to_gray(img)

        img_x, img_y = img_gray.shape

        # normalize 
        img_gray = img_gray / 255

        # set to magnitude range of mri data
        mri_magn = np.random.choice(magn_ranges)

        img_gray = img_gray * mri_magn
        
        # image is too small
        if img_x < x or img_y < y:
            print(f"Image {os.path.basename(file_path)} too small")

        print(img_gray.shape)

        if img_x > x*2:
            img_gray = img_gray[img_x//2-x:img_x//2 +x, :]
        if img_y > y*2:
            img_gray = img_gray[:, img_y//2-y:img_y//2 +y]
        #make smaller cropping window

        #update
        img_x, img_y = img_gray.shape

        print(img_gray.shape)       


        img_3D = fully_rotate_2Dimage_in_3D_with_interpolation(img_gray)

        print(img_x, img_x//10, (9*img_x)//10-x)
        print(img_y, img_y//10, (9*img_y)//10-y)
        print(img_3D.shape[-1], img_3D.shape[-1]//10, (9*img_3D.shape[-1])//10-z)
        # crop image randomly
        x_rand = np.random.randint(img_x//10, (9*img_x)//10-x)
        y_rand = np.random.randint(img_y//10, (9*img_y)//10-y)
        z_rand = np.random.randint(img_3D.shape[-1]//10, img_3D.shape[-1]-z)

        # crop the image in random position
        img_gray_resized = img_3D[x_rand:x_rand+x, y_rand:y_rand+y, z_rand:z_rand+z]

        # # rotate the image
        # rot_img = np.zeros((img_x, img_y, z))

        # rot_img[:, :img_y//2, z//2] = img_gray[:, :img_y//2]
        # rot_img[np.where(rot_img < 1/100 * mri_magn)] = 0

        # rot_orig = rot_img.copy()
        # tol = mri_magn
        # tol_0 = 1/10 * mri_magn
        # print(mri_magn, mri_magn/100)
        # print('Create 3D image by rotating by 180 degrees..')
        # for angle in range(0, 180, 1):
            
        #     rotated = rotate_2Dimage_in_3D(rot_orig, angle, (1, 2))
        #     rotated = np.abs(rotated)
        #     # rotated[np.where(rotated < tol_0)] = 0
        #     rotated /= np.max(rotated)
        #     rotated *= mri_magn
        #     #remove interpolation artifacts
        #     # rotated[np.where(np.abs(rotated) <= mri_magn/100)] = 0
            
        #     # rot_img += rotated
        #     # # rot_img[np.where(np.abs(rot_img) >= tol)] = tol
        #     # rot_img /= np.max(rot_img)
        #     # rot_img *= mri_magn
        
        #     rot_img[np.where((np.abs(rotated)>tol_0))] = rotated[np.where((np.abs(rotated)>tol_0))]

        #make it symmetric
        # rot_img[:, :, z//2:] = rot_img[:, :, :z//2][:, :, ::-1]

        # plt.subplot(1, 2, 1)
        # plt.imshow(rot_img[x//2, :, :])
        # plt.subplot(1, 2, 2)
        # plt.imshow(rot_img[:, :, z//4])
        # plt.show()

        # use gaussian blurring to remove interpolation artifacts
        # rot_img = scipy.ndimage.gaussian_filter(rot_img, sigma=1)


        
        # plt.subplot(1, 2, 1)
        # plt.imshow(rot_img[x//2, :, :])
        # plt.subplot(1, 2, 2)
        # plt.imshow(rot_img[:, :, z//4])
        # plt.show()
        # rot_img[:, :, z//2] = img_gray_resized # to reset the middle 

        # # save
        # h5functions.save_to_h5(save_path, f'{model}_test40_gaussfltr_tol10th', rot_img, expand_dims=False)
        # rotated = rotate_2Dimage_in_3D(rot_orig, 30, (1, 2))
        # rotated = np.abs(rotated)

        # h5functions.save_to_h5(save_path, f'{model}_rotated30_and_29', rotated, expand_dims=False)

        
        #------stack along z and t axis for 3D construction--------

         # stack in 3D along z axis
        # img_gray_resized = np.repeat(img_gray_resized[:, :, None], z, axis=-1)

        # print(img_gray_resized.shape)
        # img_gray_resized = rot_img

        # use same image in time for temporal coherency
        img_gray_resized = np.repeat(img_gray_resized[None, :, :, :], t, axis=0)

        print(img_gray_resized.shape)

        assert((t, x, y, z) == img_gray_resized.shape) #shape of new magnitude should match datamodel shape

        #-------save--------
        print(f'------- Magnitude to corresponding Model {model} saved to {save_path} -------')
        h5functions.save_to_h5(save_path, model, img_gray_resized, expand_dims=False)