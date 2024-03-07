import numpy as np
import os
from PIL import Image
import h5py
import h5functions 
import scipy.ndimage

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

if __name__ == "__main__":

    # path to the directory with the images
    directory = 'data/flowerdataset'

    save_path = 'data/flower_magn_data_4D_rotate.h5'

    # get all file paths
    file_paths = get_paths(directory)

    in_silico_model_size = {'M1': (50, 72, 70, 76), 'M2': (50, 84, 60, 96), 'M3': (50, 72, 82, 84), 'M4': (50, 62, 58, 84) }
    models = list(in_silico_model_size.keys())
    new_mag_data = {}

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

        # rotate the image
        rot_img = np.zeros((x, y, z))

        rot_img[:, :, z//2] = img_gray_resized

        rot_orig = rot_img.copy()
        tol = 20
        for angle in range(0, 360, 1):
            rotated = rotate_2Dimage_in_3D(rot_orig, angle, (1, 2))
    
            rot_img[np.where(np.abs(rot_img) <= tol)] = rotated[np.where(np.abs(rot_img) <= tol)]

        h5functions.save_to_h5(save_path, f'{model}_testrot_new9', rot_img, expand_dims=False)

        exit()

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