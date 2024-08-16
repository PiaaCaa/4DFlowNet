import random as rnd
import numpy as np
import csv
import h5py

def write_header(filename):
    fieldnames = ['source', 'target','index', 'start_x', 'start_y', 'start_z', 'rotate', 'rotation_plane', 'rotation_degree_idx', 'coverage']
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def write_header_temporal(filename):
    fieldnames = ['source', 'target','axis', 'index', 'start_t', 'start_1', 'start_2', 'step_t', 'reverse', 'coverage']
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def create_temporal_mask(mask, n_frames):
    '''
    from static mask create temporal mask of shape (n_frames, h, w, d)
    '''
    assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
    return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)
    

def generate_random_patches(input_filename, target_filename, output_filename, index, n_patch, binary_mask, patch_size, minimum_coverage, empty_patch_allowed, apply_all_rotation=True):
    """
    Generate random patches from the input and target images for SPATIAL super-resolution pairs.

    Args:
        input_filename (str): The filename of the input image.
        target_filename (str): The filename of the target image.
        output_filename (str): The filename of the output CSV file to save the patches.
        index (int): The index of the image to generate patches from.
        n_patch (int): The number of patches to generate.
        binary_mask (ndarray): The binary mask used to select valid regions for patch generation.
        patch_size (int): The size of the patches to generate.
        minimum_coverage (float): The minimum coverage required for a patch to be considered valid.
        empty_patch_allowed (int): The maximum number of empty patches allowed.
        apply_all_rotation (bool, optional): Whether to apply all possible rotations to the patches. Defaults to True.
    """

    empty_patch_counter = 0
            
    # foreach row, create n number of patches
    j = 0
    not_found = 0
    while j < n_patch:
        if not_found > 100:
            print(f"Cannot find enough patches above {minimum_coverage} coverage, please lower the minimum_coverage")
            break

        can_still_take_empty_patch = empty_patch_counter < empty_patch_allowed
        # print('patch number', j)
        patch = PatchData(input_filename, target_filename, patch_size)
        
        # default, no rotation
        patch.create_random_patch(binary_mask, index)
        patch.calculate_patch_coverage(binary_mask, minimum_coverage)

        # before saving, do a check
        # print(patch.coverage)
        if patch.coverage < minimum_coverage:
            if can_still_take_empty_patch:
                
                print('Taking this empty one',patch.coverage)
                empty_patch_counter += 1
                
            else:
                # skip empty patch because we already have one
                not_found += 1
                continue

        patch.write_to_csv(output_filename)

        # apply ALL rotation
        if apply_all_rotation:
            patch.rotate = 1
            for plane_nr in range(1,4):
                # rotation plane 1,2,3
                patch.rotation_plane = plane_nr

                for rotation_idx in range(1,4):
                    # rotation index 1,2,3
                    patch.rotation_degree_idx = rotation_idx
                    patch.write_to_csv(output_filename)
                # /end of rotation idx
            # /end of plane
        else:
            patch.rotate = 1
            # do 1 random rotation
            patch.rotation_plane = rnd.randint(1,3)
            patch.rotation_degree_idx = rnd.randint(1,3)
            patch.write_to_csv(output_filename)

        # break
        j += 1
    # /end of while n_patch


def generate_temporal_random_patches_all_axis(input_filename, target_filename, output_filename, axis, index, n_patch, binary_mask, patch_size, minimum_coverage, empty_patch_allowed, reverse=False, step_t =2):
    """
    Generate temporal random patches along a specified axis.

    Args:
        input_filename (str): The filename of the input data.
        target_filename (str): The filename of the target data.
        output_filename (str): The filename to save the generated patches.
        axis (int): The axis along which to generate the patches (0 for temporal, 1 for width, 2 for depth).
        index (int): The index along the specified axis to generate the patches.
        n_patch (int): The number of patches to generate.
        binary_mask (ndarray): The binary mask indicating the valid regions for patch generation.
        patch_size (tuple): The size of each patch (t, w, d).
        minimum_coverage (float): The minimum coverage required for a patch to be considered valid.
        empty_patch_allowed (int): The maximum number of empty patches allowed.
        reverse (bool, optional): Whether to reverse the created patches. Defaults to False.
    """
    
    empty_patch_counter = 0
            
    # foreach row, create n number of patches
    j = 0
    not_found = 0

    # #take away h path, i.e. only patches of size (t, w, d)
    
    if axis == 0:   binary_mask=binary_mask[:,index,  :,        :]
    elif axis == 1: binary_mask=binary_mask[:,:,     index,     :]
    elif axis == 2: binary_mask=binary_mask[:,:,      : ,   index]
    

    while j < n_patch:
        
        if not_found > 100:
            print(f"Cannot find enough patches above {minimum_coverage} coverage, please lower the minimum_coverage")
            break

        can_still_take_empty_patch = empty_patch_counter < empty_patch_allowed
        # print('patch number', j)
        patch = TemporalPatchData(input_filename, target_filename, patch_size)
        
        # default, no rotation
        patch.create_random_patch(binary_mask, index, axis, step_t=step_t)
        patch.calculate_patch_coverage(binary_mask, minimum_coverage)

        # before saving, do a check
        if patch.coverage < minimum_coverage:
            if can_still_take_empty_patch:
                
                print('Taking this empty one',patch.coverage)
                empty_patch_counter += 1
                
            else:
                # skip empty patch because we already have one
                not_found += 1
                continue

        patch.write_to_csv(output_filename)
        j+=1

        #if reverse = True, created patch should only be reversed when random choice is True
        if reverse:
            # rev = np.random.choice([True, False])
            # if rev: 
            patch.reverse = -1
            patch.write_to_csv(output_filename)

        
    # /end of while n_patch


class PatchData:
    def __init__(self, source_file, target_file, patch_size):
        self.patch_size = patch_size

        self.source_file = source_file
        self.target_file = target_file
        self.idx = None
        self.start_x = None
        self.start_y = None
        self.start_z = None
        self.rotate = 0
        self.rotation_plane = 0
        self.rotation_degree_idx = 0
        self.coverage = 0

    def create_random_patch(self, u, index):
        self.idx = index
        self.start_x = rnd.randrange(0, u.shape[0] - self.patch_size + 1) 
        self.start_y = rnd.randrange(0, u.shape[1] - self.patch_size + 1) 
        self.start_z = rnd.randrange(0, u.shape[2] - self.patch_size + 1) 

    def set_patch(self, index, x, y, z):
        self.idx = index
        self.start_x = x
        self.start_y = y
        self.start_z = z

    def calculate_patch_coverage(self, binary_mask, minimum_coverage=0.2):
        patch_region = np.index_exp[self.start_x:self.start_x+self.patch_size, self.start_y:self.start_y+self.patch_size, self.start_z:self.start_z+self.patch_size]
        patch = binary_mask[patch_region]

        self.coverage = np.count_nonzero(patch) / self.patch_size ** 3
        self.coverage = np.round(self.coverage * 1000) / 1000 # round the number to 3 decimal digit

    def create_random_rotation(self):
        is_rotate = rnd.randint(0,1) # 50-50 percent chance of augmentation
        if is_rotate == 0:
            plane_nr = 0
            degree_idx = 0
        else:
            plane_nr = rnd.randint(1,3)
            degree_idx = rnd.randint(1,3)
        
        self.rotate = is_rotate
        self.rotation_plane = plane_nr
        self.rotation_degree_idx = degree_idx

        return is_rotate, plane_nr, degree_idx

    def write_to_csv(self, output_filename):
        fieldnames = ['source', 'target', 'index', 'start_x', 'start_y', 'start_z', 'rotate', 'rotation_plane', 'rotation_degree_idx', 'coverage']
        with open(output_filename, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'source': self.source_file, 'target': self.target_file, 'index': self.idx, 
            'start_x': self.start_x, 'start_y': self.start_y, 'start_z': self.start_z,
            'rotate': self.rotate, 'rotation_plane': self.rotation_plane, 'rotation_degree_idx': self.rotation_degree_idx,
            'coverage': self.coverage})

class TemporalPatchData:
    def __init__(self, source_file, target_file, patch_size):
        self.patch_size = patch_size

        self.source_file = source_file
        self.target_file = target_file # this fila has same size as source file with added noise
        self.axis = None 
        self.idx = None
        self.start_t = None
        self.start_1 = None
        self.start_2 = None
        self.reverse = 1
        self.step_t = 2
        self.coverage = 0

    def create_random_patch(self, u, index, axis, step_t=2):
        self.step_t = step_t
        self.axis = axis
        self.idx = index
        self.start_t = rnd.randrange(2, u.shape[0] - self.patch_size*self.step_t + 1) #TODO delete this later- only to not consider first two frames in cs data
        if self.start_t <2:
            print('START IS BEFORE 2!!')
        self.start_1 = rnd.randrange(0, u.shape[1] - self.patch_size) 
        self.start_2 = rnd.randrange(0, u.shape[2] - self.patch_size) 

    def set_patch(self, index, x, y, z):
        self.idx = index
        self.start_t = x
        self.start_1 = y
        self.start_2 = z

    def calculate_patch_coverage(self, binary_mask, minimum_coverage=0.2):
        #important: take every second time step
        patch_region = np.index_exp[self.start_t:self.start_t+self.patch_size:self.step_t, self.start_1:self.start_1+self.patch_size, self.start_2:self.start_2+self.patch_size]
        patch = binary_mask[patch_region]

        self.coverage = np.count_nonzero(patch) / self.patch_size ** 3
        self.coverage = np.round(self.coverage * 1000) / 1000 # round the number to 3 decimal digit


    def write_to_csv(self, output_filename):
        
        fieldnames = ['source', 'target','axis', 'index', 'start_t', 'start_1', 'start_2', 'step_t', 'reverse', 'coverage']
        with open(output_filename, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'source': self.source_file, 'target': self.target_file,'axis': self.axis, 'index': self.idx, 
            'start_t': self.start_t, 'start_1': self.start_1, 'start_2': self.start_2, 'step_t': self.step_t,'reverse': self.reverse, 'coverage': self.coverage})