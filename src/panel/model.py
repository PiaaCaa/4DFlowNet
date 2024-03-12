import numpy as np
import h5py
from dataclasses import dataclass
from scipy.ndimage import binary_erosion

def create_dynamic_mask(mask, N_frames):
    """ This function creates a dynamic mask from a static mask"""
    dynamic_mask = np.repeat(mask, N_frames, axis = 0)
    return dynamic_mask


@dataclass
class Evaluate_NNModel():
    """ This class makes the calculations and returns important values to the controller"""

    hr_data = {}
    lr_data = {}
    pred_data = {}

    def __init__(self, model_name,hr_file, lr_file,pred_dir, pred_file, vel_colnames = ['u', 'v', 'w'], use__dyn_mask = True, offset = False,  **params):
        
        self.model_name = model_name
        self.vel_colnames = vel_colnames

        self.hr_file = hr_file
        self.lr_file = lr_file
        self.pred_file = pred_file
        self.eval_dir = f'{pred_dir}/plots'
        self.use__dyn_mask = use__dyn_mask
        self.load_mask(self.hr_file)
        self.b_mask, self.c_mask = self.create_boundary_mask(self.mask)
        self.hr_data   = self.load_vel_h5(self.hr_file)
        self.lr_data   = self.load_vel_h5(self.lr_file)
        self.pred_data = self.load_vel_h5(self.pred_file, vel_colnames=['u_combined', 'v_combined', 'w_combined'])

        self.N_frames = self.hr_data["u"].shape[0]
        self.min_v = {}
        self.max_v = {}
        for vel in self.vel_colnames:
            self.min_v[vel] = np.quantile(self.hr_data[vel][np.where(self.mask !=0)].flatten(), 0.01)
            self.max_v[vel] = np.quantile(self.hr_data[vel][np.where(self.mask !=0)].flatten(), 0.99)
        # next steps: include interpolation files
        # include step size
        # create tests for the data, e.g. dame dimensions, etc., also that mask is composed of 0 and 1
    
    def load_vel_h5(self, filename, vel_colnames = []):
        if len(vel_colnames) == 0: vel_colnames = self.vel_colnames
        data = {}
        with h5py.File(filename, mode = 'r' ) as h5:
            for vel,v in zip(vel_colnames, self.vel_colnames):
                data[v] = np.asarray(h5[vel])
        return data

    def load_mask(self,filename):
        with h5py.File(filename, mode = 'r' ) as h5:
            self.mask = np.asarray(h5['mask'])

        if len(self.mask.squeeze().shape) ==3:
            self.mask = create_dynamic_mask(self.mask, self.hr_data['u'].shape[0])
        return

    def load_interpolation_todo():
        return
    
    def create_minimal_mask(self):
        """With a moving domain we create a mask which is the intersection of the mask over time"""
        intersec_mask = np.sum(self.mask, axis = 0)
        self.mask_min = np.zeros_like(intersec_mask)
        self.mask_min[np.where(intersec_mask !=0)] = 1
        return
    
    def create_boundary_mask(self, binary_mask):
        '''
        returns boundary and core mask given a binary mask. 
        Note that mask values should be 0 and 1
        '''

        if (len(binary_mask.shape)==3):
            print("Create boundary mask for 3D data")
            core_mask = binary_erosion(binary_mask)
            boundary_mask = binary_mask - core_mask
            return boundary_mask, core_mask
        
        core_mask       = np.zeros_like(binary_mask)
        boundary_mask   = np.zeros_like(binary_mask)

        for t in range(binary_mask.shape[0]):
            core_mask[t, :, :, :] = binary_erosion(binary_mask[t, :, :, :])
            boundary_mask[t, :, :, :] = binary_mask[t, :, :, :] - core_mask[t, :, :, :]

            
        assert(np.linalg.norm(binary_mask - (boundary_mask + core_mask))== 0 ) # check that there is no overlap between core and boundary mask
        return boundary_mask, core_mask
    
    def update_calculation(self, t_frame, slicing_x,  axis):
        """ This function updates the calculation"""
        self.t_frame = t_frame
        self.slicing_x = slicing_x
        self.axis = axis

        idx_slice = self.get_slice_idx(t_frame, slicing_x,  axis)
        self.lr_slice = self.lr_data['u'][idx_slice]
        self.hr_slice = self.hr_data['u'][idx_slice]
        self.pred_slice = self.pred_data['u'][idx_slice]

        return

    def simple_view_lr(self, idx):
        """ This function returns a simple view of the data"""
        return self.lr_data[idx]
    

    def calculate_speed(self, data):
        """ This function returns the mean speed"""
        return np.sqrt(data['u']**2 + data['v']**2 + data['w']**2)
    
    def mean_velocity(self, vel):
        """ This function returns the mean velocity"""
        return np.average(vel, weights=self.mask, axis = 0)
    
    def get_slice_idx(self, t_frame, slicing_x,  axis):
        """ This function returns a slice of the data"""
        if axis == 'x':
            return np.index_exp[t_frame, slicing_x, :, :]
        elif axis == 'y':
            return np.index_exp[t_frame, :, slicing_x, :]
        elif axis == 'z':
            return np.index_exp[t_frame, :, :, slicing_x]
        else:
            print('ERROR: axis not defined')
            return None
    
    def get_slice(self, data, t_frame, slicing_x,  axis):
        """ This function returns a slice of the data"""
        return data[self.get_slice_idx(t_frame, slicing_x,  axis)]
    

    def create_random_indices(self, mask, p= 0.1):
        '''
        This function generates random indices in a 3D mask based on a given threshold.
        It assumes that the mask is a 3D array.
        The function randomly selects 'n' samples from the mask that have values greater than the threshold.
        It returns the x, y, and z indices of the selected samples.
        '''

        assert(len(mask.shape)==3) # Ensure that the mask is 3D

        n = int(p*np.count_nonzero(mask))
        mask_threshold = 0.9
        sample_pot = np.where(mask > mask_threshold)  # Find indices where mask values are greater than the threshold
        rng = np.random.default_rng()

        # Sample 'n' random samples without replacement
        sample_idx = rng.choice(len(sample_pot[0]), replace=False, size=n)

        # Get the x, y, and z indices of the selected samples
        x_idx = sample_pot[0][sample_idx]
        y_idx = sample_pot[1][sample_idx]
        z_idx = sample_pot[2][sample_idx]
        

        return x_idx, y_idx, z_idx

    def save_random_indices_core(self, idx_x, idx_y, idx_z):
        self.rdn_idxx_c = idx_x
        self.rdn_idxy_c = idx_y
        self.rdn_idxz_c = idx_z

    def save_random_indices_boundary(self, idx_x, idx_y, idx_z):
        self.rdn_idxx_b = idx_x
        self.rdn_idxy_b = idx_y
        self.rdn_idxz_b = idx_z


    def calculate_relative_error_normalized(self, ):
        '''
        Calculate relative error with tanh as normalization
        '''

        # if epsilon is set to 0, we will get nan and inf
        epsilon = 1e-5 

        u_diff = np.square(self.pred_data['u'] - self.hr_data['u'])
        v_diff = np.square(self.pred_data['v'] - self.hr_data['v'])
        w_diff = np.square(self.pred_data['w'] - self.hr_data['w'])

        diff_speed = np.sqrt(u_diff + v_diff + w_diff)
        actual_speed = np.sqrt(np.square(self.hr_data['u']) + np.square(self.hr_data['v']) + np.square(self.hr_data['w'])) 
        # actual speed can be 0, resulting in inf
        relative_speed_loss = np.tanh(diff_speed / (actual_speed + epsilon))
        # Make sure the range is between 0 and 1

        # Apply correction, only use the diff speed if actual speed is zero
        condition = np.not_equal(actual_speed, np.array([0.]))
        corrected_speed_loss = np.where(condition, relative_speed_loss, diff_speed)

        multiplier = 1e4 # round it so we don't get any infinitesimal number
        corrected_speed_loss = np.round(corrected_speed_loss * multiplier) / multiplier

        # Apply mask
        # binary_mask_condition = (mask > threshold)
        binary_mask_condition = np.equal(self.mask, 1.0)          
        corrected_speed_loss = np.where(binary_mask_condition, corrected_speed_loss, np.zeros_like(corrected_speed_loss))

        # Calculate the mean from the total non zero accuracy, divided by the masked area
        # reduce first to the 'batch' axis
        mean_err = np.sum(corrected_speed_loss, axis=(1,2,3)) / (np.sum(self.mask, axis=(1, 2, 3)) + 1) 

        # now take the actual mean
        # mean_err = tf.reduce_mean(mean_err) * 100 # in percentage
        mean_err = mean_err * 100

        self.RE = mean_err
        return mean_err
    
