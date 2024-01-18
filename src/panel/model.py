import numpy as np
import h5py
from dataclasses import dataclass

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


        # next steps: include interpolation files
        # include step size
        # create tests for the data, e.g. dame dimensions, etc., also that mask is composed of 0 and 1
    
    def __post_init__(self):
        print('Performing post init ')
        self.load_mask(self.hr_file)
        self.hr_data   = self.load_vel_h5(self.hr_file)
        self.lr_data   = self.load_vel_h5(self.lr_file)
        self.pred_data = self.load_vel_h5(self.pred_file)

        self.min_v = {}
        self.max_v = {}
        for vel in self.vel_colnames:
            self.min_v[vel] = np.quantile(self.hr_data[vel][np.where(self.mask !=0)].flatten(), 0.01)
            self.max_v[vel] = np.quantile(self.hr_data[vel][np.where(self.mask !=0)].flatten(), 0.99)

        

    
    def load_vel_h5(self, filename, vel_colnames = []):
        if len(vel_colnames) == 0: vel_colnames = self.vel_colnames
        data = {}
        with h5py.File(filename, mode = 'r' ) as h5:
            for vel in vel_colnames:
                data[vel] = np.asarray(h5[vel])
        return data

    def load_mask(self,filename):
        with h5py.File(filename, mode = 'r' ) as h5:
            self.mask = np.asarray(h5['mask'])

        if len(self.mask.squeeze().shape) ==3:
            self.mask = create_dynamic_mask(self.mask, self.hr_data['u'].shape[0])
        return

    def load_interpolation_todo():
        return
    
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
    
