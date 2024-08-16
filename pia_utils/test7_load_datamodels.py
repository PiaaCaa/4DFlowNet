import numpy as np 
import h5py

if __name__ == '__main__':

    data_models = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']

    data_dir = 'data/CARDIAC'

    for model in data_models: 
        
        print(f'--------TESTING model {model}--------')
        model_100frames = f'{data_dir}/{model}_2mm_step1_static_dynamic.h5'
        model_50frames  = f'{data_dir}/{model}_2mm_step2_static_dynamic.h5'

        with h5py.File(model_100frames, 'r') as f:
            u_100frames = np.array(f['u'])
            v_100frames = np.array(f['v'])
            w_100frames = np.array(f['w'])
        
        with h5py.File(model_50frames, 'r') as f:
            u_50frames = np.array(f['u'])
            v_50frames = np.array(f['v'])
            w_50frames = np.array(f['w'])

        print('Difference:', np.linalg.norm(u_100frames[::2, :, :, :]- u_50frames))
        print('Difference:', np.linalg.norm(v_100frames[::2, :, :, :]- v_50frames))
        print('Difference:', np.linalg.norm(w_100frames[::2, :, :, :]- w_50frames))


