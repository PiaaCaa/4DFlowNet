import numpy as np 
import h5py
import matplotlib.pyplot as plt
from prepare_data.h5functions import save_to_h5

def plot_masks(noise_mask, velocity_mask, static_mask, idx):
    plt.subplot(1, 3, 1)
    plt.imshow(noise_mask[idx, :, :])
    plt.axis('off')
    plt.title('Noise mask')

    plt.subplot(1,3, 2)
    plt.imshow(velocity_mask[idx, :, :])
    plt.axis('off')
    plt.title('Velocity mask')

    plt.subplot(1,3, 3)
    plt.imshow(static_mask[idx, :, :])
    plt.axis('off')
    plt.title('Static mask')

    plt.show()


def eddy_current_compensation(vel, magn,  velocity_threshold,magn_threshold,  t_diastole, order = 1, plot_mask = False):
    """
    Eddy current compensation. Assumes that eddy currents are static over time och slowly varying over space.
    Method is described in: Influence of Eddy Current, Maxwell and Gradient Field Corrections on 3D Flow Visualization of 3D CINE PC-MRI Data
    R. Lorenz,1 J. Bock,1 J. Snyder,1,2 J.G. Korvink,3,4 B.A. Jung,1 and M. Mark
    """

    # Step 1. Find static mask in t_diastole frame
    noise_mask = magn[t_diastole] <= magn_threshold*(magn.max())
    velocity_mask = np.std(vel, axis=0) < velocity_threshold
    static_mask = (velocity_mask.astype(int) - noise_mask.astype(int))
    static_mask[static_mask == -1] = 0
    static_mask = static_mask.astype(bool)

    if plot_mask:
        plot_masks(noise_mask, velocity_mask, static_mask, 30)
    
    # Step 2. Fitting a plane (1st order / 2nd order) with least squares method to the static regions of the last time frame (late diastole) for time resolved data (in vivo).
    # The plane was fitted to the last diastolic time frame to ensure minimal blood flow. 

    x_static, y_static, z_static = np.where(static_mask)
    x_all, y_all, z_all = np.meshgrid(np.arange(vel.shape[1]), np.arange(vel.shape[2]), np.arange(vel.shape[3]), indexing='ij')

    if order == 1: 
        A = np.vstack((np.ones_like(x_static), x_static, y_static, z_static)).T
        coefficients, residuals, rank, s = np.linalg.lstsq(A, vel[t_diastole, x_static, y_static, z_static], rcond=None)
        a0, a1, a2, a3 = coefficients
        fitted_plane = a0 + a1*x_all + a2*y_all + a3*z_all
    
    elif order == 2:
        A = np.vstack((np.ones_like(x_static), x_static, y_static, z_static, x_static**2, y_static**2, z_static**2, x_static*y_static, x_static*z_static, y_static*z_static)).T
        coefficients, residuals, rank, s = np.linalg.lstsq(A, vel[t_diastole, x_static, y_static, z_static], rcond=None)
        a0x, a1x, a2x, a3x, a4x, a5x, a6x, a7x, a8x, a9x = coefficients
        fitted_plane = a0x + a1x*x_all + a2x*y_all + a3x*z_all + a4x*x_all**2 + a5x*y_all**2 + a6x*z_all**2 + a7x*x_all*y_all + a8x*x_all*z_all + a9x*y_all*z_all


    plane_3d = np.zeros_like(vel)
    plane_3d[0, x_all, y_all, z_all] = fitted_plane
    plane_3d[1::, :, :,  :] = plane_3d[0, :, :, :]

    # Step 3. Applying the correction by subtraction of the fitted surface from the PC-MRI data for all time frames (in vivo)

    corrected_velocity = vel - plane_3d

    return corrected_velocity


if __name__ == '__main__':
    print('Running test for eddy current compensation')
    
    # load velocity data: 
    show_plot = False
    data_dir = 'data/PIA/THORAX/P05/h5'
    path_input = f'{data_dir}/P05.h5' 
    path_output = f'{data_dir}/P05_edd_corr.h5'

    with h5py.File(path_input, 'r') as f:
        print(f.keys())
        u = np.array(f['u'])
        v = np.array(f['v'])
        w = np.array(f['w'])
        mag = np.array(f['mag_u'])


    u_corrected = eddy_current_compensation(u, mag, 0.03,0.02,  -1, order = 2)
    v_corrected = eddy_current_compensation(v, mag, 0.03,0.02,  -1, order = 2, plot_mask = True)
    w_corrected = eddy_current_compensation(w, mag, 0.03,0.02,  -1, order = 2, plot_mask = True)

    # save corrected data save_to_h5(output_filepath, col_name, dataset, expand_dims = True):
    save_to_h5(path_output, 'u', u_corrected, expand_dims = False)
    save_to_h5(path_output, 'v', v_corrected, expand_dims = False)
    save_to_h5(path_output, 'w', w_corrected, expand_dims = False)
    with h5py.File(path_input, 'r') as f:
        for key in f.keys():
            if key not in ['u', 'v', 'w']:
                save_to_h5(path_output, key, np.array(f[key]), expand_dims = False)



    if show_plot:
        idx_line = np.index_exp[5, 30, 50, 40:120]
        idx_plane = np.index_exp[5, 30, :, :]
        
        # plot only line profile
        plt.plot(u[idx_line], label='Original u')
        plt.plot(u_corrected[idx_line], label='Corrected u')
        plt.plot(u[idx_line] - u_corrected[idx_line], label='Eddy current correction (u - u_corr)')
        plt.title('Line profile of u')
        plt.legend()
        plt.show()

        plt.subplot(1, 3, 1)
        plt.imshow(u[idx_plane])
        plt.axis('off')
        plt.title('Original u')

        plt.subplot(1, 3, 2)
        plt.imshow(u_corrected[idx_plane])
        plt.axis('off')
        plt.title('Corrected u')

        plt.subplot(1, 3, 3)
        plt.imshow(u[idx_plane] - u_corrected[idx_plane])
        plt.axis('off')
        plt.title('Eddy current correction (u - u_corr)')

        plt.show()

        # plot surface of correction:
        edd_corr = u[idx_plane] - u_corrected[idx_plane]
        fig = plt.figure()
        x = np.arange(0, edd_corr.shape[0])
        y = np.arange(0, edd_corr.shape[1])
        x, y = np.meshgrid(x, y)
        z = edd_corr.T
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        plt.show()




    

    


