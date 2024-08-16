
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from prepare_data import fft_downsampling as fft_fcts
from prepare_data import h5functions
import prepare_data.cfl as cfl
from prepare_data import kspace_sampling as ks


def transform_cfl_format_test(data):
    """Assumption that data is of shape (t, x, y, z, c)"""
    assert len(data.shape) == 5, 'Data should be of shape (t, x, y, z, c)'
    print('Convert from shape', data.shape, 'to shape', data.transpose(1, 2, 3, 4, 0)[:, :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis,np.newaxis, :].shape)
    return data.transpose(1, 2, 3, 4, 0)[:, :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis,np.newaxis, :]

def vel_to_phase(vel, venc):
    return vel / venc * np.pi + np.pi 



if __name__ == '__main__':

    # Select data
    data_dir = '/mnt/c/Users/piacal/Code/SuperResolution4DFlowMRI/Temporal4DFlowNet/data'
    path_datamodel = f'{data_dir}/CARDIAC/M1_2mm_step2_static_dynamic.h5'
    save_as = 'results/kspacesampling'

    # load data
    with h5py.File(path_datamodel, mode = 'r') as h5:
        vel_u = np.array(h5['u'])[::2, :, :, :]
        venc_u = np.max(np.array(h5['u_max']))
        mask = np.array(h5['mask']).squeeze()[::2, :, :, :]

    magn  = mask.copy()


    # save input data for cs
    if False:
        sample_mask = cfl.readcfl(f'{save_as}/sample_mask_int')[:, :, :, 0, :, :, :, :, :, :, :]
        print('sample mask shape', sample_mask.shape, sample_mask.dtype)
        sample_mask = ks.reshape_from_cfl(sample_mask)
        # sample_mask = adjust_image_size(sample_mask, vel_u.shape)
        print('sample mask shape', sample_mask.shape)
        plt.subpplot(1, 2, 1)
        plt.imshow(vel_u[0, 30, :, :])

        plt.subpplot(1, 2, 1)
        plt.imshow(magn[0, 30, :, :])
        plt.show()

        


        # K-space data and save
        vel_u      = ks.adjust_image_size_centered(vel_u, sample_mask.shape)
        magn_large = ks.adjust_image_size_centered(magn, sample_mask.shape)
        complex_img = np.multiply(magn_large, np.exp(1j * ks.vel_to_phase_norm(vel_u)))
        u_kspace = ks.complex_image_to_centered_kspace(complex_img)
        u_kspace = np.multiply(u_kspace, sample_mask)
        #normalizer better? 
        # to vel space again to crop
        plt.imshow(np.abs(u_kspace[0, 30, :, :]))
        plt.show()
        comp_u = ks.centered_kspace_to_complex_img(u_kspace)
        comp_u = ks.adjust_image_size_centered(comp_u, mask.shape)

        print('vel_u shape', vel_u.shape)
        plt.imshow(np.angle(comp_u[0, 30, :, :]))
        plt.show()
        u_kspace = ks.complex_image_to_centered_kspace(comp_u)

        u_kspace = np.repeat(u_kspace[:, :, :, :, np.newaxis], 9, axis=-1)
        cfl.writecfl(f'{save_as}/u_kspace_small_sparse_9coils_newsnorm', transform_cfl_format_test(u_kspace))

        # Create sensitivy data with only ones
        # sens = np.ones(u_kspace[0, :, :, :].shape)
        # # sens /= sens.sum()
        # cfl.writecfl(f'{save_as}/coil_sensitivity_small_onesfloat_nonnorm_9coils', transform_cfl_format_test(sens[np.newaxis, :, :, :,:]))

        # print(transform_cfl_format_test(sens[np.newaxis, :, :, :,:]).shape)
        # bart command
        # bart pics -i30 --wavelet haar -d5 results/kspacesampling/u_kspace_small_nonsparse_new results/kspacesampling/coil_sensitivity_small_ones_nonnorm results/kspacesampling/output_test_small3
        # bart pics --wavelet haar -d5 -R W:7:0:0.0015 -R W:1024:0:0.0075 results/kspacesampling/u_kspace_small_sparse results/kspacesampling/coil_sensitivity_small_ones_nonnorm results/kspacesampling/output_test_small_nonsparse
        # bart pics -d5 -D --wavelet haar -R W:7:0:0.0015 -R W:1024:0:0.0075 results/kspacesampling/FromAlex/alex_ksp results/kspacesampling/FromAlex/alex_sens_static results/kspacesampling/FromAlex/result_cs
        # bart pics -d5 -D --wavelet haar -R W:7:0:0.0015 -R W:1024:0:0.0075 results/kspacesampling/u_kspace_small_sparse_8coils results/kspacesampling/coil_sensitivity_small_ones_nonnorm_8coils results/kspacesampling/output_test_small_nonsparse_coils8
        # bart pics -d5 -D --wavelet haar -R W:7:0:0.0015 -R W:1024:0:0.0075 results/kspacesampling/u_kspace_small_sparse_9coils_newsampl results/kspacesampling/coil_sensitivity_small_ones_nonnorm_9coils results/kspacesampling/output_test_small_nonsparse_coils9
        # bart pics -d5 -D --wavelet haar -R W:7:0:0.0015 -R W:1024:0:0.0075 results/kspacesampling/u_kspace_small_sparse_9coils_newsampl results/kspacesampling/coil_sensitivity_small_onesfloat_nonnorm_9coils results/kspacesampling/output_test_small_nonsparse_coils9_new2
        # bart pics -d5 -D --wavelet haar -R W:7:0:0.0015 -R W:1024:0:0.0075 results/kspacesampling/u_kspace_small_sparse_9coils_newsnorm results/kspacesampling/coil_sensitivity_small_onesfloat_nonnorm_9coils results/kspacesampling/output_test_small_nonsparse_coils9_new3

    if False:
        coils = 9
        kspace = cfl.readcfl(f'{save_as}/u_kspaceM1_hr').squeeze().transpose(4, 0, 1, 2, 3)
        kspace_small = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3], kspace.shape[-1]))
        for i in range(kspace.shape[-1]):
            print(kspace.shape)
            #adjust image size
            compl_img  = fft_fcts.centered_kspace_to_complex_img(kspace[:, :, :, :, i])
            print(compl_img.shape)
            compl_img  = ks.adjust_image_size_centered(compl_img, mask.shape)
            kspace_small[:, :, :, :, i] = fft_fcts.complex_image_to_centered_kspace(compl_img)

        # sens = adjust_image_size(sens, mask.shape)
        # print('kspace shape:', kspace.shape, 'sum', kspace.sum())

        # save as cfl
        cfl.writecfl(f'{save_as}/u_kspace16_small', transform_cfl_format_test(kspace_small))
        # cfl.writecfl(f'{save_as}/coil_sensitivity_small_ones_nonnorm_8coils', transform_cfl_format_test(sens))

    # check on saved input data
    if False:
        kspace = cfl.readcfl(f'{save_as}/u_kspace_small_sparse_9coils')
        print('kspace shape:', kspace.shape, 'sum', kspace.sum())

        sens = cfl.readcfl(f'{save_as}/coil_sensitivity_small_ones_nonnorm_9coils')#squeeze()
        print('shape sens: ', sens.shape)
        

    if True: 
        kspace = cfl.readcfl(f'{save_as}/u_kspaceM1_hr').squeeze().transpose(4, 0, 1, 2, 3)
        print('kspace shape:', kspace.shape)
        kspace = kspace[:, :, :, :, 0] #only use one sensitivity map
        u_recon, _  = fft_fcts.centered_kspace_to_velocity_img(kspace, magn, venc_u, normalized_0_2pi=False)
        print(np.sum(kspace))

        sens = cfl.readcfl(f'{save_as}/coil_sensitivity_small_ones_nonnorm_8coils')#squeeze()
        print('shape sens: ', sens.shape)
        
        plt.subplot(1, 3, 1)
        plt.imshow(u_recon[10, 30, :, :])

        plt.subplot(1, 3, 2)
        plt.imshow(np.abs(kspace[0, 30, :, :]))

        plt.subplot(1, 3, 3)
        # plt.imshow(np.abs(sens[30, :, :]))
        plt.show() 

    # check on results
    if False: 

        file = 'results/kspacesampling/output_test16_scoils'
        res = cfl.readcfl(file).squeeze()
        res = ks.reshape_from_cfl(res)
        # res = np.multiply(res, mask)
        print('sums', res.sum())

        if True:
            # vel = phase_norm_to_vel(np.angle(res-np.pi), np.min(vel_u), np.max(vel_u))
            #save this as h5
            h5functions.save_to_h5('results/kspacesampling/res_cs16_9coils_large_scoils.h5', 'res', res, expand_dims=False)
            print('Saved')
        print(res.dtype, res.shape, res.sum())
        plt.subplot(1, 3, 1)
        plt.imshow(np.abs(res[0, 100, :, :]))
        plt.title('absolute value')

        plt.subplot(1, 3, 2)
        plt.imshow(np.angle(res[0, 100, :, :]))
        # plt.imshow(np.multiply(np.angle(res)/ (np.pi) * venc_u, mask)[0, 20, :, :])
        plt.title('angle()/pi * venc')

        plt.subplot(1, 3, 3)
        plt.imshow(np.imag(res[0, 100, :, :]))
        plt.title('imaginary part')
        plt.show()
        
        if False: 
            h5functions.save_to_h5('results/kspacesampling/k_space_samlp_coilsens_test15.h5', 'data abs', np.abs(res), expand_dims=False)
            h5functions.save_to_h5('results/kspacesampling/k_space_samlp_coilsens_test15.h5', 'data angle', np.angle(res)/(2*np.pi) * venc_u, expand_dims=False)
            # h5functions.save_to_h5('results/kspacesampling/k_space_samlp_coilsens_test12_swap.h5', 'data reconstr', res_recon, expand_dims=False)
