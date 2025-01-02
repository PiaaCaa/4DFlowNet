import numpy as np
import scipy

"""
This file contains functions which help to fuse the magnitude images from invivo to insilico data
"""

def idx_invivo_to_insilico(mask_v_transformed, mask_LV_s):
    """
    Get index for invivo and insilico data for transformation, 3D and 4D data
    """


    x_coord_v, y_coord_v, z_coord_v = np.where(mask_v_transformed>0)
    x_coord_LV_s, y_coord_LV_s, z_coord_LV_s = np.where(mask_LV_s>0)

    # find center of gravity in invivo and insilico w.r.t left ventricle
    x_LV_v,y_LV_v, z_LV_v  = int(np.sum(x_coord_v)/np.sum(mask_v_transformed)), int(np.sum(y_coord_v)/np.sum(mask_v_transformed)), int(np.sum(z_coord_v)/np.sum(mask_v_transformed))
    x_LV_s, y_LV_s, z_LV_s = int(np.sum(x_coord_LV_s)/np.sum(mask_LV_s)),  int(np.sum(y_coord_LV_s)/np.sum(mask_LV_s)), int(np.sum(z_coord_LV_s)/np.sum(mask_LV_s))

    print('center of insilico LV ', x_LV_s, y_LV_s, z_LV_s)
    print('center of invivo LV', x_LV_v, y_LV_v, z_LV_v)

    X_v, Y_v, Z_v = mask_v_transformed.shape
    X_s, Y_s, Z_s = mask_LV_s.shape

    # indices for 3 dim data (spatial)
    idx_insilico = np.index_exp[int(np.maximum(0, x_LV_s - x_LV_v )):int(np.minimum(x_LV_s +X_v- x_LV_v-1, X_s -1)), 
                                int(np.maximum(0, y_LV_s - y_LV_v )):int(np.minimum(y_LV_s +Y_v- y_LV_v-1, Y_s -1)), 
                                int(np.maximum(0, z_LV_s - z_LV_v )):int(np.minimum(z_LV_s +Z_v- z_LV_v-1, Z_s -1))]
    idx_invivo =   np.index_exp[int(np.maximum(0, -(x_LV_s-x_LV_v))):int(np.minimum(X_v-1, x_LV_v+ X_s- x_LV_s-1)), 
                                int(np.maximum(0, -(y_LV_s-y_LV_v))):int(np.minimum(Y_v-1, y_LV_v+ Y_s- y_LV_s-1)), 
                                int(np.maximum(0, -(z_LV_s-z_LV_v))):int(np.minimum(Z_v-1, z_LV_v+ Z_s- z_LV_s-1))]


    print('insilico indices: x min/max', int(np.maximum(0, x_LV_s - x_LV_v )),int(np.minimum(x_LV_s +X_v- x_LV_v-1, X_s -1)), 
                            'y min/max', int(np.maximum(0, y_LV_s - y_LV_v )),int(np.minimum(y_LV_s +Y_v- y_LV_v-1, Y_s -1)), 
                            'z min/max', int(np.maximum(0, z_LV_s - z_LV_v )),int(np.minimum(z_LV_s +Z_v- z_LV_v-1, Z_s -1)))

    # indices for 4D data
    idx_t_insilico = np.index_exp[:, int(np.maximum(0, x_LV_s - x_LV_v )):int(np.minimum(x_LV_s +X_v- x_LV_v-1, X_s -1)), 
                                int(np.maximum(0, y_LV_s - y_LV_v )):int(np.minimum(y_LV_s +Y_v- y_LV_v-1, Y_s -1)), 
                                int(np.maximum(0, z_LV_s - z_LV_v )):int(np.minimum(z_LV_s +Z_v- z_LV_v-1, Z_s -1))]
    idx_t_invivo =   np.index_exp[:, int(np.maximum(0, -(x_LV_s-x_LV_v))):int(np.minimum(X_v-1, x_LV_v+ X_s- x_LV_s-1)), 
                                int(np.maximum(0, -(y_LV_s-y_LV_v))):int(np.minimum(Y_v-1, y_LV_v+ Y_s- y_LV_s-1)), 
                                int(np.maximum(0, -(z_LV_s-z_LV_v))):int(np.minimum(Z_v-1, z_LV_v+ Z_s- z_LV_s-1))]
    
    return idx_insilico, idx_t_insilico, idx_invivo, idx_t_invivo



def combine_mag_images(mask_v, mask_s, mag_u_invivo ,idx_insilico, idx_invivo, indivial_transformation_scale_rotate):
    """
    Combine magnitude images from invivo and insilico data
    """
    
    # Step 1: fill 
    surrsounding_tissue_size = 20
    #get average by getting tissue values in bounding box
    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(mask_v)
    idx_bounding_box = np.index_exp[rmin-surrsounding_tissue_size:rmax+surrsounding_tissue_size, 
                                    cmin-surrsounding_tissue_size:cmax+surrsounding_tissue_size, 
                                    zmin-surrsounding_tissue_size:zmax+surrsounding_tissue_size]
    bounding_box_mag_u = mag_u_invivo[idx_bounding_box]
    bounding_box_mask = mask_v[idx_bounding_box]
    avg_surrounding_tissue = np.average(bounding_box_mag_u[np.where(bounding_box_mask==0)])

    # Step 2: Get original intensity values 
    avg_fluid_region_invivo = np.average(mag_u_invivo[np.where(mask_v == 1)])
    std_fluid_region_invivo = np.std(mag_u_invivo[np.where(mask_v == 1 )])

    #Step 2.1 replace masked invivo region with average of surrounding tissue
    mag_u_invivo[np.where(mask_v > 0)] = avg_surrounding_tissue

    # Step 3
    # transformation of magnitude
    temp = indivial_transformation_scale_rotate(mag_u_invivo, scale = [2,2,2], interpolation='linear')
    mag_u_transformed = np.zeros_like(mask_s)
    mag_u_transformed[idx_insilico] = temp[idx_invivo]
    
    # transformation of mask
    mask_v_transformed = np.zeros_like(mask_s) 
    mask_v_transformed[idx_insilico] =  indivial_transformation_scale_rotate(mask_v, scale = [2,2,2], interpolation='NN')[idx_invivo]
    combined_mask = mask_v_transformed.copy()
    combined_mask += mask_s*2
    
    # Step 5
    # fill vales from insilico mask with normal ditributed values with same standard deviation as original. Then smoothen the result with gaussian filter
    smoothen = np.ones(mask_s.shape)*avg_fluid_region_invivo
    smoothen[np.where(mask_s==1)] = np.random.normal(avg_fluid_region_invivo, np.sqrt(std_fluid_region_invivo), size=mag_u_transformed[np.where(mask_s)].shape).reshape(mag_u_transformed[np.where(mask_s==1)].shape)
    smoothen = scipy.ndimage.gaussian_filter(smoothen, sigma = 0.7)

    mag_u_transformed[np.where(mask_s==1)] = smoothen[np.where(mask_s==1)]

    # return values cropped on the in silico data
    return mag_u_transformed[idx_insilico], combined_mask[idx_insilico] 



def bbox2_3D(mask):
    '''
    Get bounding box of 3D mask
    code taken from https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    '''
    # check if there is a segmentation in that image
    if np.sum(mask) == 0:
        print('No segmentation in given mask')
        x, y, z = mask.shape
        return 0, x, 0, y, 0, z

    r = np.any(mask, axis=(1, 2))
    c = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax