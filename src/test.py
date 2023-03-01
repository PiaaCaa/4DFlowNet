import numpy as np
import h5py
import random as rnd
from utils import prediction_utils
from utils import evaluate_utils

def create_temporal_mask(mask, n_frames):
    '''
    from static mask create temporal mask of shape (n_frames, h, w, d)
    '''
    assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
    temporal_mask = np.zeros((n_frames, mask.shape[0], mask.shape[1], mask.shape[2]))
    for i in range(n_frames):
        temporal_mask[i, :, :, :] = mask
    return temporal_mask

if __name__ == "__main__": 
    gt_file = 'Temporal4DFlowNet/data/CARDIAC/M3_2mm_step5_static.h5'
    lr_file = 'Temporal4DFlowNet/data/CARDIAC/M3_2mm_step2_static_TLR.h5'

    print("Testing first batch file")
    print("If you can read this, completed sbatch file")

    # patchify_file1 = 'Temporal4DFlowNet/results/Temporal4DFlowNet_20230210-0333/Testset_result_model3_2_0333_temporal_new.h5'
    # patchify_file2 = 'Temporal4DFlowNet/results/Temporal4DFlowNet_20230210-0333/Testset_result_model3_2_0333_temporal.h5' #temporal_new_6_eff_pad_size.h5
    # patchify_file3 = 'Temporal4DFlowNet/results/Temporal4DFlowNet_20230210-0333/Testset_result_model3_2_0333_temporal_new_6_eff_pad_size.h5'
    # save_differenc = 'Temporal4DFlowNet/results/Temporal4DFlowNet_20230210-0333//Testset_result_model3_2_0333_temporal_diff.h5'

    # vel_volnames = ["u", "v", "w"]
    # with h5py.File(patchify_file1, mode = 'r' ) as p1:
    #     with h5py.File(patchify_file2, mode = 'r' ) as p2:
    #         with h5py.File(patchify_file3, mode = 'r' ) as p3:
    #             with h5py.File(gt_file, mode = 'r' ) as gt:
    #                 mask = gt["mask"]
    #                 temporal_mask = create_temporal_mask(mask, p1["u"].shape[1] )

    #                 for vel in vel_volnames:
    #                     diff12 = np.abs(np.asarray(p1[vel]) -  np.asarray(p2[vel])).transpose((1, 0, 2, 3))


    #                     diff12 = np.multiply(diff12, temporal_mask)

    #                     # prediction_utils.save_to_h5(save_differenc, f"{vel}_diff_no_padd_new", diff23, compression='gzip')
    #                     exit()
    # with h5py.File(lr_file, mode = 'r' ) as p1:
    #     for vel in vel_volnames:
    #         print(p1[vel].shape)

    #     print("mask shape",p1['mask'].shape)  
    #test this branch
    #evaluate_utils.plot_relative_error([gt_file, gt_file, gt_file], [patchify_file1, patchify_file2, patchify_file3], ["effective padsize (-4)","no effective padsize" ,"effective padsize (-6)"], save_as='Temporal4DFlowNet/results/Temporal4DFlowNet_20230210-0333/Error_comparison_patchify_testset.png')
                        



