import numpy as np
import os
import csv
from Network.PatchHandler3D_temporal import PatchHandler4D, PatchHandler4D_all_axis
from Network.TrainerController_temporal import TrainerController_temporal

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

def write_settings_into_csv_file(filename,name, training_file, validation_file, test_file, epochs,batch_size,patch_size, low_resblock, high_resblock, upsampling_type, low_block_type, high_block_type, post_block_type, sampling, notes):
    """
        Write settings into csv file to store training runs
    """
    print(f"Write settings into overview file {filename}")
    fieldnames = ["Name","training_file","validation_file","test_file","epochs","batch_size","patch_size","res_increase","low_resblock","high_resblock","upsampling_type", "low_block_type", "high_block_type", "post_block_type", "sampling",  "notes"]
    with open(filename, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'Name':name, "training_file":training_file, "validation_file":validation_file, "test_file":test_file, "epochs":epochs, "batch_size":batch_size, "patch_size":patch_size, "res_increase":res_increase, "low_resblock":low_resblock, "high_resblock":high_resblock, 
                             "upsampling_type": upsampling_type, 'low_block_type': low_block_type, 'high_block_type':high_block_type, 'post_block_type':post_block_type, 'sampling':sampling, "notes":notes })

if __name__ == "__main__":
    data_dir = 'data/CARDIAC'
    
    # ---- Patch index files ----
    training_file = '{}/Temporal16MODEL23_2mm_step2_invivoP04P03_magn_tempsmooth_toeger.csv'.format(data_dir) 
    validate_file = '{}/Temporal16MODEL1_2mm_step2_invivoP01_magn_tempsmooth_toeger.csv'.format(data_dir) #Temporal16MODEL23_2mm_step2_invivoP04P03_magn_tempsmooth_toeger.csv

    QUICKSAVE = True
    benchmark_file = '{}/Temporal16MODEL4_2mm_step2_invivoP02_magn_tempsmooth_toeger.csv'.format(data_dir)
    
    overview_csv = 'c:/Users/piacal/Code/SuperResolution4DFlowMRI/Temporal4DFlowNet/results/Overview_models.csv'

    restore = False
    if restore:
        model_dir = "Temporal4DFlowNet/models/Temporal4DFlowNet_20240118-1300"
        model_file = "Temporal4DFlowNet-best.h5"

    # Adapt how patches are saved for temporal domain if True a different loading scheme is used
    load_patches_all_axis = True

    print('Check, that all the files exist:', os.path.isfile(training_file), os.path.isfile(validate_file), os.path.isfile(benchmark_file), os.path.isfile(overview_csv))

    # Hyperparameters optimisation variables
    initial_learning_rate = 2e-4
    epochs =  1
    batch_size = 32
    mask_threshold = 0.6
    lr_decay_epochs = 0

    # Network setting
    network_name = 'Temporal4DFlowNet'
    patch_size = 16
    res_increase = 2
    
    # Residual blocks, default (8 LR ResBlocks and 4 HR ResBlocks)
    n_low_resblock = 8
    n_hi_resblock = 4
    low_res_block  = 'resnet_block' # 'resnet_block' 'dense_block' csp_block
    high_res_block = 'resnet_block' #'resnet_block'
    upsampling_block = 'linear'     #'Conv3DTranspose'#'nearest_neigbor'#'linear' #' 'linear'  'nearest_neigbor' 'Conv3DTranspose'
    post_processing_block = None #  'unet_block'#None#'unet_block'
    sampling = 'Cartesian' # this is not used for training but saved in the csv file for a better overview of what data it was trained on 

    shuffle = True       

    #notes: if something about this training is more 'special' is can be added to the overview csv file
    notes= 'Test L1 loss'
    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset =   load_indexes(validate_file)
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    if load_patches_all_axis: 
        z = PatchHandler4D_all_axis(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    else:
        z = PatchHandler4D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    trainset = z.initialize_dataset(trainset, shuffle=shuffle, n_parallel=None)

    # VALIDATION iterator
    if load_patches_all_axis: 
        valdh = PatchHandler4D_all_axis(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    else:
        valdh = PatchHandler4D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    valset = valdh.initialize_dataset(valset, shuffle=shuffle, n_parallel=None)

    # # Bechmarking dataset, use to keep track of prediction progress per best model
    testset = None
    if QUICKSAVE and benchmark_file is not None:
        # WE use this bechmarking set so we can see the prediction progressing over time
        benchmark_set = load_indexes(benchmark_file)
        if load_patches_all_axis: 
            ph = PatchHandler4D_all_axis(data_dir, patch_size, res_increase, batch_size, mask_threshold)
        else:
            ph = PatchHandler4D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
        # No shuffling, so we can save the first batch consistently
        testset = ph.initialize_dataset(benchmark_set, shuffle=False) 

    # ------- Main Network ------
    print(f"4DFlowNet Patch {patch_size}, lr {initial_learning_rate}, batch {batch_size}")
    network = TrainerController_temporal(patch_size, res_increase, initial_learning_rate, QUICKSAVE, network_name, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block =  upsampling_block, post_processing_block=post_processing_block, lr_decay_epochs=lr_decay_epochs)
    network.init_model_dir()

    if restore:
        print(f"Restoring model {model_file}...")
        network.restore_model(model_dir, model_file)
        print("Learning rate", network.optimizer.lr.numpy())
    
    # write into csv file

    write_settings_into_csv_file(overview_csv,network.unique_model_name, os.path.basename(training_file) , os.path.basename(validate_file), os.path.basename(benchmark_file), epochs,batch_size,patch_size, n_low_resblock, n_hi_resblock,upsampling_block, low_res_block, high_res_block, post_processing_block, sampling, notes)
    
    network.train_network(trainset, valset, n_epoch=epochs, testset=testset)
