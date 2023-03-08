import numpy as np
import os
import csv
from Network.PatchHandler3D_temporal import PatchHandler4D, PatchHandler4D_all_axis
from Network.TrainerController_temporal import TrainerController_temporal
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

def write_settings_into_csv_file(filename,name, training_file, validation_file, test_file, epochs,batch_size,patch_size, low_resblock, high_resblock, notes):
    print("Write settings into overview file")
    fieldnames = ["Name","training_file","validation_file","test_file","epochs","batch_size","patch_size","res_increase","low_resblock","high_resblock","notes"]
    with open(filename, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'Name':name, "training_file":training_file, "validation_file":validation_file, "test_file":test_file, "epochs":epochs, "batch_size":batch_size, "patch_size":patch_size, "res_increase":res_increase, "low_resblock":low_resblock, "high_resblock":high_resblock, "notes":notes })

if __name__ == "__main__":
    data_dir = 'Temporal4DFlowNet/data/CARDIAC'
    
    # ---- Patch index files ----
    training_file = '{}/Temporal14MODEL23_2mm_step2_all_axis_extended_dynamic_mask.csv'.format(data_dir) 
    validate_file = '{}/Temporal14MODEL1_2mm_step2_all_axis_extended_dynamic_mask.csv'.format(data_dir)

    QUICKSAVE = True
    benchmark_file = '{}/Temporal14MODEL4_2mm_step2_all_axis_extended_dynamic_mask.csv'.format(data_dir)
    
    overview_csv = 'Temporal4DFlowNet/results/Overview_models.csv'

    restore = False
    if restore:
        model_dir = "4DFlowNet/models/4DFlowNet"
        model_file = "4DFlowNet-best.h5"

    # Adapt how patches are saved for temporal domain if True a different loading scheme is used
    load_patches_all_axis = True

    # if load_patches_all_axis:
    #     assert #TODO, check that title is correct, since it implied which kind of loading it uses

    # Hyperparameters optimisation variables
    initial_learning_rate = 2e-4
    epochs =  70
    batch_size = 32
    mask_threshold = 0.6

    # Network setting
    network_name = 'Temporal4DFlowNet'
    patch_size = 14
    res_increase = 2
    # Residual blocks, default (8 LR ResBlocks and 4 HR ResBlocks)
    low_resblock = 8
    hi_resblock = 4
    block= 'resnet_block' # 'resnet_block' 'dense_block' csp_block
    upsampling_block = 'default'#'Conv3Dtranspose'

    #notes: if something about this training is more 'special' is can be added to the overview csv file
    notes= 'Retraining on dynamical mask'

    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset =   load_indexes(validate_file)
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    if load_patches_all_axis: 
        z = PatchHandler4D_all_axis(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    else:
        z = PatchHandler4D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    trainset = z.initialize_dataset(trainset, shuffle=True, n_parallel=None)

    # VALIDATION iterator
    if load_patches_all_axis: 
        valdh = PatchHandler4D_all_axis(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    else:
        valdh = PatchHandler4D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    valset = valdh.initialize_dataset(valset, shuffle=True, n_parallel=None)

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
    network = TrainerController_temporal(patch_size, res_increase, initial_learning_rate, QUICKSAVE, network_name, low_resblock, hi_resblock, block, upsampling_block =  upsampling_block)
    network.init_model_dir()

    if restore:
        print(f"Restoring model {model_file}...")
        network.restore_model(model_dir, model_file)
        print("Learning rate", network.optimizer.lr.numpy())
    
    # write into csv file

    write_settings_into_csv_file(overview_csv,network.unique_model_name, os.path.basename(training_file) , os.path.basename(validate_file), os.path.basename(benchmark_file), epochs,batch_size,patch_size, low_resblock, hi_resblock, notes)
    
    network.train_network(trainset, valset, n_epoch=epochs, testset=testset)
