import numpy as np
import os
from Network.PatchHandler3D_temporal import PatchHandler4D
from Network.TrainerController_temporal import TrainerController_temporal
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

if __name__ == "__main__":
    data_dir = 'Temporal4DFlowNet/data/CARDIAC'
    
    # ---- Patch index files ----
    training_file = '{}/Temporal10MODEL12_2_temporal.csv'.format(data_dir) 
    validate_file = '{}/Temporal10MODEL3_2_temporal.csv'.format(data_dir)

    QUICKSAVE = True
    benchmark_file = '{}/Temporal10MODEL3_2_temporal.csv'.format(data_dir)
    
    restore = False
    if restore:
        model_dir = "4DFlowNet/models/4DFlowNet"
        model_file = "4DFlowNet-best.h5"

    # Hyperparameters optimisation variables
    initial_learning_rate = 2e-4
    epochs =  400
    batch_size = 20
    mask_threshold = 0.6

    # Network setting
    network_name = 'Temporal4DFlowNet'
    patch_size = 10
    res_increase = 2
    # Residual blocks, default (8 LR ResBlocks and 4 HR ResBlocks)
    low_resblock = 8
    hi_resblock = 4

    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset = load_indexes(validate_file)
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    z = PatchHandler4D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    trainset = z.initialize_dataset(trainset, shuffle=True, n_parallel=None)

    # VALIDATION iterator
    valdh = PatchHandler4D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    valset = valdh.initialize_dataset(valset, shuffle=True, n_parallel=None)

    # # Bechmarking dataset, use to keep track of prediction progress per best model
    testset = None
    if QUICKSAVE and benchmark_file is not None:
        # WE use this bechmarking set so we can see the prediction progressing over time
        benchmark_set = load_indexes(benchmark_file)
        ph = PatchHandler4D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
        # No shuffling, so we can save the first batch consistently
        testset = ph.initialize_dataset(benchmark_set, shuffle=False) 

    # ------- Main Network ------
    print(f"4DFlowNet Patch {patch_size}, lr {initial_learning_rate}, batch {batch_size}")
    network = TrainerController_temporal(patch_size, res_increase, initial_learning_rate, QUICKSAVE, network_name, low_resblock, hi_resblock)
    network.init_model_dir()

    if restore:
        print(f"Restoring model {model_file}...")
        network.restore_model(model_dir, model_file)
        print("Learning rate", network.optimizer.lr.numpy())

    network.train_network(trainset, valset, n_epoch=epochs, testset=testset)
