# Temporal 4DFlowNet
Super Resolution 4D Flow MRI using Residual Neural Network

<!-- This repsository includes an implementation of the paper [4DFlowNet: Super-Resolution 4D Flow MRI](https://www.frontiersin.org/articles/10.3389/fphy.2020.00138/full) using Tensorflow 2.9.0 with Keras.  -->
<!-- In addition, there is an implementation for a temporal instead of a spatial-super resolution problem. -->

This is an extension of the repository [4DFlowNet](https://gitlab.eecs.umich.edu/bkhardy/4DFlowNet), which includes the implementation of the paper [4DFlowNet: Super-Resolution 4D Flow MRI](https://www.frontiersin.org/articles/10.3389/fphy.2020.00138/full) concerning spatial super resolution problem. 
This implemenational in focusing on increasing the temporal resolution of the CFD 4D Flow MRI data. 



# Example results

![alt text][logo]

[logo]: https://github.com/PiaaCaa/Temporal4DFlowNet/tree/develop-temporal-extended/example.gif "Logo Title Text 2"

<!-- Below are example prediction results from an actual 4D Flow MRI of a bifurcation phantom dataset. 

LowRes input (voxel size 4mm)
<p align="left">
    <img src="https://i.imgur.com/O48FbAh.gif" width="330">
</p>

High Res Ground Truth vs noise-free Super Resolution (2mm)
<p align="left">
    <img src="https://i.imgur.com/67CRdGn.gif" width="350">
</p>

High Res Ground Truth vs noise-free Super Resolution (1mm)
<p align="left">
    <img src="https://i.imgur.com/DMQa2Lr.gif" width="350">
</p> -->

<!-- # Enviroment Setup
Because Big Blue's default python interpreter is shared by everyone, I use venv
in order to create a personal environment that won't mess with C Heart or other
important codes. When your venv is activated, all changes made to the python environment
will be applied to your own personal profile, allowing you to make edits/explore code 
without worrying about affecting others.

1. Creating your virtual environment:

    Navigate to your home directory and type into the terminal: 

    <code>python3 -m venv ./venv </code>

    The files associated with your virtual environment should now be stored under 
    a directory called "venv".

2. Activating your virtual environment:

    This part is slightly trickier as it depends on your current working directory. Assuming
    that you are still in your home directory, your venv can be activated with

    <code>source venv/bin/activate </code>

    You should now see <code>(venv)</code> in your terminal window. If you are in another directory, the general
    format of the activation command is

    <code>source path/to/venv/bin/activate </code>

3) Deactivating your virtual environment:

    Simply type <code>deactivate</code> and venv will deactivate.

4) Installing relevant python packages:

    When your venv is activated, type in the command

    <code>pip install -r requirements.txt</code>

    This assumes that requirements.txt is in your current working directory. You can find requirements.txt
    in the 4DFlowNet base directory. Currently there is only one requirement that covers everything, but I will
    add to this file if anything else comes up. This will help to prevent package version conflicts between everyone.

    Your virtual environment should now be ready to run 4DFlowNet. For other questions (such as setting a default 
    interpreter in VS Code, etc.) you can reach me at bkhardy@umich.edu! -->


# General training setup from CFD data for temporal super-resolution

## Prepare dataset

To prepare training or validation dataset, we assume a High resolution CFD dataset is available. As an example we have provided this under TODO

How to prepare training/validation dataset.

    1. Generate lowres dataset
        >> Configure the datapath and filenames in prepare_temporal_lowres_dataset.py
        >> Run prepare_temporal_lowres_dataset.py
        >> This will generate a separate HDF5 file for the low resolution velocity data.
    2. Generate random patches from the LR-HR dataset pairs.
        >> Configure the datapath and filenames in prepare_patches.py
        >> For temporal problem, set .. to True. 
        >> Configure patch_size, rotation option, and number of patches per frame
        >> Run prepare_patches.py
        >> This will generate a csv file that contains the patch information.

## Training

The trainer accepts csv files to define training and validation. A benchmark set is used to keep prediction progress everytime a model is being saved as checkpoint. Example csv files are provided in the /data folder. TODO

To train a new 4DFlowNet Network:

    1. Put all data files (HDF5) and CSV patch index files in the same directory (e.g. /data)
    2. Open trainer_temporal.py and configure the data_dir and the csv filenames
    3. Adjust hyperparameters. The default values from the paper are already provided in the code.
    4. Run trainer_temporal.py

Adjustable parameters:

|Parameter  | Description   |
|------|--------------|
| patch_size| The input low resolution image will be split into isotropic patches. Adjust according to computation power and image size |
| res_increase| Upsample ratio. Adjustable to any integer. More upsample ratio requires more computation power. *Note*: res_increase=1 will denoise the image at the current resolution |
| batch_size| Batch size per prediction. Keep it low. |
| initial_learning_rate| Initial learning rate |
| epochs | number of epochs |
| mask_threshold| Mask threshold for non-binary mask. This is used to measure relative error (accuracy) |
| network_name | The network name. The model will be saved in this name_timestamp format |
|QUICKSAVE| Option to run a "bechmark" dataset everytime a model is saved |
|benchmark_file| A patch index file (CSV) contains a list of patches. Only the first batch will be read and run into prediction. |
| low_resblock | Number of residual blocks in low resolution space within 4DFlowNet. |
| hi_resblock | Number of residual blocks in high resolution space within 4DFlowNet. |


## Standard Aortic Training Setup
|Parameter  | Value   |
|------|--------------|
| patch_size| 16 |
| res_increase| 2 |
| batch_size| 20 |
| initial_learning_rate| 1e-4 |
| epochs | 150 |
| mask_threshold| 0.6 |
| network_name | 4DFlowNet-aortic |
|QUICKSAVE| True |
|training_file| aorta0102_patches.csv |
|validate_file| aorta03_patches.csv |
|benchmark_file| aorta03_patches.csv |
| low_resblock | 8 |
| hi_resblock | 4 |

<!-- ## Standard Cerebrovascular Training Setup
|Parameter  | Value   |
|------|--------------|
| patch_size| 12 |
| res_increase| 2 |
| batch_size| 20 |
| initial_learning_rate| 2e-4 |
| epochs | 60 |
| mask_threshold| 0.6 |
| network_name | 4DFlowNet-cerebro |
|QUICKSAVE| True |
|training_file| newtrain12.csv |
|validate_file| newval12.csv |
|benchmark_file| newbenchmark12.csv |
| low_resblock | 8 |
| hi_resblock | 4 | -->


<!-- # Running prediction on MRI data
## Prepare data from MRI (for prediction purpose)
*NOTE*: all of the provided datasets are already in HDF5 format, making this step unnecessary for current use cases.
To prepare 4D Flow MRI data to HDF5, go to the prepare_data/ directory and run the following script:

    >> python prepare_data.py --input-dir [4DFlowMRI_CASE_DIRECTORY]

    >> usage: prepare_mri_data.py [-h] --input-dir INPUT_DIR
                           [--output-dir OUTPUT_DIR]
                           [--output-filename OUTPUT_FILENAME]
                           [--phase-pattern PHASE_PATTERN]
                           [--mag-pattern MAG_PATTERN] [--fh-mul FH_MUL]
                           [--rl-mul RL_MUL] [--in-mul IN_MUL] 

Notes: 
*  The directory must contains the following structure:
    [CASE_NAME]/[Magnitude_or_Phase]/[TriggerTime]
* There must be exactly 3 Phase and 3 Magnitude directories 
* To get the required directory structure, [DicomSort](https://dicomsort.com/) is recommended. Sort by SeriesDescription -> TriggerTime.
* In our case, VENC and velocity direction is read from the SequenceName DICOM HEADER. Code might need to be adjusted if the criteria is different. -->

## Prediction

To run a prediction:

    1. Go to src/ and open predictor_temporal.py and configure the input_filename and output_filename if necessary
    2. Run predictor_temporal.py

Adjustable parameters:

|Param  | Description   | Default|
|------|--------------|--------:|
| patch_size| The image will be split into isotropic patches. Adjust according to computation power and image size.  | 24|
| res_increase| Upsample ratio. Adjustable to any integer. More upsample ratio requires more computation power. *Note*: res_increase=1 will denoise the image at the current resolution |2|
| batch_size| Batch size per prediction. Keep it low. |8|
| round_small_values|Small values are rounded down to zero. Small value is calculated based on venc, according to Velocity per 1 pixel value = venc/2048 |True|
| low_resblock | Number of residual blocks in low resolution space within 4DFlowNet. |8|
| hi_resblock | Number of residual blocks in high resolution space within 4DFlowNet. |4|



<!-- ## Contact Information

If you encounter any problems, feel free to contact me by email.

Pia Callmer -->