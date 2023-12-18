#!/bin/bash

# Array of file paths
# file_paths=(
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M1_2mm_step2_invivoP01_magnitude_adapted_noisy.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M2_2mm_step2_invivoP04_magnitude_adapted_noisy.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M3_2mm_step2_invivoP03_magnitude_adapted_noisy.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M4_2mm_step2_invivoP02_magnitude_adapted_noisy.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M1_2mm_step2_invivoP01_magnitude_adapted.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M2_2mm_step2_invivoP04_magnitude_adapted.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M3_2mm_step2_invivoP03_magnitude_adapted.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M4_2mm_step2_invivoP02_magnitude_adapted.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M1_2mm_step2_invivoP01_magnitude_noisy.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M2_2mm_step2_invivoP04_magnitude_noisy.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M3_2mm_step2_invivoP03_magnitude_noisy.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M4_2mm_step2_invivoP02_magnitude_noisy.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M1_2mm_step2_invivoP01_magnitude.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M2_2mm_step2_invivoP04_magnitude.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M3_2mm_step2_invivoP03_magnitude.h5",
#     "/proj/multipress/users/x_piaca/Temporal4DFlowNet/data/CARDIAC/M4_2mm_step2_invivoP02_magnitude.h5",
# )

# Loop through each file path
for file_path in "${file_paths[@]}"; do
    # Check if the file exists
    if [ -f "$file_path" ]; then
        # Remove the file
        rm "$file_path"
        echo "File '$file_path' removed successfully."
    else
        echo "File '$file_path' does not exist."
    fi
done
