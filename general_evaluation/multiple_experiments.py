# Trying multiple experiments

# Import libraries
import subprocess
import os

# Defining main variables
parameter_lambda_seg = [0.5,0.75, 1]
parameter_fth = [0.5,0.75, 1]
parameter_depth = [3,5,7]
parameter_n_layers_D = [3,5,7]
parameter_structured_shape_iter = [0,50]

list_experiments = []
for lambda_seg in parameter_lambda_seg:
    for fth in parameter_fth:
        for depth in parameter_depth:
            for n_layers_D in parameter_n_layers_D:
                for structured_shape_iter in parameter_structured_shape_iter:
                    experiment = {}
                    experiment["lambda_seg"] = lambda_seg
                    experiment["fth"] = fth
                    experiment["depth"] = depth
                    experiment["n_layers_D"] = n_layers_D
                    experiment["structured_shape_iter"] = structured_shape_iter
                    experiment["name"] = f"unest_ls{lambda_seg}_fth{fth}_depth{depth}_nld{n_layers_D}_ssi{structured_shape_iter}"
                    list_experiments.append(experiment)

for exp in list_experiments:
    # Construct the command to run train.py with the current set of parameters
    command = [
        "python", "train.py",
        "--dataroot", "../../data/adult-2d-unsupervised/",
        "--gpu_ids", "1",
        "--display_id", "0",
        "--model", "structured_trans",
        "--name", str(exp["name"]),
        "--dataset_mode", "unaligned_mask",
        "--depth", str(exp["depth"]),
        "--structured_shape_iter", str(exp["structured_shape_iter"]),
        "--lambda_seg", str(exp["lambda_seg"]),
        "--fth", str(exp["fth"]),
        "--out_kernel", "7",
        "--preprocess", "none",
        "--netG", "unest",
        "--vit_img_size", "224"," 224",
        "--window_size", "2",
        "--batch_size", "28",
        "--n_epochs", "50",
        "--n_layers_D", str(exp["n_layers_D"]),
        "--n_epochs_decay", "50",
        "--display_freq", "5000",
        "--print_freq", "1000",
        "--save_epoch_freq", "25",
        "--save_latest_freq", "500",
        "--Aclass", "A",
        "--Bclass", "B",
        '--no_flip'
    ]
    
    # Run the command using subprocess
    result = subprocess.run(command, capture_output=True, text=True)