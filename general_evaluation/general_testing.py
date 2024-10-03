import subprocess
import os

cases = os.listdir("checkpoints")

for case in cases:

    case_files = os.listdir(f"checkpoints/{case}")
    if [file for file in case_files if 'latest' in file] == []:
        next

    name = case
    dataset = "../../data/experimental_unest_resized/"
    stage = 'first'
    model = 'structured_trans'
    netG = "unest"
    modality = "mri2ct"

    fth = name.split("_")[1].replace('fth','')
    depth = name.split("_")[2].replace('depth','')
    nld = name.split("_")[3].replace('nld','')

    cp = ['best']

    for epoch in cp:
        # # Define the script and its arguments
        scripts_with_args = [
            ("test.py", [
                "--dataroot", dataset,
                "--gpu_ids", "0",
                "--model", "structured_trans",
                "--name", name,
                "--dataset_mode", "unaligned_mask",
                "--depth", depth,
                "--fth", fth,
                "--n_layers_D", nld,
                "--out_kernel", "7",
                "--preprocess", "none",
                "--netG", "unest",
                "--vit_img_size", "224", "224",
                "--batch_size", "32",
                "--no_flip",
                "--epoch", epoch
            ]),


        ("general_evaluation/voxel_creation.py", [
                "--results_folder", f"{name}/test_{epoch}/",
                "--final_voxels_folder", f"{name}/",
                "--size_input", "224"
            ]),
        
            ("general_evaluation/evaluation.py", [
                "--results_folder", f"{name}/",
                "--final_voxels_folder", f"{name}/",
                "--stage", stage,
                "--modality", modality

            ])]

        # Execute each script with arguments
        print(f'Results with epoch {epoch}')

        for script_and_argument in scripts_with_args:
            if len(script_and_argument) == 2:
                script, args = script_and_argument
                subprocess.run(["python", script] + args)
            else:
                subprocess.run(["python", script_and_argument])