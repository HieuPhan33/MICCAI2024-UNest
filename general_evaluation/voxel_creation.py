# Import libraries
import nibabel as nib
import numpy as np
from PIL import Image
import os
import argparse
import shutil
from scipy import ndimage

# Call the parser
parser = argparse.ArgumentParser()
parser.add_argument('--results_folder', default='results/secondstage_chunk_ssim2b/', help='path to intermediate results (2nd stage)')
parser.add_argument('--final_voxels_folder', default='second_stage_chunks/', help='path to leave input of 2nd stage')
parser.add_argument('--size_input', default=224, type = int, help='length of sides of input image')
args = parser.parse_args()

# Extract the numeric part of images of unpaired images
def extract_numeric_part_unpaired(file_name):
    return int(file_name.split('_')[3])

# Extract the numeric part of images of paired images
def extract_numeric_part_paired(file_name):
    return int(file_name.split('_')[2])

# Remove output folder if already exists
if os.path.exists('vol_results/' + args.final_voxels_folder):
    shutil.rmtree('vol_results/' + args.final_voxels_folder)
# Create output folder
os.makedirs('vol_results/' + args.final_voxels_folder)
os.makedirs('vol_results/' + args.final_voxels_folder + 'real_A')
os.makedirs('vol_results/' + args.final_voxels_folder + 'real_B')
os.makedirs('vol_results/' + args.final_voxels_folder + 'fake_A')
os.makedirs('vol_results/' + args.final_voxels_folder + 'fake_B')

def resize_volume(img):

    desired_depth = args.size_input
    desired_width = args.size_input
    desired_height = args.size_input

    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]

    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img         

for folder in ['real_A','real_B','fake_A','fake_B']:

    # Define main subdirectories
    sct_folder = 'results/' + args.results_folder + folder + '/'

    # Obtain the images
    sct_images = os.listdir(sct_folder)

    # Remove output folder if already exists
    if os.path.exists('vol_results/' +args.final_voxels_folder + folder):
        shutil.rmtree('vol_results/' +args.final_voxels_folder + folder)
    # Create output folder
    os.makedirs('vol_results/' +args.final_voxels_folder + folder)

    # Identify the voxels id
    voxels_idx = np.unique([img.split("_")[0] for img in sct_images if 'paired' not in img])

    # Filter per orientation
    voxels_idx_orientation = voxels_idx

    # Create voxels 1 by 1
    for voxel_idx in voxels_idx_orientation:

        # Concatenate images
        slicing_image = [img for img in sct_images if img.split("_")[0] == voxel_idx]
        # Sort the list using the custom sorting key
        sorted_slicing_names = sorted(slicing_image, key=extract_numeric_part_unpaired)

        # Check how many images I have per voxel
        images_voxel = len(sorted_slicing_names)

        # Concatenate 2D Slices
        for idx, img in enumerate(sorted_slicing_names):

            # Open a PNG image
            image_path = sct_folder +  img
            image = Image.open(image_path)

            # Convert the image to a numpy array
            image_array = np.array(image)
            image_array = image_array.reshape(1,args.size_input,args.size_input)

            # Incorporate slices in the template
            if idx == 0:
                image3d = image_array
            else:
                image3d = np.concatenate((image3d,image_array), axis = 0)

        # Rotate to get the proper orientation
        image3d = np.rot90(image3d, axes = (0, 2))
        image3d = np.flip(image3d, axis = 0)
        image3d = resize_volume(image3d)
    
        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(image3d, affine=np.diag([1, 1, 1, 1]))  # Assuming an identity affine matrix

        # Save the NIfTI image to a file
        #print('saving image ', 'vol_results/' + args.final_voxels_folder + folder + '/' + voxel_idx +'.nii.gz')
        nib.save(nifti_img, 'vol_results/' + args.final_voxels_folder + folder +'/' + voxel_idx +'.nii.gz')
            