import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
random.seed(100)
from torchvision import io

def reindex(file_path):
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)

    x_value = int(name.split('_')[-1])
    x_padded = f"{x_value:10d}"

    new_filename = name.replace(f"_{x_value}", f"_{x_padded}")
    new_file_path = os.path.join(directory, new_filename + ext)
    return new_file_path
    
class UnalignedMaskDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + f'_{opt.Aclass}')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + f'_{opt.Bclass}')  # create a path '/path/to/data/trainB'
        self.dir_maskA = os.path.join(opt.dataroot, opt.phase + f'_mask{opt.Aclass}')  # create a path '/path/to/data/trainA'
        self.dir_maskB = os.path.join(opt.dataroot, opt.phase + f'_mask{opt.Bclass}')  # create a path '/path/to/data/trainB'
        btoA = self.opt.direction == 'BtoA'
        # if btoA:
        #     self.dir_A, self.dir_B, self.dir_maskA, self.dir_maskB = self.dir_B, self.dir_A, self.dir_maskB, self.dir_maskA
        # print(f"Load A from {self.dir_A}")
        # print(f"Load B from {self.dir_B}")
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.maskA_paths = sorted(make_dataset(self.dir_maskA, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.maskB_paths = sorted(make_dataset(self.dir_maskB, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        # if opt.half_data:
        #     self.A_paths, self.B_paths, self.maskA_paths, self.maskB_paths = self.A_paths[::2], self.B_paths[::2], self.maskA_paths[::2], self.maskB_paths[::2]
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        if opt.half: ## Select first half and last half for two modalities
            self.A_size, self.B_size = self.A_size//2, self.B_size//2
            self.A_paths, self.maskA_paths, self.B_paths, self.maskB_paths = self.A_paths[:self.A_size], self.maskA_paths[:self.A_size], self.B_paths[self.B_size:], self.maskB_paths[self.B_size:]
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        # Save relative position of each img
        self.relative_pos_A = [int(img.split(".")[-2].split("_")[-1]) for img in self.A_paths]
        self.relative_pos_B = [int(img.split(".")[-2].split("_")[-1]) for img in self.B_paths]

        # Define range of adjacent slices to consider
        if opt.phase == 'train':
            self.position_based_range = opt.position_based_range*10


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   
            #index_B = random.randint(0, self.B_size - 1)
            
            # Randomize the index for domain B to avoid fixed pairs.
            # Check the relative position of the image (Position based selection PBS)
            A_path_spplited = A_path.split(".")
            A_relative_position = A_path_spplited[-2].split("_")[-1]
            # Convert to a number
            A_relative_position = float(A_relative_position)
            # Obtain the images in a similar range (Position based selection)
            potential_indexes = [index for index, value in enumerate(self.relative_pos_B) if (A_relative_position-self.position_based_range) <= value <= (A_relative_position + self.position_based_range)]
            # Define position of B image
            potential_indexes = list(set(potential_indexes) & set(potential_indexes))
            index_position = random.randint(0, len(potential_indexes) - 1)
            index_B = potential_indexes[index_position]
            
        # Select the proper images
        B_path = self.B_paths[index_B]
        maskA_path = self.maskA_paths[index_A]
        maskB_path = self.maskB_paths[index_B]
        
        # Select images A and B
        A_img = io.read_image(A_path)
        B_img = io.read_image(B_path)

        # Select the masks A and B
        A_mask = io.read_image(maskA_path)
        B_mask = io.read_image(maskB_path)
        # apply image transformation
        A, A_mask = self.transform_A(A_img, A_mask)
        B, B_mask = self.transform_B(B_img, B_mask)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,
        'A_mask': A_mask, 'B_mask': B_mask}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
