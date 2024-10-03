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

class AlignedDataset(BaseDataset):
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
        
        self.A_paths = make_dataset(self.dir_A, opt.max_dataset_size)   # load images from '/path/to/data/trainA'
        self.A_paths = sorted(self.A_paths, key=lambda x: reindex(x))
        self.B_paths = make_dataset(self.dir_B, opt.max_dataset_size)    # load images from '/path/to/data/trainB'
        self.B_paths = sorted(self.B_paths, key=lambda x: reindex(x))

        if opt.half_data:
            self.A_paths, self.B_paths = self.A_paths[::2], self.B_paths[::2]

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        if opt.half:
            self.A_size, self.B_size = self.A_size//2, self.B_size//2
            self.A_paths, self.B_paths= self.A_paths[:self.A_size], self.B_paths[self.B_size:]
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

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
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.A_size]
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = io.read_image(A_path)
        B_img = io.read_image(B_path)
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)


        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
