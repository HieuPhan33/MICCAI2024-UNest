# MICCAI2024-UNest

This is the official repo for our MICCAI 2024 paper: "Structural Attention: Rethinking Transformer for Unpaired Medical Image Synthesis".

## Setting Docker environment

Create a Docker image based on the `Dockerfile` available in this repository using `docker build -t unest:latest .` 
After that you can run:

```
docker run --name unest --gpus all --shm-size=16g -it -v /path/to/data/root:/data unest:latest
```

## Installation
- ```pip install -r requirements.txt```
- Download SAM ckpt: ```wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth```

## Data preparation
Download MR-PET-CT MRXFDG dataset. Run the code `preprocessing_images.py` to process images in the proper format for UNest and getting the masks. 

## Train UNest

Here are the sampled command to train UNest for MR to CT translation.
```
python train.py --dataroot ../../data/ --gpu_ids 0 --display_id 0 --model structured_trans \ 
--name base_model --dataset_mode unaligned_mask depth 4 --structured_shape_iter 0 \
--lambda_seg 1 --fth 0.75 --out_kernel 7 --load_size 200 --pad_size 224 --preprocess \ 
resize_pad --netG unest --vit_img_size 224 224 --window_size 2 --batch_size 32 \ 
--n_epochs 50 --n_epochs_decay 50 --display_freq 5000 --print_freq 1000 --save_epoch_freq 5 \
--save_latest_freq 500 --no_flip --Aclass A --Bclass B --include_mask_val --n_layers_D 3
```

Notable arguments include:
- structured_shape_iter: Directly use ground-truth segmentation instead of the predicted segmentation in the first few iterations.
- lambda_seg: weights of segmentation loss
- fth: threshold to determine whether a pixel is foreground or background
- out_kernel: the final kernel of the decoder to regress each pixel
- vit_img_size: match the image resolution # Only modify when changing iamge resolution
- upsample: what convolution techniques used to upsampling the image features in the decoder
- window_size: window size for local attention # Divisible by the height/width of images


## Related Projects
[cyclegan-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
[ViT-V-Net](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch) |
[Recursive-Cascade-Networks](https://github.com/microsoft/Recursive-Cascaded-Networks) <br>

Please check out our concurrent work on unpaired medical image synthesis: [MaskGAN](https://github.com/HieuPhan33/MaskGAN)
