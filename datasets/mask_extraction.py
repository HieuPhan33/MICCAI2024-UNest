import numpy as np
import matplotlib.pyplot as plt
import ants
import pandas as pd
import cv2
# import imageio
from PIL import Image
import torchvision.transforms.functional as TF
import math

import imageio
import os
import glob
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        #color_mask = [0,0,1]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

#### SAM

sam_checkpoint = "sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamAutomaticMaskGenerator(sam)

####

def visualize(img, filename, step=10):
    shapes = img.shape
    for i, shape in enumerate(shapes):
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(12,12))
        start = shape//2 - step*4
        for t, ax in enumerate(axes.flatten()):
            if i == 0:
                data = img[start + t*step, :, :]
            elif i == 1:
                data = img[:, start+t*step, :]
            else:
                data = img[:, :, start+t*step]
            ax.imshow(data, cmap='gray', origin='lower')
            ax.axis('off')
        fig.tight_layout()
        plt.savefig(f'{filename}_{i}.png')
        plt.clf()

def normalize(img, min_, max_):
    return (img - min_)/(max_ - min_)

def preprocess(img, crop=0, crop_h=0, ignore_zero=True):
    img = np.transpose(img, (0,2,1))[:,::-1,::-1]

    if ignore_zero:
        mask_ = img.sum(axis=(1,2)) > 0
        img = img[mask_]

    if crop > 0:
        length = img.shape[0]
        img = img[int(crop*length): int((1-crop)*length)]

    
    if crop_h > 0:
        if img.shape[1] > 200:
            crop_h = 0.8
        new_h = int(crop_h*img.shape[1])
        img = img[:, :new_h]

    return img

def get_sam_bg_mask_points(model, img):
    model.set_image(image)
    input_point = np.array([[img.shape[1]//2, 0]])
    input_label = np.array([0])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    return masks, scores, logits


def get_sam_fg_mask(sam_model, image, embedding=False):
    masks = sam_model.generate(image)

    ### Embedding
    image_embedding = None
    if embedding:
        sam_model.predictor.set_image(image)
        image_embedding = sam_model.predictor.get_image_embedding().cpu().numpy()

    #largest_mask = [masks[np.argmax([m['area'] for m in masks])]]
    largest_masks = sorted(masks, key=lambda d: d['area'], reverse=True)[:5]
    i = 0
    while np.average(image[largest_masks[i]['segmentation']]) < 0.5:
        i += 1
        if i == len(largest_masks):
            break
    if i == len(largest_masks):
        return np.zeros((image.shape[0], image.shape[1])), image_embedding
    return largest_masks[i]['segmentation'], image_embedding


def clip_values(img, min_=None, max_=None, th=50):
    if min_ is None:
        min_ = img.min()
    if max_ is None:
        max_ = img.max()
    # print(np.isnan(img))
    # print(np.isnan(img).any())

    if np.any(np.isnan(img)):
        print("Has Nan")
        img[np.isnan(img)] = min_
    img = np.clip(img, min_, max_)
    img = np.uint8(255*normalize(img, min_, max_))

    mask = np.zeros(img.shape).astype(np.int32)
    mask[img > th] = 1

    return img, mask


### Train
data_dir = '/data/data/MRXFDG-PET-CT-MRI'
out_dir = f'/data2/data/processed-pet-mri'
root_a = f'{data_dir}/MNI/*/*T1w.nii.gz'
root_b = f'{data_dir}/MNI/*/*pet.nii.gz'
root_c = f'{data_dir}/MNI/*/*ct.nii.gz'
output_a_dir = f'{out_dir}/train_A'
output_b_dir = f'{out_dir}/train_B'
output_c_dir = f'{out_dir}/train_C'
output_a_mask_dir = f'{out_dir}/train_maskA'
output_b_mask_dir = f'{out_dir}/train_maskB'
output_c_mask_dir = f'{out_dir}/train_maskC'

output_a_val_dir = f'{out_dir}/val_A'
output_b_val_dir = f'{out_dir}/val_B'
output_c_val_dir = f'{out_dir}/val_C'
output_a_mask_val_dir = f'{out_dir}/val_maskA'
output_b_mask_val_dir = f'{out_dir}/val_maskB'
output_c_mask_val_dir = f'{out_dir}/val_maskC'

output_a_test_dir = f'{out_dir}/test_A'
output_b_test_dir = f'{out_dir}/test_B'
output_c_test_dir = f'{out_dir}/test_C'
output_a_mask_test_dir = f'{out_dir}/test_maskA'
output_b_mask_test_dir = f'{out_dir}/test_maskB'
output_c_mask_test_dir = f'{out_dir}/test_maskC'

os.makedirs(output_a_dir, exist_ok=True)
os.makedirs(output_b_dir, exist_ok=True)
os.makedirs(output_c_dir, exist_ok=True)
os.makedirs(output_a_mask_dir, exist_ok=True)
os.makedirs(output_b_mask_dir, exist_ok=True)
os.makedirs(output_c_mask_dir, exist_ok=True)

os.makedirs(output_a_val_dir, exist_ok=True)
os.makedirs(output_b_val_dir, exist_ok=True)
os.makedirs(output_c_val_dir, exist_ok=True)
os.makedirs(output_a_mask_val_dir, exist_ok=True)
os.makedirs(output_b_mask_val_dir, exist_ok=True)
os.makedirs(output_c_mask_val_dir, exist_ok=True)

os.makedirs(output_a_test_dir, exist_ok=True)
os.makedirs(output_b_test_dir, exist_ok=True)
os.makedirs(output_c_test_dir, exist_ok=True)
os.makedirs(output_a_mask_test_dir, exist_ok=True)
os.makedirs(output_b_mask_test_dir, exist_ok=True)
os.makedirs(output_c_mask_test_dir, exist_ok=True)

a_files = sorted(glob.glob(root_a))
b_files = sorted(glob.glob(root_b))
c_files = sorted(glob.glob(root_c))
train_len = int(len(a_files)*0.8)
val_len = int(len(a_files)*0.1)

train_idx = np.arange(0, train_len)
val_idx = np.arange(train_len, train_len + val_len)

#resample=(1.0, 1.0, 1.0)
th = 0
results = 'vis'
os.makedirs(results, exist_ok=True)
overwrite=True


print("Process PET scans")
for idx, filepath in enumerate(tqdm(b_files)):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    if not overwrite and len(glob.glob(f'{output_b_dir}/{filename}*')) > 0:
        continue
    img = ants.image_read(filepath)
    print("PET resolution ", img.shape)
    #img = ants.resample_image(img, resample, False, 1)
    img = img.numpy()
    
    img, mask = clip_values(img, th=th)
    img = img*mask
    ignore_zero = idx in train_idx
        
    img = preprocess(img, crop=0.0, crop_h=0.0, ignore_zero=ignore_zero)
    for i in range(len(img)):
        image = np.uint8(255*normalize(img[i], img[i].min(), img[i].max()))
        image_ = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask, emb = get_sam_fg_mask(predictor, image_, embedding=False)
        if idx in train_idx:
            imageio.imwrite(f'{output_b_dir}/{filename}_{i}.png', image)
            imageio.imwrite(f'{output_b_mask_dir}/{filename}_{i}.png', np.uint8(255*mask))
        elif idx in val_idx:
            imageio.imwrite(f'{output_b_val_dir}/{filename}_{i}.png', image)
            imageio.imwrite(f'{output_b_mask_val_dir}/{filename}_{i}.png', np.uint8(255*mask))
        else:
            imageio.imwrite(f'{output_b_test_dir}/{filename}_{i}.png', image)
            imageio.imwrite(f'{output_b_mask_test_dir}/{filename}_{i}.png', np.uint8(255*mask))




print("Process MRI scans")

for idx, filepath in enumerate(tqdm(a_files)):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    if not overwrite and len(glob.glob(f'{output_b_dir}/{filename}*')) > 0:
        continue
    img = ants.image_read(filepath)
    print("MRI resolution ", img.shape)
    #img = ants.resample_image(img, resample, False, 1)
    img = img.numpy()
    
    img, mask = clip_values(img, th=25)
    img = img*mask
    ignore_zero = idx in train_idx

    img = preprocess(img, crop=0.0, crop_h=0.0, ignore_zero=ignore_zero)
    
    for i in range(len(img)):
        image = np.uint8(255*normalize(img[i], img[i].min(), img[i].max()))
        #image = image.transpose((1, 0))[::-1, ::-1]
        # image = image[..., None]
        image_ = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #print(image.shape)
        mask, emb = get_sam_fg_mask(predictor, image_, embedding=False)
        #print(f"{filename}", largest_masks[0]['area']/(image.shape[0] * image.shape[1]))
        if idx in train_idx:
            imageio.imwrite(f'{output_a_dir}/{filename}_{i}.png', image)
            imageio.imwrite(f'{output_a_mask_dir}/{filename}_{i}.png', np.uint8(255*mask))
        elif idx in val_idx:
            imageio.imwrite(f'{output_a_val_dir}/{filename}_{i}.png', image)
            imageio.imwrite(f'{output_a_mask_val_dir}/{filename}_{i}.png', np.uint8(255*mask))
        else:
            imageio.imwrite(f'{output_a_test_dir}/{filename}_{i}.png', image)
            imageio.imwrite(f'{output_a_mask_test_dir}/{filename}_{i}.png', np.uint8(255*mask))

print("Process CT scans")
for idx, filepath in enumerate(tqdm(c_files)):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    if not overwrite and len(glob.glob(f'{output_c_dir}/{filename}*')) > 0:
        continue
    img = ants.image_read(filepath)
    print("CT resolution ", img.shape)
    #img = ants.resample_image(img, resample, False, 1)
    img = img.numpy()
    img[img == 0] = -800
    
    img, mask = clip_values(img, th=25, min_=-800, max_=2000)
    img = img*mask
    ignore_zero = idx in train_idx

    img = preprocess(img, crop=0.0, crop_h=0.0, ignore_zero=ignore_zero)
    
    for i in range(len(img)):
        image = np.uint8(255*normalize(img[i], img[i].min(), img[i].max()))
        image_ = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #print(image.shape)
        mask, emb = get_sam_fg_mask(predictor, image_, embedding=False)
        #print(f"{filename}", largest_masks[0]['area']/(image.shape[0] * image.shape[1]))
        if idx in train_idx:
            imageio.imwrite(f'{output_c_dir}/{filename}_{i}.png', image)
            imageio.imwrite(f'{output_c_mask_dir}/{filename}_{i}.png', np.uint8(255*mask))
        elif idx in val_idx:
            imageio.imwrite(f'{output_c_val_dir}/{filename}_{i}.png', image)
            imageio.imwrite(f'{output_c_mask_val_dir}/{filename}_{i}.png', np.uint8(255*mask))
        else:
            imageio.imwrite(f'{output_c_test_dir}/{filename}_{i}.png', image)
            imageio.imwrite(f'{output_c_mask_test_dir}/{filename}_{i}.png', np.uint8(255*mask))  