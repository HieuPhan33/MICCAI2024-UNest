
import PIL
import torch
import numpy as np
import wandb
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import monai

def val_visualizations_over_batches(real_a, 
                                    real_b, 
                                    fake_b):
    """We define a function to visualizate val images of sets
        real_A, real_B and fake_B
    """
    results = []
    # Save real A
    real_a = real_a.cpu().numpy()
    real_a = real_a[4:8,:,:,:]
    real_a = np.concatenate(real_a, axis=1)
    # Save real B
    real_b = real_b.cpu().numpy()
    real_b = real_b[4:8,:,:,:]
    real_b = np.concatenate(real_b, axis=1)
    # Save prediction
    fake_b = fake_b.cpu().numpy()
    fake_b = fake_b[4:8,:,:,:]
    fake_b = np.concatenate(fake_b, axis=1)

    # Append results to a final output
    results.append(real_a)
    results.append(real_b)
    results.append(fake_b)

    # Transforming results to tuple
    return tuple(results)

def val_mask_visualizations_over_batches(
                                    mask_a,
                                    mask_b,
                                    mask_seg_a,
                                    mask_seg_b):
    """We define a function to visualizate mask val images
    """
    results = []
    mask_a = mask_a.cpu().numpy()
    mask_a = mask_a[4:8,:,:,:]
    mask_a = np.concatenate(mask_a, axis=1)
    results.append(mask_a)
    mask_b = mask_b.cpu().numpy()
    mask_b = mask_b[4:8,:,:,:]
    mask_b = np.concatenate(mask_b, axis=1)
    results.append(mask_b)
    mask_seg_a = mask_seg_a.cpu().numpy()
    mask_seg_a = mask_seg_a[4:8,:,:,:]
    mask_seg_a = np.concatenate(mask_seg_a, axis=1)
    results.append(mask_seg_a)
    mask_seg_b = mask_seg_b.cpu().numpy()
    mask_seg_b = mask_seg_b[4:8,:,:,:]
    mask_seg_b = np.concatenate(mask_seg_b, axis=1)
    results.append(mask_seg_b)

    # Transforming results to tuple
    return tuple(results)

def val_visualizations_over_chunks(real_a, real_b, fake_b):
    """Another auxiliar function to show visualizations over chunks
    """
    # Save real A
    real_a = real_a.cpu().numpy()
    real_a = real_a[0,4:8,:,:]
    real_a = np.concatenate(real_a, axis=0)
    # Save real B
    real_b = real_b.cpu().numpy()
    real_b = real_b[0,4:8,:,:]
    real_b = np.concatenate(real_b, axis=0)
    # Save fake B
    fake_b = fake_b.cpu().numpy()
    fake_b = fake_b[0,4:8,:,:]
    fake_b = np.concatenate(fake_b, axis=0)

    return real_a, real_b, fake_b

def validation(val_set, model, opt):
    #ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    # Getting MAE
    metric_mae = torch.nn.L1Loss()
    # Getting MSE
    metric_mse = torch.nn.MSELoss()
    # Getting SSIM
    metric_ssim = monai.metrics.SSIMMetric(spatial_dims=2, reduction = 'mean')
    # Getting PSNR
    metric_psnr = PeakSignalNoiseRatio()

    model.eval()
    # Set zero
    mae_base = 0.0
    mse_base = 0.0
    ssim_base = 0.0
    psnr_base= 0.0
    mae_fake = 0.0
    mse_fake = 0.0
    ssim_fake = 0.0
    psnr_fake = 0.0

    # Check number of total batches for validation set
    batches = len(val_set.dataloader)
    # Select values which represent 20th, 40th, 60th, and 80th percentiles
    percentile_values = np.percentile(np.arange(0,batches + 1), [20,40,60,80]).astype(int)

    for i, data in enumerate(val_set):  # inner loop within one epoch

        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()   # calculate loss functions, get gradients, update network weights
        visuals = model.get_current_visuals()
        real_A = visuals['real_A']
        real_B = visuals['real_B']
        fake_B = visuals['fake_B']

        if opt.include_mask_val:
            mask_a = visuals['mask_A']
            mask_b = visuals['mask_B']
            mask_seg_a = visuals['seg_A']
            mask_seg_b = visuals['seg_B']

        # Get metrics comparing real A with real B
        mae_base += metric_mae(real_A.cpu(), real_B.cpu())
        mse_base += metric_mse(real_A.cpu(), real_B.cpu())
        ssim_base += metric_ssim(real_A.cpu(), real_B.cpu()).mean()
        psnr_base += metric_psnr(real_A.cpu(), real_B.cpu())
        # Get metrics comparing fake B with real B
        mae_fake += metric_mae(fake_B.cpu(), real_B.cpu())
        mse_fake += metric_mse(fake_B.cpu(), real_B.cpu())
        ssim_fake += metric_ssim(fake_B.cpu(), real_B.cpu()).mean()
        psnr_fake += metric_psnr(fake_B.cpu(), real_B.cpu())

        # Create visualizations
        if not opt.wdb_disabled: 
            if i == percentile_values[0]: 
                imgA_wb, imgB_wb, fakeB_wb = val_visualizations_over_batches(real_A,real_B,fake_B)
                if opt.include_mask_val:
                    maskA_wb, maskB_wb, segA_wb, segB_wb = val_mask_visualizations_over_batches(mask_a,mask_b,mask_seg_a,mask_seg_b)
            elif i == percentile_values[1] or i == percentile_values[2]:
                imgA2_wb, imgB2_wb, fakeB2_wb = val_visualizations_over_batches(real_A,real_B,fake_B)
                imgA_wb = np.concatenate((imgA_wb, imgA2_wb), axis=2)
                imgB_wb = np.concatenate((imgB_wb, imgB2_wb), axis=2)
                fakeB_wb = np.concatenate((fakeB_wb, fakeB2_wb), axis=2)
                if opt.include_mask_val:
                    maskA2_wb, maskB2_wb, segA2_wb, segB2_wb = val_mask_visualizations_over_batches(mask_a,mask_b,mask_seg_a,mask_seg_b)
                    maskA_wb = np.concatenate((maskA_wb, maskA2_wb), axis=2)
                    maskB_wb = np.concatenate((maskB_wb, maskB2_wb), axis=2)
                    segA_wb = np.concatenate((segA_wb, segA2_wb), axis=2)
                    segB_wb = np.concatenate((segB_wb, segB2_wb), axis=2)
            elif i == percentile_values[3]:
                imgA2_wb, imgB2_wb, fakeB2_wb = val_visualizations_over_batches(real_A,real_B,fake_B)
                imgA_wb = np.concatenate((imgA_wb, imgA2_wb), axis=2)
                imgA_wb = ((imgA_wb + 1) * 127.5).astype(np.uint8)
                imgA_wb = PIL.Image.fromarray(np.squeeze(imgA_wb))
                imgA_wb = imgA_wb.convert("L")
                imgB_wb = np.concatenate((imgB_wb, imgB2_wb), axis=2)
                imgB_wb = ((imgB_wb + 1) * 127.5).astype(np.uint8)
                imgB_wb = PIL.Image.fromarray(np.squeeze(imgB_wb))
                imgB_wb = imgB_wb.convert("L")
                fakeB_wb = np.concatenate((fakeB_wb, fakeB2_wb), axis=2)
                fakeB_wb = ((fakeB_wb + 1) * 127.5).astype(np.uint8)
                fakeB_wb = PIL.Image.fromarray(np.squeeze(fakeB_wb))
                fakeB_wb = fakeB_wb.convert("L")
                if opt.include_mask_val:
                    maskA2_wb, maskB2_wb, segA2_wb, segB2_wb = val_mask_visualizations_over_batches(mask_a,mask_b,mask_seg_a,mask_seg_b)   
                    maskA_wb = np.concatenate((maskA_wb, maskA2_wb), axis=2)
                    maskA_wb = ((maskA_wb + 1) * 127.5).astype(np.uint8)
                    maskA_wb = PIL.Image.fromarray(np.squeeze(maskA_wb))
                    maskA_wb = maskA_wb.convert("L")
                    maskB_wb = np.concatenate((maskB_wb, maskB2_wb), axis=2)
                    maskB_wb = ((maskB_wb + 1) * 127.5).astype(np.uint8)
                    maskB_wb = PIL.Image.fromarray(np.squeeze(maskB_wb))
                    maskB_wb = maskB_wb.convert("L")
                    segA_wb = np.concatenate((segA_wb, segA2_wb), axis=2)
                    segA_wb = ((segA_wb + 1) * 127.5).astype(np.uint8)
                    segA_wb = PIL.Image.fromarray(np.squeeze(segA_wb))
                    segA_wb = segA_wb.convert("L")
                    segB_wb = np.concatenate((segB_wb, segB2_wb), axis=2)
                    segB_wb = ((segB_wb + 1) * 127.5).astype(np.uint8)
                    segB_wb = PIL.Image.fromarray(np.squeeze(segB_wb))
                    segB_wb = segB_wb.convert("L")

    # Send data to Wandb
    if not opt.wdb_disabled: 
        wandb.log({"val/examples": [wandb.Image(imgA_wb, caption="realA"),wandb.Image(imgB_wb, caption="realB"),wandb.Image(fakeB_wb, caption="fakeB")]})                                     
        if opt.include_mask_val:
            wandb.log({"val/examples_masks": [wandb.Image(maskA_wb, caption="maskA"),wandb.Image(maskB_wb, caption="maskB"),wandb.Image(segA_wb, caption="segA"),wandb.Image(segB_wb, caption="segB")]})                                     

    # Return metrics for comparison B to A and B to B_hat                             
    return (mae_base/batches).cpu().numpy(), (mse_base/batches).cpu().numpy(), (ssim_base/batches).cpu().numpy(), (psnr_base/batches).cpu().numpy(), \
           (mae_fake/batches).cpu().numpy(), (mse_fake/batches).cpu().numpy(), (ssim_fake/batches).cpu().numpy(), (psnr_fake/batches).cpu().numpy()
