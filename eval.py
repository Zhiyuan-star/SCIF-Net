import numpy as np
import torch
from tqdm import tqdm
from dice_loss import metrics, metrics_3d
import os


def eval_net(net, loader, bach_size, device, dataset, scal):
    net.eval()
    n_val = len(loader)  # the number of batch
    val_num = len(loader) * bach_size  # the number of val_data

    f1, acc, sen, auc, dice, hd95 = 0, 0, 0, 0, 0, 0
    with tqdm(total=val_num, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['masks_2st']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(imgs)

            f, a, s, au, d, h = metrics(mask_pred, true_masks, batch['image'].shape[0], dataset, scal)
            f1 += f
            acc += a
            sen += s
            auc += au
            dice += d
            hd95 += h
            pbar.update(imgs.shape[0])
    return f1 / max(n_val, 1), acc / max(n_val, 1), sen / max(n_val, 1), auc / max(n_val, 1), dice / max(n_val,
                                                                                                         1), hd95 / max(
        n_val, 1)


def eval_3d(net, device):
    dice = 0.
    hd95 = 0.
    nii_file_path = './data/Knee/test/images/'
    mask_file_path = './data/Knee/test/masks/'
    file_list = os.listdir(nii_file_path)
    n_val = len(file_list)
    with tqdm(total=n_val, desc='Validation round', unit='file', leave=False) as pbar:
        for file in file_list:
            mask_file = file.split('.')[0] + "_seg.nii"
            d, h = metrics_3d(net, nii_file_path + file, mask_file_path + mask_file, device)
            dice += d
            hd95 += h
            pbar.update(1)
    return dice / max(n_val, 1), hd95 / max(n_val, 1)
