import numpy as np
import torch.nn as nn
import torch
import random
from utils.losses import mask_DiceLoss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch.nn.functional as F

CE = nn.CrossEntropyLoss(reduction='none')

def features_discrepancy_loss(features_A, features_B):
    cos_dis = nn.CosineSimilarity(dim=1, eps=1e-6)
    # normalize different latent features
    features_A = F.normalize(features_A, dim=1)
    features_B = F.normalize(features_B, dim=1)
    loss = 1 + cos_dis(features_A, features_B.detach()).mean()
    return loss

def supervison_loss(outputs, label, weight_mask=None, class_num=2):
    DICE = mask_DiceLoss(nclass=class_num)
    label = label.type(torch.int64)
    if weight_mask is None:
        loss_ce = F.cross_entropy(outputs, label)
        loss_dice = DICE(outputs, label)
    else:
        loss_ce = (CE(outputs, label) * weight_mask).sum() / (weight_mask.sum() + 1e-16)
        loss_dice = DICE(outputs, label, weight_mask)
    return (loss_ce + loss_dice) / 2

def generate_mask_3D(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, img_x - patch_pixel_x)
    h = np.random.randint(0, img_y - patch_pixel_y)
    z = np.random.randint(0, img_z - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long()

def generate_mask_2D(img, mask_ratio):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*mask_ratio), int(img_y*mask_ratio)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

