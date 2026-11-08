#!/usr/bin/env python3
# coding:utf-8

# **********************************************************
# * Author        : danylgao (Bin-Bin Gao)
# * Email         : danylgao@tencent.com
# * Create time   : 2022-11-25 20:18
# * Last modified : 2022-11-25 20:18
# * Filename      : cutpaste.py
# * Description   : borrowed from https://github.com/LilitYolyan/CutPaste/blob/main/cutpaste.py
# **********************************************************

import glob
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .perlin import rand_perlin_2d_np


class CutPaste(object):

    def __init__(self, transform = True, type = 'binary'):

        '''
        This class creates to different augmentation CutPaste and CutPaste-Scar. Moreover, it returns augmented images
        for binary and 3 way classification
        :arg
        :transform[binary]: - if True use Color Jitter augmentations for patches
        :type[str]: options ['binary' or '3way'] - classification type
        '''
        self.type = type
        if transform:
            self.transform = transforms.ColorJitter(brightness = 0.4,
                                                      contrast = 0.4,
                                                      saturation = 0.4,
                                                      hue = 0.1)
        else:
            self.transform = None

    @staticmethod
    def crop_and_paste_patch(image, mask, patch_w, patch_h, transform, rotation=False):
        """
        Crop patch from original image and paste it randomly on the same image.
        :image: [PIL] _ original image
        :patch_w: [int] _ width of the patch
        :patch_h: [int] _ height of the patch
        :transform: [binary] _ if True use Color Jitter augmentation
        :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range
        :return: augmented image
        """

        org_w, org_h = image.size
        patch_mask = None
        aug_mask = mask.copy()

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
        if transform:
            patch = transform(patch)
            patch_mask =  Image.new("L", patch.size, 255)
        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
            patch_mask = patch.split()[-1]
             
        # new location
        paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        aug_image = image.copy()
        
        aug_image.paste(patch, (paste_left, paste_top), mask=patch_mask)
        aug_mask.paste(patch_mask, (paste_left, paste_top), mask=patch_mask)
         
        return aug_image, aug_mask

    def cutpaste(self, image, mask, area_ratio = (0.02, 0.15), aspect_ratio = ((0.3, 1) , (1, 3.3))):
        '''
        CutPaste augmentation
        :image: [PIL] - original image
        :area_ratio: [tuple] - range for area ratio for patch
        :aspect_ratio: [tuple] -  range for aspect ratio
        :return: PIL image after CutPaste transformation
        '''

        img_area = image.size[0] * image.size[1]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w  = int(np.sqrt(patch_area*patch_aspect))
        patch_h = int(np.sqrt(patch_area/patch_aspect))
        cutpaste, cpmask = self.crop_and_paste_patch(image, mask, patch_w, patch_h, self.transform, rotation = False)
        return cutpaste, cpmask

    def cutpaste_scar(self, image, mask, width = [2, 16], length = [10, 25], rotation = (-45, 45)):
        '''
        :image: [PIL] - original image
        :width: [list] - range for width of patch
        :length: [list] - range for length of patch
        :rotation: [tuple] - range for rotation
        :return: PIL image after CutPaste-Scare transformation
        '''
        patch_w, patch_h = random.randint(*width), random.randint(*length)
        cutpaste_scar, cutpaste_scar_mask = self.crop_and_paste_patch(image, mask, patch_w, patch_h, self.transform, rotation = rotation)
        return cutpaste_scar, cutpaste_scar_mask 

    def __call__(self, image, mask):
        '''
        :image: [PIL] - original image
        :return: if type == 'binary' returns original image and randomly chosen transformation, else it returns
                original image, an image after CutPaste transformation and an image after CutPaste-Scar transformation
        '''
        if self.type == 'binary':
            aug = random.choice([self.cutpaste, self.cutpaste_scar])
            return aug(image, mask)

        elif self.type == 'cutpaste':
            return self.cutpaste(image, mask) 
        elif self.type == 'cutpaste_scar':
            return self.cutpaste_scar(image, mask) 
           
class PerlinPaste(object):
    def __init__(self, paste_data_root, paste_img_paths):
        '''
        '''
        import imgaug.augmenters as iaa
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                      iaa.Solarize(0.5, threshold=(32, 128)),
                      iaa.Posterize(),
                      iaa.Fliplr(0.5),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      ]
        self.perlin_scale = 6
        self.min_perlin_scale = 1
        self.paste_data_root = paste_data_root
        self.anomaly_paths = paste_img_paths 
        # sorted(glob.glob(dtd_dir + "/images" + "/*/*.jpg"))
        
    def perlin_and_paste(self, normal_image, normal_mask=None):
        """
        """
        import imgaug.augmenters as iaa
        h, w = normal_image.shape[1], normal_image.shape[0] 
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug_trans = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                            )
        
        anomaly_idx = torch.randint(0, len(self.anomaly_paths), (1,)).item()
        paste_img_path = os.path.join(self.paste_data_root, self.anomaly_paths[anomaly_idx]['img_path'])
      
        anomaly_image = cv2.imread(paste_img_path)
        anomaly_image = cv2.resize(anomaly_image, dsize=(h, w))
        
        aug_anomaly_image = aug_trans(image=anomaly_image)
        
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((h, w), (perlin_scalex, perlin_scaley))
        rot = iaa.Sequential([iaa.Affine(rotate=(-5, 5))])
        perlin_noise = rot(image=perlin_noise)

        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = aug_anomaly_image.astype(np.float32) * perlin_thr  # 255.0
       
        beta = torch.rand(1).numpy()[0] * 0.8
       
        aug_image = normal_image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * normal_image * (perlin_thr)
        aug_mask = normal_mask.copy()
        aug_mask = np.maximum(aug_mask, np.uint8(perlin_thr[:, :, 0]*255))

        return aug_image, aug_mask
    
    def __call__(self, normal_image, normal_mask=None):
        '''
        '''
        return self.perlin_and_paste(normal_image, normal_mask) 
          
