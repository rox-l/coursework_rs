#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:26:01 2021

@author: roxannelai
"""

# Import libraries
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Path to directories
input_path = "/Users/roxannelai/Desktop/RemoteSensingAnalysis_A1/Report1/input/"
output_path = "/Users/roxannelai/Desktop/RemoteSensingAnalysis_A1/Report1/output/"
#%%
# read in images using imread()
# -1 is for uint16

bands = ['lr_red', 'lr_green', 'lr_blue', 'lr_nir']
resized_list = []

pan = cv2.imread(input_path+'pan.tif', -1)
h, w = pan.shape

# resize images
for b in bands: 
    path = input_path + b + '.tif'
    img = cv2.imread(path, -1)
    resized_list.append(np.uint16(cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)))
    
resized_list.append(pan)
print(len(resized_list))

#%% Weighted Brovey function

def brovey_sharpen(bandlist, W=0.25):
    """ 
    Function to create pan-sharpened image using the weight Brovey method.
    
    Args: 
        bandist (list): List of image bands in order: RED, GREEN, BLUE, NIR, pan, where: 
            RED (single band raster): red band image
            GREEN (single band raster): green band image
            BLUE (single band raster): blue band image
            NIR (single band raster): nir band image
            pan (single band raster): panchromatic image
        W (float): weights for Browley equation 
    
    Returns: 
        lpansharpened_image: rgb weight brovey pansharpened image
    
    """
    RED, GREEN, BLUE, NIR, pan = bandlist
    DNF = (pan - W * NIR)/(W * RED + W * GREEN + W * BLUE)
    brovey_applied_list = [np.uint16((band * DNF)) for band in bandlist]
    brovey_red, brovey_green, brovey_blue = brovey_applied_list[:3]
    pansharpened_image = cv2.merge((brovey_red, brovey_green, brovey_blue))
    return pansharpened_image

pansharpened_img = brovey_sharpen(resized_list)
# print(pansharpened_img.shape)

#%%
cv2.imwrite(output_path+ 'pan2test.tif', pansharpened_img)

