import cv2
import numpy as np
from matplotlib import pyplot as plt
from dvn.utils import get_itk_array, get_itk_image, make_itk_image, write_itk_image
import os as os
import argparse
from cv2_rolling_ball import subtract_background_rolling_ball
import tensorflow as tf
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk

print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


source_prediction_folder = "/public/yangxiaodu/vesselsegmentation/data/data_raw"
output_folder="/public/yangxiaodu/vesselsegmentation/data/data_pre/"
train_cases = subfiles(source_prediction_folder, suffix=".nii.gz", join=False)
for t in train_cases:
    img_file = join(source_prediction_folder, t)
    img = get_itk_array(img_file)
    imgCLAHE = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    roll_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    background = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    img = img.astype(np.uint8)

    constrast = {}
    brightness = {}
    prefix = os.path.basename(img_file).split('.')[0]
    print(prefix)
    for i in range(img.shape[0]):
        constrast[i] = img[i, :, :].std()
        brightness[i] = img[i, :, :].mean()
        # print(prefix, constrast, brightness)
    constrast = list(constrast.values())
    brightness = list(brightness.values())
    if max(constrast) <= 12 and max(brightness) <= 10:  #only process the volumes with weaker vessels
        print(max(constrast), max(brightness))
        print("This volume need histogram equalization.")
        Flag=True
    else:
        Flag=False

    for i in range(img.shape[0]):
        if Flag:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            imgCLAHE[i, :, :] = clahe.apply(img[i, :, :])
        else:
            imgCLAHE[i,:,:] = img[i,:,:]
        roll_img[i,:,:], background[i,:,:] = subtract_background_rolling_ball(imgCLAHE[i,:,:], 50, light_background=False, use_paraboloid=False,
                                                       do_presmooth=True)
    roll_img = roll_img.astype(np.float32)
    X4_image = make_itk_image(roll_img)
    nii_filename_clip4= output_folder+prefix+ '.nii.gz'
    write_itk_image(X4_image, nii_filename_clip4)





