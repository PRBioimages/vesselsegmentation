import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import scipy.misc
from skimage import morphology
from skimage.morphology import skeletonize_3d
from scipy import ndimage as ndi
from dvn.utils import get_itk_array, make_itk_image, write_itk_image, get_itk_image
from skimage.io import imread
import SimpleITK as sitk
from dvn.metrics import metric_dice, f1_socre, accuracy_bin, sensitivity, specificity, precision
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
from skimage.morphology import disk, rectangle, binary_dilation, binary_erosion, binary_closing, binary_opening, \
    rectangle, remove_small_objects
import networkx as nx
from scipy.spatial import distance

import sys
from keras import backend as K

def save_path():
    import os
    rootdir = os.path.join('/home/xdyang/vesselsegmentation/3dunet_results/from_scratch_test6_last/prediction/other')
    write_path = open('/home/xdyang/vesselsegmentation/code/process/post-process.txt', 'w')
    for (dirpath, dirnames, filenames) in os.walk(rootdir):
        for filename in filenames:
                write_path.write(os.path.join(rootdir,filename) + '\n')
    write_path.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Postprocessing binary vessel segmentation')
    parser.add_argument('--filenames', dest='filenames', type=str,default='/home/xdyang/vesselsegmentation/code/process/post-process.txt')
    parser.add_argument('--maskFilename', dest='maskFn', type=str,
                        default=None,#'/home/xdyang/vesselsegmentation/code/process/test_label.txt',
                        help='a mask file applied to evaluate results')
    parser.add_argument('--output1', dest='output1', type=str,
                        default='/home/xdyang/vesselsegmentation/3dunet_results/from_scratch_test6_last/prediction/other/denoise',
                        help='output folder for storing denoising results')
    parser.add_argument('--output2', dest='output2', type=str,
                        default='/home/xdyang/vesselsegmentation/3dunet_results/from_scratch_test6_last/prediction/other/close',
                        help='output folder for storing closing results')
    parser.add_argument('--f', dest='format', type=str, default='.nii.gz',
                        help='NIFTI file format for saving outputs (default: .nii.gz)')
    args = parser.parse_args()

    return args

def norm_data_255(data):
    data = data - np.min(data)
    data = data * 255.0 / (np.max(data)-np.min(data))
    return data

def preprocess_data(data, outputFn, ifn,m,mfn=[],ifsave=True):
    prefix = os.path.basename(ifn).split('.')[0]
    data = data.astype(np.int)
    pad = np.pad(data, (m, m))
    data = ndi.binary_closing(pad, iterations=m).astype(np.uint8)
    data = data[m:-m, m: -m, m: -m]
    if ifsave:
        ofn1 = os.path.join(outputFn +'/'+str(m)+ '/' + prefix + 'closing.nii.gz')
        save_data(data=data, img=get_itk_image(ifn),filename=ofn1)
    data = np.asarray(ndi.binary_fill_holes(data), dtype='uint8')
    if ifsave:
        ofn2=os.path.join(outputFn +'/'+str(m)+ '/' + prefix + 'filling.nii.gz')
        save_data(data=data, img=None,filename=ofn2)

    if mfn != []:
        mask = get_itk_array(mfn)
        index = []
        for ind in range(mask.shape[0]):
            if np.any(mask[ind, :, :]) == True:
                index.append(ind)
        data1 = data[index, :, :]
        mask1 = mask[index, :, :]
        dice1 = round(metric_dice(mask1, data1), 4)
        acc_voxel1 = round(accuracy_bin(mask1, data1), 4)
        sensitivity2 = round(sensitivity(mask1, data1), 4)
        specificity2 = round(specificity(mask1, data1), 4)
        precision2 = round(precision(mask1, data1), 4)
        name = prefix
    else:
        dice1 = acc_voxel1 = sensitivity2 = specificity2 = precision2 = 0
        name = prefix

    return data, dice1, acc_voxel1, sensitivity2, specificity2, precision2, name

def closing(data, n):
    data = data.astype(np.int)
    data = ndi.binary_closing(data, iterations=n).astype(np.int)
    return data


def openning(data, n):
    data = data.astype(np.int)
    data = ndi.binary_opening(data, iterations=n).astype(np.int)
    return data


def fill_holes(data):
    data = np.asarray(ndi.binary_fill_holes(data), dtype='uint8')
    return data


def save_data(data, img, filename):
    out_img = make_itk_image(data, img)
    write_itk_image(out_img, filename)


def get_spacing(fn):
    # img = imread(fn)
    img = nib.load(os.path.abspath(fn))
    # img_affine = image.affine
    img = img.get_data()
    img_itk = sitk.GetImageFromArray(img.astype(np.float32))
    spacing = np.array(img_itk.GetSpacing())
    return spacing

def skeleton_demo(image,thr):
    binary = image>thr
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    return  skeleton

def binary_and_show(image,thr):
    binary = image>thr
    show_picture = binary.astype(np.uint8) * 255
    return show_picture


def calc_length(img):
    image = img
    image.astype(dtype='uint8', copy=False)
    sum_v = np.sum(image)
    return sum_v


def denoise(data,outputFn3,ifn,n,mfn=[],ifsave=True):
    # for n in range(100,500,10):
    spacing = get_spacing(ifn)
    label_img, num = label(data, connectivity=data.ndim, return_num=True)
    region = regionprops(label_img)
    noise_patch = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=data.dtype)
    for o in range(len(region)):
        object_list_coords = region[o].coords
        pixel_num = region[o].area
        object_volume = spacing[0] * spacing[1] * spacing[2] * pixel_num
        if object_volume <= n:
            for v in range(len(object_list_coords)):
                noise_patch[object_list_coords[v][0], object_list_coords[v][1], object_list_coords[v][2]] = 1
                data[object_list_coords[v][0], object_list_coords[v][1], object_list_coords[v][2]] = 0

    prefix = os.path.basename(ifn).split('.')[0]
    if ifsave:
        ofn2 = os.path.join(outputFn3 +'/'+str(n)+ '/' + prefix + 'noise_patch.nii.gz')
        save_data(data=noise_patch, img=get_itk_image(ifn), filename=ofn2)
        ofn2 = os.path.join(outputFn3 +'/'+str(n)+ '/' + prefix + 'data_disnoise.nii.gz')
        save_data(data=data, img=get_itk_image(ifn), filename=ofn2)

    if mfn != []:
        mask = get_itk_array(mfn)
        index = []
        for ind in range(mask.shape[0]):
            if np.any(mask[ind, :, :]) == True:
                index.append(ind)
        # print(index)
        data1 = data[index, :, :]
        mask1 = mask[index, :, :]

        dice1 = round(metric_dice(mask1, data1), 4)
        acc_voxel1 = round(accuracy_bin(mask1, data1), 4)
        sensitivity2 = round(sensitivity(mask1, data1), 4)
        specificity2 = round(specificity(mask1, data1), 4)
        precision2 = round(precision(mask1, data1), 4)
        name = prefix
    else:
        dice1=acc_voxel1=sensitivity2=specificity2=precision2=0
        name=prefix

    return data, dice1, acc_voxel1, sensitivity2, specificity2, precision2, name


def run():
    args = parse_args()
    outputFn1 = args.output1
    outputFn2 = args.output2
    fmt = args.format
    filenames = args.filenames
    masks = args.maskFn

    print('----------------------------------------')
    print(' Postprocessing Parameters ')
    print('----------------------------------------')
    print('Input files:', filenames)
    print('Mask file:', masks)
    print('Output folder:', outputFn1)
    print('Output folder:', outputFn2)
    print('Output format:', fmt)

    with open(os.path.abspath(args.filenames)) as f:
        iFn = f.readlines()
    iFn = [x.strip() for x in iFn]

    if masks is not None:
        with open(os.path.abspath(args.maskFn)) as f:
            mFn = f.readlines()
        mFn = [x.strip() for x in mFn]
    else:
        mFn=[]

    i = 0
    dice_list= {}
    acc_voxel_list = {}
    sensitivity_list = {}
    specificity_list = {}
    precision_list = {}
    name_list ={}

    for t in range(2):
        dice_list[t]= {}
        acc_voxel_list[t]= {}
        sensitivity_list[t] = {}
        specificity_list[t] = {}
        precision_list[t]= {}
        name_list[t] = {}

    v=0
    if mFn!=[]:
        # Grid search for suitable post-processing parameters
        mean_dice_c=[]
        # for n in range(10,600,10):
        n=300
        for m in range(1,30):
            i = 0
            if not os.path.exists(outputFn2 + '/' + str(m)):
                os.mkdir(outputFn2 + '/' + str(m))
            for ifn, mfn in zip(iFn, mFn):
                print('post-processing for :', ifn)
                data = get_itk_array(ifn)
                post4_result, dice_list[1][i], acc_voxel_list[1][i], sensitivity_list[1][i], specificity_list[1][i], precision_list[2][i], name_list[1][i] = preprocess_data(data, outputFn2, ifn, m, mfn, ifsave=True)
                i = i + 1
            t=1
            if t == 1:
                f = open(outputFn1 + '/' + 'metric.txt', 'a')
            if t == 2:
                f = open(outputFn2 + '/' + 'metric_test1.txt', 'a')
            f.write('name:' + str(name_list[t])  +'\n' + "dice_bins:" + str(dice_list[t]) +str(sum(dice_list[t].values())/len(dice_list[t])) +'\n' + "acc_voxel1:" + str(
                acc_voxel_list[t]) +str(sum(acc_voxel_list[t].values())/len(acc_voxel_list[t])) + "\n" + "sensitivity2:" + str(sensitivity_list[t]) +str(sum(sensitivity_list[t].values())/len(sensitivity_list[t])) +"\n" + "specificity2:" + str(
                specificity_list[t]) +str(sum(specificity_list[t].values())/len(specificity_list[t]))+ "\n" + "precision2:" + str(precision_list[t]) +str(sum(precision_list[t].values())/len(precision_list[t]))+ '\n')
            f.close()
            mean_dice_c.append(sum(dice_list[2].values())/len(dice_list[2]))
        return mean_dice_c
    else:
        m=10
        n=290
        if not os.path.exists(outputFn2 + '/' + str(m)):
            os.mkdir(outputFn2 + '/' + str(m))
        if not os.path.exists(outputFn1+ '/' + str(n)):
            os.mkdir(outputFn1 + '/' + str(n))
        for ifn in iFn:
            print('post-processing :', ifn)
            data = get_itk_array(ifn)
            post1_result, dice_list[0][i], acc_voxel_list[0][i], sensitivity_list[0][i], specificity_list[0][i], precision_list[0][i], name_list[0][i] = denoise(data, outputFn1, ifn, n,ifsave=True)
            post2_result, dice_list[1][i], acc_voxel_list[1][i], sensitivity_list[1][i], specificity_list[1][i], precision_list[1][i], name_list[1][i] = preprocess_data(post1_result, outputFn2, ifn, m, ifsave=True)
            i = i + 1



if __name__ == '__main__':
    save_path()
    mean_dice=run()

