import argparse
import os
import numpy as np
import scipy.misc
from skimage.morphology import skeletonize_3d
from scipy import ndimage as ndi
from dvn.utils import get_itk_array, make_itk_image, write_itk_image, get_itk_image
from batchgenerators.utilities.file_and_folder_operations import *

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Centerlines, Bifurcations and Radius from binary vessel segmentation')
    parser.add_argument('--output', dest='output', type=str, default='/home/xdyang/vesselsegmentation/3dunet_results/from_scratch_test6_last/prediction/other/quantify',
                   help='output folder for storing quantitative results')
    parser.add_argument('--no-c', dest='save_centerlines', action='store_false',
                   help='Do not save centerline extraction')
    parser.add_argument('--o_cens', dest='suffix_cens', type=str, default='_cens',
                   help='filename suffix for renaming CENTERLINE output files (default: _cens)')
    parser.add_argument('--no-b', dest='save_bifurcations', action='store_false',
                   help='Do not save bifurcation detection')
    parser.add_argument('--o_bifs', dest='suffix_bifs', type=str, default='_bifs',
                   help='filename suffix for renaming BIFURCATION output files (default: _bifs)')
    parser.add_argument('--no-r', dest='save_radius', action='store_false',
                   help='Do not save radius estimates')
    parser.add_argument('--o_rads', dest='suffix_rads', type=str, default='_rads',
                   help='filename suffix for renaming RADIUS output files (default: _rads)')
    parser.add_argument('--f', dest='format', type=str, default='.nii.gz',
                   help='NIFTI file format for saving outputs (default: .nii.gz)')
    args = parser.parse_args()

    return args

def extract_centerlines(segmentation):
    skeleton = skeletonize_3d(segmentation)
    skeleton.astype(dtype='uint8', copy=False)
    print ('Skeleton length:........', calc_length(skeleton), ' pixel')
    return skeleton


def calc_length(img):
    image = img
    image.astype(dtype='uint8', copy=False)
    sum_v = np.sum(image)
    return sum_v


def extract_bifurcations(centerlines):
    a = centerlines
    a.astype(dtype='uint8', copy=False)
    sh = np.shape(a)
    bifurcations = np.zeros(sh,dtype='uint8')
    endpoints = np.zeros(sh,dtype='uint8')

    for x in range(1,sh[0]-1):
        for y in range(1,sh[1]-1):
            for z in range(1,sh[2]-1):
                if a[x,y,z]== 1:
                    local = np.sum([a[ x-1,  y-1,  z-1],
                    a[ x-1,  y-1,  z],
                    a[ x-1,  y-1,  z+1],
                    a[ x-1,  y,  z-1],
                    a[ x-1,  y,  z],
                    a[ x-1,  y,  z+1],
                    a[ x-1,  y+1,  z-1],
                    a[ x-1,  y+1,  z],
                    a[ x-1,  y+1,  z+1],
                    a[ x,  y-1,  z-1],
                    a[ x,  y-1,  z],
                    a[ x,  y-1,  z+1],
                    a[ x,  y,  z-1],
                    a[ x,  y,  z],
                    a[ x,  y,  z+1],
                    a[ x,  y+1,  z-1],
                    a[ x,  y+1,  z],
                    a[ x,  y+1,  z+1],
                    a[ x+1,  y-1,  z-1],
                    a[ x+1,  y-1,  z],
                    a[ x+1,  y-1,  z+1],
                    a[ x+1,  y,  z-1],
                    a[ x+1,  y,  z],
                    a[ x+1,  y,  z+1],
                    a[ x+1,  y+1,  z-1],
                    a[ x+1,  y+1,  z],
                    a[ x+1,  y+1,  z+1]])

                    if local > 3:
                        bifurcations[x,y,z] = 1

    bifurcations.astype(dtype='uint8', copy=False)
    endpoints.astype(dtype='uint8', copy=False)
    print ('# of Bifurcations:..... ', calc_length(bifurcations))
    return bifurcations, calc_length(bifurcations),endpoints

def extract_radius(segmentation, centerlines,spacing):
    image = segmentation
    skeleton = centerlines
    transf = ndi.distance_transform_edt(image,sampling=spacing,return_indices=False)
    radius_matrix = transf*skeleton
    av_rad = np.true_divide(radius_matrix.sum(),(radius_matrix!=0).sum())
    print ('Maximum radius:........ ', np.max(radius_matrix), 'um')  # adding the information of voxel spacing
    print ('Mean radius:........    ', av_rad, 'um')
    return radius_matrix,np.max(radius_matrix),av_rad

def preprocess_data(data,fn,prefix):
    data = data.astype(np.int)
    data = ndi.binary_closing(data, iterations=3).astype(np.int)
    data = np.asarray(ndi.binary_fill_holes(data), dtype='uint8')
    return data

def save_data(data, img, filename):
    out_img = make_itk_image(data, img)
    write_itk_image(out_img, filename)
    
def save_image_slice(data,filename,slice_num,project=False,cmax=1.0):
    data = np.asarray(data, dtype='float32')
    if project:
        scipy.misc.toimage(sum(data[slice_num-5:slice_num+4,:,:]), cmin=0.0, cmax=cmax).save(filename)
    else:
        scipy.misc.toimage(data[slice_num,:,:], cmin=0.0, cmax=cmax).save(filename)
    
def run():
    args = parse_args()
    # filenames = args.filenames
    outputFn = args.output
    save_cen = args.save_centerlines
    save_bif = args.save_bifurcations
    save_rad = args.save_radius
    cen_suffix = args.suffix_cens
    bif_suffix = args.suffix_bifs
    rad_suffix = args.suffix_rads
    fmt = args.format
    source_prediction_folder = "/home/xdyang/vesselsegmentation/3dunet_results/from_scratch_test6_last/prediction/other/denoise"
    prediction_cases = subfiles(join(source_prediction_folder, 'miss1'), suffix=".nii.gz", join=False)
    print('----------------------------------------')
    print(' Feature Extraction Parameters ')
    print('----------------------------------------')
    # print ('Input files:', filenames)
    # print ('Output folder:', outputFn)
    print ('Output format:', fmt)
    print ('Centerline file suffix:', cen_suffix)
    print ('Bifurcation file suffix:', bif_suffix)
    print ('Radius file suffix:', rad_suffix)
    print ('Save centerline extractions:', save_cen)
    print ('Save bifurcation detections:', save_bif)
    print ('Save radius estimates:', save_rad)
    print ('----------------------------------------')

    for fn in prediction_cases:
        # fn = join(source_prediction_folder, 'miss1', fn)
        print ('predicting features for :', fn)
        cen = None
        bif = None
        rad = None
        prefix = os.path.basename(fn).split('.')[0]
        data = get_itk_array(fn)
        print(np.shape(data))
        img = get_itk_image(fn)
        # assign voxel information
        if "#5-CD31-6" in fn:
            spacing  = [1.14, 1.14,0.799]
            print("#5-CD31-6")
        elif "-10x" in fn:
            spacing= [1.14, 1.14,0.799]
            print("x10")
        elif "539" in fn:
            spacing = [0.446, 0.446,4]
            print("539")
        elif "-40x" in fn:
            spacing = [0.285, 0.285,0.799]
            print("-40x")
        elif "#5-CD31-1" in fn:
            spacing = [0.568, 0.568,0.685]
            print("#5-CD31-1done")
        else:
            spacing= [0.568, 0.568,0.799]
            print("other")
        if save_rad:
            cen = extract_centerlines(segmentation=data)
            rad,max_rad,mean_rad = extract_radius(segmentation=data, centerlines=cen,spacing=spacing)

        if save_bif:
            if cen is None:
                cen = extract_centerlines(segmentation=data)

            bif,no_bif,_ = extract_bifurcations(centerlines=cen)
            ofn = os.path.join(outputFn, prefix + bif_suffix + fmt)
            save_data(data=np.asarray(bif, dtype='uint8'), img=img, filename=ofn)

        if save_cen:
            if cen is None:
                cen = extract_centerlines(segmentation=data)

            ofn = os.path.join(outputFn, prefix + cen_suffix + fmt)
            save_data(data=np.asarray(cen, dtype='uint8'), img=img, filename=ofn)
            # save_image_slice(cen,'../feat_output_test/centerlines_slice_25.png',25,project=True)
        f = open(outputFn + '/' + 'metric.txt', 'a')
        f.write(prefix +':  '+'max_rad:'+ str(max_rad) +'  '+'mean_rad:'+str(mean_rad)+'  '+'no of bif:'+str(no_bif)+'\n')
        f.close()
    print('finished!')

if __name__ == '__main__':
    run()
