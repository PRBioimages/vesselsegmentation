import os
import glob

from process.prediction import run_validation_cases
from dvn.utils import get_itk_array, write_itk_imageArray, make_itk_image, write_itk_image
from process.data import write_data_to_file, open_data_file
import numpy as np
from process.utils.utils import pickle_dump, pickle_load

config = dict()
prediction_dir = '/home/xdyang/vesselsegmentation/3dunet_results/from_scratch_test6_last/prediction/other' # predict new data
config["model_file"] = '/home/xdyang/vesselsegmentation/3dunet_results/from_scratch_test6_last/model/all/model6.0.h5'
config["data_file"] = "/home/xdyang/vesselsegmentation/code/process/glioma_data_test.h5"
config["inputFns"]='/home/xdyang/vesselsegmentation/code/process/test_input.txt'
config["labelFns"] =None#'/home/xdyang/vesselsegmentation/code/process/test_label.txt'  # There is no label data in testing set.
config["nb_channels"] = 1
config["padding_output"] = '/home/vesselsegmentation/code/process/padding'
config["testing_file"] = os.path.abspath("/home/xdyang/vesselsegmentation/code/process/testing_ids.pkl")
config["labels"] = (1,)  # the label numbers on the input



def save_path():  # create input text files.
    import os
    rootdir = os.path.join('/home/xdyang/vesselsegmentation/data/data_pre')
    write_path = open('/home/xdyang/vesselsegmentation/code/process/test_input.txt', 'w')
    for (dirpath, dirnames, filenames) in os.walk(rootdir):
        for filename in filenames:
                write_path.write(os.path.join(rootdir,filename) + '\n')
    write_path.close()

def norm_data(data):
    data = data - np.min(data)
    data = data * 1.0 / (np.max(data) - np.min(data))
    return data

def get_image_shape(inputFn, n_in=1):

    with open(os.path.abspath(inputFn)) as f:
        iFn = f.readlines()
    iFn = [x.strip() for x in iFn]
    shape_arr0 = list()
    shape_arr1 = list()
    shape_arr2 = list()
    for ifn in iFn:
        if n_in == 1:
            X = norm_data(get_itk_array(ifn))
        X = np.array(X).astype('float32')
        shape_arr0.append(X.shape[0])  # 127
        shape_arr1.append(X.shape[1])  # 512
        shape_arr2.append(X.shape[2])  # 512
    image_shape = (max(shape_arr2), max(shape_arr1), max(shape_arr0))
    print(image_shape)
    return image_shape

def image_padding(inputFn, labelFn=None, output=None, n_in=1, image_shape=(512, 512, 300), flag=None):
    with open(os.path.abspath(inputFn)) as f:
        iFn = f.readlines()
    iFn = [x.strip() for x in iFn]
    if labelFn is not None:
        with open(os.path.abspath(labelFn)) as f:
            lFn = f.readlines()
        lFn = [x.strip() for x in lFn]
    testing_data_files = list()
    oshape = []
    name=[]
    if labelFn is not None:
        for ifn, lfn in zip(iFn, lFn):
            prefix = os.path.basename(ifn).split('.')[0]
            prefix1 = os.path.basename(lfn).split('.')[0]
            if n_in == 1:
                X = norm_data(get_itk_array(ifn))
            X = np.array(X).astype('float32')
            Y = norm_data(get_itk_array(lfn))
            Y = np.array(Y).astype('float32')
            oshape.append(X.shape)
            name.append(prefix1)
            if os.path.exists(output + '/' + 'test/data/' + prefix + '.nii.gz'):
                subject_files = [str(output + '/' + 'test/data/' + prefix + '.nii.gz'),
                                 str(output + '/' + 'test/label/' + prefix1 + '.nii.gz')]
            else:
                dummy_data = np.zeros((image_shape[2], image_shape[1], image_shape[0]), dtype=X.dtype)
                dummy_data[:X.shape[0], :X.shape[1], :X.shape[2]] = X
                dummy_label = np.zeros((image_shape[2], image_shape[1], image_shape[0]), dtype=Y.dtype)
                dummy_label[:Y.shape[0], :Y.shape[1], :Y.shape[2]] = Y
                index = []
                for ind in range(dummy_label.shape[0]):
                    if np.all(dummy_label[ind, :, :] == 0) == True:
                        dummy_label[ind, :, :][:] = 2
                        index.append(ind)
                print(index)
                data = make_itk_image(dummy_data)
                label = make_itk_image(dummy_label)
                if flag == 'test':
                    write_itk_image(data, output + '/' + 'test/data/' + prefix + '.nii.gz')
                    write_itk_image(label, output + '/' + 'test/label/' + prefix1 + '.nii.gz')
                    subject_files = [str(output + '/' + 'test/data/' + prefix + '.nii.gz'),
                                     str(output + '/' + 'test/label/' + prefix1 + '.nii.gz')]
                elif flag == 'val':
                    write_itk_image(data, output + '/' + 'val/data/' + prefix + '.nii.gz')
                    write_itk_image(label, output + '/' + 'val/label/' + prefix1 + '.nii.gz')
                    subject_files = [str(output + '/' + 'val/data/' + prefix + '.nii.gz'),
                                     str(output + '/' + 'val/label/' + prefix1 + '.nii.gz')]
            testing_data_files.append(tuple(subject_files))
    else:
        for ifn in iFn:
            prefix = os.path.basename(ifn).split('.')[0]
            if n_in == 1:
                X = norm_data(get_itk_array(ifn))
            X = np.array(X).astype('float32')
            oshape.append(X.shape)
            name.append(prefix)
            if os.path.exists(output + '/' + 'test/data/' + prefix + '.nii.gz'):
                subject_files = [str(output + '/' + 'test/data/' + prefix + '.nii.gz'),None]
            else:
                dummy_data = np.zeros((image_shape[2], image_shape[1], image_shape[0]),
                                      dtype=X.dtype)
                dummy_data[:X.shape[0], :X.shape[1], :X.shape[2]] = X

                data = make_itk_image(dummy_data)

                if flag == 'test':
                    write_itk_image(data, output + '/' + 'test/data/' + prefix + '.nii.gz')
                    subject_files = [str(output + '/' + 'test/data/' + prefix + '.nii.gz'),None]
            testing_data_files.append(tuple(subject_files))
    return testing_data_files,oshape,name

def main(overwrite=True):

    if overwrite or not os.path.exists(config["data_file"]):
        config["image_shape"] = get_image_shape(config["inputFns"], n_in=config["nb_channels"])
        testing_data_files,oshape,name = image_padding(config["inputFns"], config["labelFns"], config["padding_output"],
                                            n_in=config["nb_channels"],image_shape=config["image_shape"], flag='test')
        write_data_to_file(testing_data_files, config["data_file"], image_shape=config["image_shape"],truth_dtype=np.float32,
                           normalize=False)
    data_file_opened = open_data_file(config["data_file"])
    nb_samples = data_file_opened.root.data.shape[0]
    pickle_dump(list(range(nb_samples)), config["testing_file"])
    run_validation_cases(training_keys_file=config["testing_file"],
                         model_file=config["model_file"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir,origin_shape=oshape,name=name,overlap=16,permute=True)


if __name__ == "__main__":
    save_path()
    main()
