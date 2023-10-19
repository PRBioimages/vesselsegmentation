from __future__ import print_function
import os
import glob
import tensorflow as tf
print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse

import numpy as np
from dvn.utils import get_itk_array, write_itk_imageArray, make_itk_image, write_itk_image
from dvn import losses, metrics
from dvn.losses import soft_dice
from dvn.metrics import dice
from dvn.unet3d import unet_test6
from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plt
import keras as K
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import TensorBoard
import sys

sys.path.append("/home/xdyang/vesselsegmentation")
from process.data import write_data_to_file, open_data_file
from process.generator import get_training_and_validation_generators
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

K.backend.set_image_data_format('channels_first')
print(K.backend.image_data_format())

config = dict()
# config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
# config["image_shape"] = (1024, 1024, 127)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = (192,192,48)  # switch to None to train on the whole image
config["labels"] = (1,)  # the label numbers on the input
config["n_labels"] = 1  # len(config["labels"])
config["nb_channels"] = 1
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]

config["batch_size"] = 2  # 6
config["validation_batch_size"] = 10  # 12
'''''''''
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
'''''''''
config["validation_split"] = None  # portion of the data that will be used for training 
config["flip"] = True  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = 0.25  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 5  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped
config["data_file"] = os.path.abspath("/home/xdyang/vesselsegmentation/code/process/glioma_data.h5")  # converting training data into h5 files
config["training_file"] = os.path.abspath("/home/xdyang/vesselsegmentation/code/process/training_ids.pkl") # storing the ids of training data
config["validation_file"] = os.path.abspath("/home/xdyang/vesselsegmentation/code/process/validation_ids.pkl") # storing the ids of validation data
config["inputFns"] = '/home/xdyang/vesselsegmentation/code/process/train_input.txt'
config["labelFns"] = '/home/xdyang/vesselsegmentation/code/process/train_label.txt'

config["output"] = '/home/xdyang/vesselsegmentation/code/process/padding'
config["overwrite"] = False  # If True, will previous files. If False , will use previously written files.
# run(overwrite=config["overwrite"])

def norm_data(data):
    data = data - np.min(data)
    data = data * 1.0 / (np.max(data) - np.min(data))
    return data

def train_model(model, train_data, steps_per_epoch, n_epochs, lr, weighted_cost, model_folder,initial_epoch=0):

    adam = K.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(optimizer=adam, loss=losses.soft_dice, metrics=[metrics.dice()])
    print(initial_epoch)
    print(n_epochs)
    h = model.fit_generator(generator=train_data, steps_per_epoch=steps_per_epoch, epochs=n_epochs,
                            initial_epoch=initial_epoch,use_multiprocessing=True,workers=6)
    return model, lr, h


def get_image_shape(inputFn, labelFn, n_in=1):

    with open(os.path.abspath(inputFn)) as f:
        iFn = f.readlines()
    iFn = [x.strip() for x in iFn]

    with open(os.path.abspath(labelFn)) as f:
        lFn = f.readlines()
    lFn = [x.strip() for x in lFn]
    shape_arr0 = list()
    shape_arr1 = list()
    shape_arr2 = list()
    for ifn, lfn in zip(iFn, lFn):
        if n_in == 1:
            X = norm_data(get_itk_array(ifn))
        X = np.array(X).astype('float32')
        Y = get_itk_array(lfn)
        shape_arr0.append(X.shape[0])  # 127
        shape_arr1.append(X.shape[1])  # 512
        shape_arr2.append(X.shape[2])  # 512
    image_shape = (max(shape_arr2), max(shape_arr1), max(shape_arr0))
    print(image_shape)
    return image_shape


def image_padding(inputFn, labelFn, output=None, n_in=1, image_shape=(512, 512, 300), flag=None):
    with open(os.path.abspath(inputFn)) as f:
        iFn = f.readlines()
    iFn = [x.strip() for x in iFn]

    with open(os.path.abspath(labelFn)) as f:
        lFn = f.readlines()
    lFn = [x.strip() for x in lFn]
    training_data_files = list()
    for ifn, lfn in zip(iFn, lFn):
        prefix = os.path.basename(ifn).split('.')[0]
        prefix1 = os.path.basename(lfn).split('.')[0]
        if os.path.exists(output + '/' + 'train/data/' + prefix + '.nii.gz'):
            subject_files = [str(output + '/' + 'train/data/' + prefix + '.nii.gz'),
                             str(output + '/' + 'train/label/' + prefix1 + '.nii.gz')]
        else:
            if n_in == 1:
                X = norm_data(get_itk_array(ifn))
            X = np.array(X).astype('float32')
            Y = norm_data(get_itk_array(lfn))
            Y = np.array(Y).astype('float32')
            dummy_data = np.zeros((image_shape[2], image_shape[1], image_shape[0]), dtype=X.dtype)
            dummy_data[:X.shape[0], :X.shape[1], :X.shape[2]] = X
            dummy_label = np.zeros((image_shape[2], image_shape[1], image_shape[0]), dtype=Y.dtype)
            dummy_label[:Y.shape[0], :Y.shape[1], :Y.shape[2]] = Y
            index = []
            for ind in range(dummy_label.shape[0]):
                if np.all(dummy_label[ind, :, :] == 0) == True:
                    dummy_label[ind, :, :][:] = 2  # handling of sparse annotations
                    index.append(ind)
            print(index)
            data = make_itk_image(dummy_data)
            label = make_itk_image(dummy_label)
            if flag == 'train':
                write_itk_image(data, output + '/' + 'train/data/' + prefix + '.nii.gz')
                write_itk_image(label, output + '/' + 'train/label/' + prefix1 + '.nii.gz')
                subject_files = [str(output + '/' + 'train/data/' + prefix + '.nii.gz'),
                                 str(output + '/' + 'train/label/' + prefix1 + '.nii.gz')]
            elif flag == 'val':
                write_itk_image(data, output + '/' + 'val/data/' + prefix + '.nii.gz')
                write_itk_image(label, output + '/' + 'val/label/' + prefix1 + '.nii.gz')
                subject_files = [str(output + '/' + 'val/data/' + prefix + '.nii.gz'),
                                 str(output + '/' + 'val/label/' + prefix1 + '.nii.gz')]
        training_data_files.append(tuple(subject_files))
    return training_data_files


def lr_schedule(epoch,initial_lr=1e-4):
    lr=initial_lr*(0.99**(epoch/10))
    return lr


def parse_args(path, name,path1,name1):
    parser = argparse.ArgumentParser(description='Train/Finetune 3D U-Net on NIFTI volumes')
    parser.add_argument('--epochs', dest='epochs', type=int, default=600,
                        help='total number of training epochs')
    parser.add_argument('--c_epochs', dest='c_epochs', type=int, default=520,
                        help='Starting epoch for continued training')
    parser.add_argument('--save-after', dest='save_after', type=int, default=10,
                        help='number of training epochs after which the model should be saved')
    parser.add_argument('--modelFolder', dest='model_folder', type=str,
                        default=path + 'model/' + name,
                        help='folder where models will be saved')
    parser.add_argument('--modelFolder_re', dest='model_folder_re', type=str,
                        default=path1 + 'model/' + name1,
                        help='folder where the pretrain models will be saved')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', dest='decay', type=float, default=0.99,
                        help='learning rate decay per epoch')
    parser.add_argument('--weighted-cost', dest='weighted_cost', action='store_true',
                        help='Whether to use weighted cost or not (default: False)')
    parser.add_argument('--refine', dest='refine', default=False,
                        help='Whether to refine the pretrained model')
    args = parser.parse_args()

    return args


def main(overwrite=False):

    path ='/home/xdyang/vesselsegmentation/3dunet_results/from_scratch_test6_last/'
    name ='all' #'#17_#7_#18_#16'
    path1='/home/xdyang/vesselsegmentation/3dunet_results/pretrain_test6/' # we have tried transfer-learning
    name1='other_data'
    args = parse_args(path, name,path1,name1)
    n_epochs = args.epochs
    c_epochs=args.c_epochs
    decay = args.decay
    weighted_cost = args.weighted_cost
    model_folder = args.model_folder
    model_folder_re=args.model_folder_re
    save_after = args.save_after
    re=args.refine

    if overwrite or not os.path.exists(config["data_file"]):
        config["image_shape"] = get_image_shape(config["inputFns"], config["labelFns"], n_in=config["nb_channels"])
        training_data_files = image_padding(config["inputFns"], config["labelFns"], config["output"],
                                            n_in=config["nb_channels"],
                                            image_shape=config["image_shape"], flag='train')
        write_data_to_file(training_data_files, config["data_file"], image_shape=config["image_shape"],truth_dtype=np.float32,normalize=False)
    data_file_opened = open_data_file(config["data_file"])


    if c_epochs==0 and re==False:
        # model = unet_test17()
        model=unet_test6()
        lr = args.learning_rate
        config["basemodel"]=None
    if c_epochs != 0 and re == False:
        config["basemodel"] = os.path.abspath(os.path.join(model_folder, 'model_last' + '.h5'))
        model = load_model(config["basemodel"], custom_objects={'soft_dice': soft_dice, 'metric': dice(),'InstanceNormalization': InstanceNormalization})
        lr=lr_schedule(c_epochs,initial_lr=args.learning_rate)

    if c_epochs==0 and re==True:
        config["basemodel"] = os.path.abspath(os.path.join(model_folder_re, 'model_last' + '.h5'))
        model = load_model(config["basemodel"], custom_objects={'soft_dice': soft_dice, 'metric': dice(),
                                                                'InstanceNormalization': InstanceNormalization})
        lr = args.learning_rate
    if c_epochs!=0 and re==True:
        config["basemodel"] = os.path.abspath(os.path.join(model_folder, 'model_last' + '.h5'))
        model = load_model(config["basemodel"], custom_objects={'soft_dice': soft_dice, 'metric': dice(),
                                                                'InstanceNormalization': InstanceNormalization})
        lr = lr_schedule(c_epochs, initial_lr=args.learning_rate)
    print(model.summary())

    rest_epochs=n_epochs-c_epochs
    n_iters = int(rest_epochs/ save_after)
    iters = [save_after for i in range(n_iters)]

    if rest_epochs % save_after > 0:
        iters.append(rest_epochs % save_after)

    print('..............................')
    print('Training Parameters')
    print('..............................')
    print('learning-rate:', lr)
    print('decay:', decay)
    print('Number of epochs:', n_epochs)
    print('start of epochs:', c_epochs)
    print('Batch size:',  config["batch_size"])
    print('Base Model:', config["basemodel"])
    print('Model save folder:', model_folder)
    print('save model after every', save_after, 'epoch(s)')
    print('...................................................\n \n')

    # get training and testing generators
    if config["validation_split"] is not None:
        train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
            data_file_opened,
            batch_size=config["batch_size"],
            data_split=config["validation_split"],
            overwrite=overwrite,
            validation_keys_file=config["validation_file"],
            training_keys_file=config["training_file"],
            n_labels=config["n_labels"],
            labels=config["labels"],
            patch_shape=config["patch_shape"],
            validation_batch_size=config["validation_batch_size"],
            validation_patch_overlap=config["validation_patch_overlap"],
            training_patch_start_offset=config["training_patch_start_offset"],
            permute=config["permute"],
            augment=config["augment"],
            skip_blank=config["skip_blank"],
            augment_flip=config["flip"],
            augment_distortion_factor=config["distort"])
    else:
        train_generator, n_train_steps = get_training_and_validation_generators(
            data_file_opened,
            batch_size=config["batch_size"],
            data_split=config["validation_split"],
            overwrite=overwrite,
            validation_keys_file=config["validation_file"],
            training_keys_file=config["training_file"],
            n_labels=config["n_labels"],
            labels=config["labels"],
            patch_shape=config["patch_shape"],
            validation_batch_size=config["validation_batch_size"],
            validation_patch_overlap=config["validation_patch_overlap"],
            training_patch_start_offset=config["training_patch_start_offset"],
            permute=config["permute"],
            augment=config["augment"],
            skip_blank=config["skip_blank"],
            augment_flip=config["flip"],
            augment_distortion_factor=config["distort"])

    if c_epochs==0:
        H_epoch = []
        H_loss = []
        H_metric = []
    else:
        H_epoch = []
        H_loss = pickle.load(open(path + "record/" + name + "/loss.d", 'rb'))
        H_metric = pickle.load(open(path + "record/" + name + "/metric.d", 'rb'))

    for i, this_epochs in enumerate(iters):

        model, lr, h = train_model(model=model, train_data=train_generator, steps_per_epoch=n_train_steps,
                                   n_epochs=i * 10 + this_epochs+ c_epochs, lr=lr,
                                   weighted_cost=weighted_cost, model_folder=model_folder,initial_epoch=i * 10+c_epochs)
        this_model_fn = os.path.abspath(os.path.join(model_folder, 'model_last' + '.h5'))
        print('saving model1......')
        model.save(this_model_fn)
        if (i * 10 + this_epochs+c_epochs)%100==0:  # Saved each 100 epochs
            this_model_fn2 = os.path.abspath(os.path.join(model_folder, 'model'+str((i * 10 + this_epochs+c_epochs)/100) + '.h5'))
            print('saving model2......')
            model.save(this_model_fn2)
        print('lr=',lr)
        lr = lr * decay
        print('.....................................................................')
        print(h.history)
        print(h.epoch)
        print(h.history['loss'])
        # print(h.history['val_loss'])
        print(h.history['metric'])
        # print(h.history['val_metric'])
        h_epoch = h.epoch
        h_loss = h.history['loss']
        h_metric = h.history['metric']
        H_epoch = H_epoch + h_epoch
        H_loss = H_loss + h_loss
        H_metric = H_metric + h_metric
        pickle.dump(H_epoch, open(path + "record/" + name + "/epoch.d", 'wb'))
        pickle.dump(H_loss, open(path + "record/" + name + "/loss.d", 'wb'))
        pickle.dump(H_metric, open(path + "record/" + name + "/metric.d", 'wb'))
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(H_metric, label='training dice')
        plt.title('model dice')
        plt.ylabel('dice')
        plt.xlabel('epoch')
        plt.legend(loc='lower right')
        plt.subplot(2, 1, 2)
        plt.plot(H_loss, label='training loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.show()
        fig.savefig(path + "record/" + name + "/result.jpg")
    data_file_opened.close()

if __name__ == "__main__":
    main(overwrite=config["overwrite"])
