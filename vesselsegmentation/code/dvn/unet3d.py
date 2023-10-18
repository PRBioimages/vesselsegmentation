from keras.models import *
from keras.layers import *

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

# """3dunet"""""""
def unet_test6(pretrained_weights=None, input_size=(1,192,192,48)):
    inputs = Input(input_size)
    conv1 = Conv3D(32, 3, padding='same')(inputs)
    conv1= InstanceNormalization(axis=1)(conv1)
    conv1= Activation('relu')(conv1)

    conv1 = Conv3D(32, 3, padding='same')(conv1)
    conv1= InstanceNormalization(axis=1)(conv1)
    conv1= Activation('relu')(conv1)
    # (1,192,192,48)--> (32,192,192,48)

    pool1 =Conv3D(32,(2, 2, 1),strides=(2,2,1))(conv1)
    #(32,192,192,48)-> #(32,96,96,48)

    conv2 = Conv3D(64, 3, padding='same')(pool1)
    conv2= InstanceNormalization(axis=1)(conv2)
    conv2= Activation('relu')(conv2)

    conv2 = Conv3D(64, 3, padding='same')(conv2)
    conv2= InstanceNormalization(axis=1)(conv2)
    conv2= Activation('relu')(conv2)
    #(32,48,96,96)--> (64,48,96,96)

    pool2 = Conv3D(64,(2, 2, 1),strides=(2,2,1))(conv2)
    #(64,48,96,96)-->(64,48,48,48)

    conv3 = Conv3D(128, 3, padding='same')(pool2)
    conv3= InstanceNormalization(axis=1)(conv3)
    conv3= Activation('relu')(conv3)

    conv3 = Conv3D(128, 3, padding='same')(conv3)
    conv3=InstanceNormalization(axis=1)(conv3)
    conv3= Activation('relu')(conv3)
    # (64,48,48,48)-->(128,48,48,48)
    # drop3 = Dropout(0.5)(conv3)

    pool3 =Conv3D(128,(2, 2, 2),strides=(2,2,2))(conv3)
    # (128,48,48,48)-->(128,24,24,24)

    conv4 = Conv3D(256, 3, padding='same')(pool3)
    conv4= InstanceNormalization(axis=1)(conv4)
    conv4= Activation('relu')(conv4)

    conv4 = Conv3D(256, 3, padding='same')(conv4)
    conv4= InstanceNormalization(axis=1)(conv4)
    conv4= Activation('relu')(conv4)
    # (128,24,24,24)-->(256,24,24,24)

    pool4 = Conv3D(256,(2, 2, 2),strides=(2,2,2))(conv4)
    # (256, 24,24,24)-->(256,12,12,12)

    conv5 = Conv3D(320, 3, padding='same')(pool4)
    conv5 = InstanceNormalization(axis=1)(conv5)
    conv5 = Activation('relu')(conv5)

    conv5 = Conv3D(320, 3, padding='same')(conv5)
    conv5 = InstanceNormalization(axis=1)(conv5)
    conv5 = Activation('relu')(conv5)

    pool5 = Conv3D(320,(2, 2, 2),strides=(2,2,2))(conv5)
    # (256, 12,12,12)-->(320,6,6,6)

    conv6 = Conv3D(320, 3, padding='same')(pool5)
    conv6 = InstanceNormalization(axis=1)(conv6)
    conv6 = Activation('relu')(conv6)

    conv6 = Conv3D(320, 3, padding='same')(conv6)
    conv6 = InstanceNormalization(axis=1)(conv6)
    conv6 = Activation('relu')(conv6)
    # (320, 6,6,6)-->(320,6,6,6)

    up1 = Deconvolution3D(320,kernel_size=(2,2,2),strides=(2,2,2))(conv6)
    # up5 = Deconvolution3D(512,)
    #-->(320,12,12,12)
    merge1 = concatenate([conv5, up1], axis=1)
    #-->(320+320,12,12,12)

    conv7 = Conv3D(320, 3, padding='same')(merge1)
    conv7 =InstanceNormalization(axis=1)(conv7)
    conv7 = Activation('relu')(conv7)

    conv7 = Conv3D(320, 3, padding='same')(conv7)
    conv7 = InstanceNormalization(axis=1)(conv7)
    conv7 = Activation('relu')(conv7)
    #-->(320,12,12,12)


    up2= Deconvolution3D(320,kernel_size=(2,2,2),strides=(2,2,2))(conv7)
    #-->(320,24,24,24)
    merge2 = concatenate([conv4, up2], axis=1)
    #-->(320+256,24,24,24)

    conv8 = Conv3D(256, 3, padding='same')(merge2)
    conv8 = InstanceNormalization(axis=1)(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv3D(256, 3, padding='same')(conv8)
    #-->(256,24,24,24)
    conv8 = InstanceNormalization(axis=1)(conv8)
    conv8 = Activation('relu')(conv8)

    up3 = Deconvolution3D(256,kernel_size=(2,2,2),strides=(2,2,2))(conv8)
    #-->(256,48,48,48)
    merge3 = concatenate([conv3, up3], axis=1)
    #-->(256+128,48,48,48)


    conv9 = Conv3D(128, 3,padding='same')(merge3)
    conv9 = InstanceNormalization(axis=1)(conv9)
    conv9 = Activation('relu')(conv9)

    conv9= Conv3D(128, 3, padding='same')(conv9)
    #-->(128,48,48,48)
    conv9 = InstanceNormalization(axis=1)(conv9)
    conv9 = Activation('relu')(conv9)

    up4 = Deconvolution3D(128,kernel_size=(2,2,1),strides=(2,2,1))(conv9)
    # -->(128,48,96,96)
    merge4= concatenate([conv2, up4], axis=1)
    # -->(64+128,48,96,96)


    conv10 = Conv3D(64, 3, padding='same')(merge4)
    conv10 = InstanceNormalization(axis=1)(conv10)
    conv10 = Activation('relu')(conv10)

    conv10 = Conv3D(64, 3, padding='same')(conv10)
    # -->(64,48,96,96)
    conv10 = InstanceNormalization(axis=1)(conv10)
    conv10 = Activation('relu')(conv10)

    up5 = Deconvolution3D(64,kernel_size=(2,2,1),strides=(2,2,1))(conv10)
    # -->(64,48,192,192)
    merge5 = concatenate([conv1, up5], axis=1)
    # -->(64+32,48,192,192)


    conv11 = Conv3D(32, 3, padding='same')(merge5)
    conv11 = InstanceNormalization(axis=1)(conv11)
    conv11 = Activation('relu')(conv11)

    conv11 = Conv3D(32, 3, padding='same')(conv11)
    # -->(32,48,192,192)
    conv11 = InstanceNormalization(axis=1)(conv11)
    conv11 = Activation('relu')(conv11)

    conv12 = Conv3D(1, 1, activation='sigmoid')(conv11)
    #-->(1,48,192,192)
    model = Model(input=inputs, output=conv12)

    # model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

