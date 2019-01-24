import numpy as np
from keras.layers import *
from keras.models import Model
from keras.initializers import *

def DC_generator(output_shape, latent_dim=100, nb_blocks=4, nb_filters=32):
    input_zz = Input(shape=(latent_dim,))

    # x = Dense(1024)(input_zz)
    # x = Activation("relu")(x)

    reshape_dim1 = output_shape[0] // 2**nb_blocks
    reshape_dim2 = output_shape[1] // 2**nb_blocks
    reshape_dim3 = nb_filters * 2**(nb_blocks - 1)

    x = Dense(reshape_dim1 * reshape_dim2 * reshape_dim3)(input_zz)
    x = Activation("relu")(x)
    x = Reshape((reshape_dim1, reshape_dim2, reshape_dim3))(x)

    filter_iter = reshape_dim3
    for b in range(nb_blocks):
        x = __up_conv_block(x, filter_iter, 5)
        filter_iter //= 2

    x = Conv2D(output_shape[-1], 3, strides=1, padding="same")(x)
    x = Activation("tanh")(x)

    model = Model(input_zz, x)

    return model

def DC_discriminator(input_shape, nb_blocks=4, nb_filters=32):
    input_img = Input(shape=input_shape)

    x = input_img
    filter_iter = nb_filters
    for b in range(nb_blocks):
        x = __down_conv_block(x, filter_iter, 3)
        filter_iter *= 2

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation("linear")(x)

    model = Model(input_img, x)

    return model

def __up_conv_block(x, nb_filters, kernel, padding="same"):
    weight_init = RandomNormal(mean=0., stddev=0.02)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(nb_filters, kernel, strides=1, padding=padding, kernel_initializer=weight_init)(x)
    x = Activation('relu')(x)
    return x

def __down_conv_block(x, nb_filters, kernel, padding="same"):
    weight_init = RandomNormal(mean=0., stddev=0.02)
    # x = Conv2D(nb_filters, kernel, strides=2, padding=padding, kernel_initializer=weight_init)(x)
    x = Conv2D(nb_filters, kernel, strides=1, padding=padding, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D()(x)
    return x
