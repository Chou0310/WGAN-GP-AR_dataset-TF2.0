import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from network.network_block import *
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow_addons.layers import InstanceNormalization

def get_generator_model(noise_dim = 128):
    noise = Input(shape=(noise_dim,))
    x = Dense(4 * 4 * 256, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((4, 4, 256))(x)
    #color img
    up_size = (2,2)
    x = UpSampling2D(up_size)(x)
    x = Conv2D(512, kernel_size=(3,3), padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.3)(x)
    x = UpSampling2D(up_size)(x)
    x = Conv2D(256, kernel_size=(3,3), padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.3)(x)
    x = UpSampling2D(up_size)(x)
    x = Conv2D(128, kernel_size=(3,3), padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.3)(x)

    x = UpSampling2D(up_size)(x)
    x = Conv2D(64, kernel_size=(3,3), padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.3)(x)

    x = UpSampling2D(up_size)(x)
    x = Conv2D(1, kernel_size=(3,3), padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.3)(x)
    x = Conv2D(1, kernel_size=(3,3), padding='same')(x)
    gen_img = Activation('tanh', name='gen_img')(x)
    g_model = Model([noise], [gen_img], name="generator")
    g_model.summary()
    return g_model


def get_discriminator_model(IMG_SHAPE=(128, 128, 1)):
    img_input = Input(shape=IMG_SHAPE)
    x = Conv2D(64, kernel_size=(5,5), strides=2)(img_input)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.3)(x)
    x = Conv2D(128, kernel_size=(5,5), strides=2)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.3)(x)
    x = Conv2D(256, kernel_size=(3,3), strides=2)(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(0.3)(x)
    x = Conv2D(512, kernel_size=(3,3), strides=2, padding='same', name='last_conv')(x)
    x = LayerNormalization()(x)
    last_conv_layer = LeakyReLU(0.3)(x)

    D_patch = Conv2D(1, kernel_size=(3,3), strides=1, name='D_patch')(last_conv_layer)
    d_model = Model(img_input, [D_patch], name="discriminator")
    d_model.summary()
    return d_model


if __name__ == '__main__':
    G = get_generator_model()
    D = get_discriminator_model()