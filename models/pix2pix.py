import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import tensorflow_addons as tfa


def downsample(filters, size, apply_batchnorm=False,apply_spectralnorm=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    if apply_spectralnorm:
        result.add(
          tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
        )
    else:
        result.add(
         tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False)
        )
        
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    else:
        result.add(tfa.layers.InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result
  
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

#     result.add(tf.keras.layers.BatchNormalization())
    result.add(tfa.layers.InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result
  
  
  
def Generator(image_shape):
    inputs = tf.keras.layers.Input(shape=image_shape)
    OUTPUT_CHANNELS = image_shape[2]
 
    # Encoder 256x256 -> 1x1
    down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
    ]
    # Decoder 1x1 -> 128x128
    up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
    ]
    # Last Decoder Layer 128x128 -> 256x256
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
   
def Discriminator(image_shape):
    initializer = tf.random_normal_initializer(0., 0.02)

    input = (
        tf.keras.layers.Input(shape=image_shape, name='input_image', dtype=tf.float32), 
        tf.keras.layers.Input(shape=image_shape, name='target_image', dtype=tf.float32),
        )
 
    x = tf.concat([input[0],input[1]],axis=-1)

    down1 = downsample(64, 4,apply_batchnorm=False, apply_spectralnorm=False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4,apply_spectralnorm=False)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4,apply_spectralnorm=False)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
#     conv = d = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(512, 4, strides=1,
#                                 kernel_initializer=initializer,
#                                 use_bias=False))(zero_pad1) # (bs, 31, 31, 512)
    conv = d = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

#     batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    batchnorm1 = tfa.layers.InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

#     last = d = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(1, 4, strides=1,
#                                 kernel_initializer=initializer, activation='sigmoid'))(zero_pad2) # (bs, 30, 30, 1)

    last = d = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer, activation='sigmoid')(zero_pad2) # (bs, 30, 30, 1)
    return tf.keras.Model(inputs=input, outputs=last)
  

def generator_loss(disc_generated_output, gen_output, target):
    LAMBDA = 100
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss
  
    
def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss
  
class pix2pix(tf.keras.Model):
    def __init__(
        self,
        generator_G,
        discriminator_X,
    ):
        super(pix2pix, self).__init__()
        self.gen_G = generator_G
        self.disc_X = discriminator_X

    def call(self, inputs):
#         fig,ax = plt.subplots(1,2)
#         ax[0].imshow(tf.reshape(inputs[0],(256,256,1)))
#         ax[1].imshow(tf.reshape(inputs[1],(256,256,1)))
#         print('X: ',tf.shape(inputs[0]))
#         print('Y: ',tf.shape(inputs[1]))
#         in2 = tf.reshape(inputs,(256,256,1))
#         plt.imshow(in2)
        return (
            self.gen_G(inputs[0]),
            self.disc_X(inputs),
        )

    def compile(
        self,
        gen_G_optimizer,
        disc_X_optimizer,
        gen_loss_fn,
        disc_loss_fn,

    ):
        super(pix2pix, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn


    def train_step(self,batch_data):
        
        input_image, target = batch_data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.gen_G(input_image[0], training=True)

            disc_real_output = self.disc_X([input_image[0], input_image[1]], training=True) 
            disc_generated_output = self.disc_X([input_image[0], gen_output], training=True)
#             disc_real_output = self.disc_X(concatenated, training=True) 
#             concatenated2 = tf.concat([input_image, gen_output],axis=-1)
#             disc_generated_output = self.disc_X(concatenated2, training=True)
    
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss_fn(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss_fn(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                              self.gen_G.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   self.disc_X.trainable_variables)

        self.gen_G_optimizer.apply_gradients(zip(generator_gradients,
                                              self.gen_G.trainable_variables))
        self.disc_X_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  self.disc_X.trainable_variables))

        return {
            'gen_total_loss': gen_total_loss, 
            'gen_gan_loss': gen_gan_loss, 
            'gen_l1_loss': gen_l1_loss, 
            'disc_loss': disc_loss, 
        }