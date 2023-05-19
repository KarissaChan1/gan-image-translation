import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import tensorflow_addons as tfa
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

def resnet_block(n_filters, input_layer):
	# weight initialization
	init = tf.random_normal_initializer(0., 0.02)
	g = tf.keras.layers.Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = tf.keras.layers.Activation('relu')(g)
	g = tf.keras.layers.Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = tf.keras.layers.Concatenate()([g, input_layer])
	return g
 
 
def define_generator(image_shape, n_resnet=9,name=None):
	# weight initialization
	init = tf.random_normal_initializer(0., 0.02)
	# image input
	in_image = tf.keras.layers.Input(shape=image_shape)
 
  # Encoder
	g = tf.keras.layers.Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = tf.keras.layers.Activation('relu')(g)
	g = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = tf.keras.layers.Activation('relu')(g)
	g = tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = tf.keras.layers.Activation('relu')(g)

  # Transformer
	for _ in range(n_resnet):
		g = resnet_block(256, g)
  
  # Decoder
	g = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = tf.keras.layers.Activation('relu')(g)
	g = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = tf.keras.layers.Activation('relu')(g)
	g = tf.keras.layers.Conv2D(1, (7,7), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
 
	out_image = tf.keras.layers.Activation('tanh')(g)
	
  # define model
	model = tf.keras.models.Model(in_image, out_image)
	return model
 
 

def define_discriminator(image_shape, name=None):
	# weight initialization
	init = tf.random_normal_initializer(0., 0.02)
	# source image input
	in_image = tf.keras.layers.Input(shape=image_shape)
	d = tf.keras.layers.GaussianNoise(0.1)(in_image)

	d = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(d)
	d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

	d = tf.keras.layers.GaussianNoise(0.1)(d)    
	d = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(d)
	d = tfa.layers.InstanceNormalization(axis=-1)(d)
	d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

	d = tf.keras.layers.GaussianNoise(0.1)(d)
	d = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(d)
	d = tfa.layers.InstanceNormalization(axis=-1)(d)
	d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

	d = tf.keras.layers.GaussianNoise(0.1)(d)
	d = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init))(d)
	d = tfa.layers.InstanceNormalization(axis=-1)(d)
	d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

  # patch output
	d = tf.keras.layers.GaussianNoise(0.1)(d)
	patch_out = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init))(d)
    # define model
	model = tf.keras.models.Model(in_image, patch_out)
	# compile model
	model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model
 
 
class CycleGan(tf.keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def call(self, inputs):
        return (
            self.disc_X(inputs),
            self.disc_Y(inputs),
            self.gen_G(inputs),
            self.gen_F(inputs),
        )

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,

    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()


    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # flair to fake fa
            fake_y = self.gen_G(real_x, training=True)
            # fa to fake flair
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (flair to fake fa to fake flair): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (fa to fake flair to fake fa) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }