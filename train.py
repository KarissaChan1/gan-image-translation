import tensorflow as tf
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import sys
from CycleGAN import define_generator, define_discriminator, CycleGan
from generators import DataGenerator, RealDataGenerator
from callbacks import GANMonitor, LearningRateReducerCb, ComputeMetrics


experiment = 'experiment10_1_unpaired'
save_root = '/project/6044157/kchan/Image_Synthesis/experiments/' + experiment


tmpdataDir = sys.argv[1]
train_dir = tmpdataDir + 'train/'
val_dir = tmpdataDir + 'val/'
test_dir = tmpdataDir + 'test/'

model_training_dir = save_root+"/model/training/"
print(model_training_dir)
os.mkdir(save_root+"/model/")
os.mkdir(model_training_dir)

model_chkpt_dir = save_root+"/model/checkpoints/"
os.mkdir(model_chkpt_dir)
print(model_chkpt_dir)

model_imgs = save_root+"/model/training_images/"
os.mkdir(model_imgs)
print(model_imgs)


# generate training data
# Training parameters
EPOCHS = 100
BATCH_SIZE = 8
img_size = (256,256)
patch = (32,32)
n_channels = 1
shuffle = True
model='cycleGAN'

# Augmentation parameters
rotation_range = None
horizontal_flip = False
vertical_flip = False
width_shift_range = None #0.05
height_shift_range = None #0.05
shear_range = None #(-15,15)
scale_range = None #(0.9,1.1)

# training data paths
path1 = train_dir+'FLAIR/'
path2 = train_dir+'MD/'

params_aug = {'batch_size': BATCH_SIZE,
              'n_channels': n_channels,
              'dim': img_size,
              'model':model,
              'patch': patch,
              'shuffle': shuffle,
              'rotation_range': rotation_range,
              'horizontal_flip': horizontal_flip,
              'vertical_flip': vertical_flip,
              'width_shift_range': width_shift_range,
              'height_shift_range': height_shift_range,
              'shear_range': shear_range,
              'scale_range': scale_range}
              
              
train_datagen = DataGenerator(path1,path2,**params_aug)


#DEFINE MODEL
# Get the generators
image_shape = (256,256,1)
gen_G = define_generator(image_shape,name="generator_G")
gen_F = define_generator(image_shape,name="generator_F")

# Get the discriminators
disc_X = define_discriminator(image_shape,name="discriminator_X")
disc_Y = define_discriminator(image_shape,name="discriminator_Y")

#TRAINING
from tensorflow.keras.callbacks import CSVLogger

# Loss function for evaluating adversarial loss
adv_loss_fn = tf.keras.losses.MeanSquaredError()

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

# Define the loss function for the discriminators (added label smoothing)
def discriminator_loss_fn(real, fake):            
    real_loss = adv_loss_fn(tf.random.uniform(tf.shape(real), minval=0.9, maxval=1), real)
    fake_loss = adv_loss_fn(tf.random.uniform(tf.shape(fake), minval=0, maxval=0.1), fake)
    return (real_loss + fake_loss) * 0.5


# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4, beta_1=0.5),
    gen_F_optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4, beta_1=0.5),
    disc_X_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
    disc_Y_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)

# Callbacks
plotter = GANMonitor(path=val_dir+'FLAIR/', save_path = model_imgs)

rad_InceptionV3 = tf.keras.models.load_model(".../RadImageNet-InceptionV3_notop.h5")
rad_inceptionv3_modified = tf.keras.models.Sequential()
rad_inceptionv3_modified.add(rad_InceptionV3)
rad_inceptionv3_modified.add(tf.keras.layers.GlobalAveragePooling2D())   
        
# get real data for FID
params_fid= {'batch_size': 256,
              'n_channels': n_channels,
              'dim': img_size,
              'model': 'inception'}
scaled_real_imgs = RealDataGenerator(train_dir+'MD/',**params_fid)

metrics = ComputeMetrics(path_input = val_dir+'FLAIR/',
                        scaled_real_imgs = scaled_real_imgs,
                        inception_model = rad_inceptionv3_modified,
                        num_fake_img = len(os.listdir(val_dir+'FLAIR/'))
                        )

checkpoint_filepath = model_chkpt_dir+"cyclegan_checkpoints.{epoch:03d}"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only = True
)

csv_logger = CSVLogger(model_training_dir+"training.log")


history = cycle_gan_model.fit_generator(
    train_datagen,
    epochs=EPOCHS,
    callbacks=[plotter, metrics, csv_logger, model_checkpoint_callback,LearningRateReducerCb(total_epochs=EPOCHS)],
)

cycle_gan_model.save(save_root+"/model/trained_model_{}".format(EPOCHS),save_format="tf")
