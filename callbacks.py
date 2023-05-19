import tensorflow as tf
import numpy as np
import os
from utils import scale_images, normalize_vol
from evaluation_metrics import calculate_fid
import matplotlib.pyplot as plt

class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, path ,save_path, num_img=15):
        self.path = path
        self.save_path = save_path
        self.list_IDs = os.listdir(self.path)
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(15, 2, figsize=(8, 30))
        for i in range(self.num_img): 
            img = np.load(self.path + self.list_IDs[i])
            
            img_input = tf.reshape(img, (-1, 256, 256, 1))
            prediction = self.model.gen_G(img_input)
            prediction2 = tf.reshape(prediction,(256,256))
            
            print(self.path + self.list_IDs[i])
            ax[i, 0].imshow(img,cmap='gray')
            ax[i, 1].imshow(prediction2,cmap='gray')
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

        plt.savefig(self.save_path+'epoch_{epoch}.png'.format(epoch=epoch+1))
        plt.show()
        plt.close()
        
class LearningRateReducerCb(tf.keras.callbacks.Callback):

    def __init__(self, total_epochs=100):
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs={}):
        old_lr_gen = self.model.gen_G_optimizer.lr.read_value()
        old_lr_disc = self.model.disc_X_optimizer.lr.read_value()
        if epoch <=int(self.total_epochs/2):
            new_lr_gen = old_lr_gen
            new_lr_disc = old_lr_disc  
        else:
            new_lr_gen = old_lr_gen - (old_lr_gen/(self.total_epochs/2))
            new_lr_disc = old_lr_disc - (old_lr_disc/(self.total_epochs/2))
        print("\nEpoch: {}. Reducing Disc Learning Rate from {} to {}".format(epoch+1, old_lr_disc, new_lr_disc))
        print("\nEpoch: {}. Reducing Gen Learning Rate from {} to {}".format(epoch+1, old_lr_gen, new_lr_gen))
        self.model.gen_G_optimizer.lr.assign(new_lr_gen)
        self.model.gen_F_optimizer.lr.assign(new_lr_gen)
        self.model.disc_X_optimizer.lr.assign(new_lr_disc)
        self.model.disc_Y_optimizer.lr.assign(new_lr_disc)
        
class LearningRateReducerCb_pix2pix(tf.keras.callbacks.Callback):

    def __init__(self, total_epochs=100):
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs={}):
        old_lr_gen = self.model.gen_G_optimizer.lr.read_value()
        old_lr_disc = self.model.disc_X_optimizer.lr.read_value()
        if epoch <=int(self.total_epochs/2):
            new_lr_gen = old_lr_gen
            new_lr_disc = old_lr_disc  
        else:
            new_lr_gen = old_lr_gen - (old_lr_gen/(self.total_epochs/2))
            new_lr_disc = old_lr_disc - (old_lr_disc/(self.total_epochs/2))
        print("\nEpoch: {}. Reducing Disc Learning Rate from {} to {}".format(epoch+1, old_lr_disc, new_lr_disc))
        print("\nEpoch: {}. Reducing Gen Learning Rate from {} to {}".format(epoch+1, old_lr_gen, new_lr_gen))
        self.model.gen_G_optimizer.lr.assign(new_lr_gen)
        self.model.disc_X_optimizer.lr.assign(new_lr_disc)


        
class ComputeMetrics(tf.keras.callbacks.Callback):

    def __init__(self, path_input, scaled_real_imgs, inception_model, num_fake_img=256, new_shape=(224,224,3)):
        self.path_input = path_input
        self.num_fake_img = num_fake_img
        self.list_fake_IDs = os.listdir(self.path_input)
        self.scaled_real_imgs = scaled_real_imgs
        self.new_shape = new_shape
        self.inception_model = inception_model

    def on_epoch_end(self, epoch, logs):
        fake_imgs = []
        for i in range(self.num_fake_img):
  
            img_input = np.load(self.path_input + self.list_fake_IDs[i])
            img_input = normalize_vol(img_input,-1,1)
            img_input = tf.reshape(img_input, (-1, 256, 256, 1))
            fake = self.model.gen_G(img_input)
            fake = tf.reshape(fake,(256,256,1))
            fake = tf.image.convert_image_dtype(fake, tf.float32)
            fake_scaled = scale_images(fake.numpy(),self.new_shape)
            fake_imgs.append(fake_scaled)

        #compute fid
        fake_imgs = np.asarray(fake_imgs)
        fid = calculate_fid(self.inception_model,self.scaled_real_imgs,fake_imgs)
        logs['fid'] = fid
        print("Epoch: {}. FID: {}".format(epoch+1,fid))